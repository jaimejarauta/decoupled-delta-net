"""Week-1 Kill Experiment runner & pre-registered gate evaluator for Top Pick C1:
Same-Subspace State Overwrite & Synchronized State-KV Rollback in Hybrid Recurrent Reasoning Models.

Usage:
  python scripts/run_kill_c1.py run --model Qwen/Qwen3.5-0.8B --dataset gsm8k --n 100
  python scripts/run_kill_c1.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybrid_rollback.cache_state import (  # noqa: E402
    capture_recurrent_snapshot,
    prefill_tokens,
    rollback_attention_only_kv,
    rollback_synchronized,
)
from hybrid_rollback.live_decode import live_decode_with_rollback  # noqa: E402
from hybrid_rollback.revolve import OnlineCheckpointPool  # noqa: E402
from hybrid_rollback.subspace_probe import (  # noqa: E402
    measure_recurrent_state_drift,
    measure_subspace_overlap,
)
from hybrid_rollback.tasks import (  # noqa: E402
    DeadEndItem,
    answers_match,
    detect_backtrack_trigger,
    load_dead_end_suite,
)

OUT_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "kill_c1.jsonl"


def _enc(tok, s: str, add_bos: bool = False) -> list[int]:
    ids = tok(s, add_special_tokens=False).input_ids
    if add_bos and tok.bos_token_id is not None:
        return [tok.bos_token_id] + ids
    return ids


def _tile_to_len(tok, unit_text: str, target_len: int) -> list[int]:
    unit = _enc(tok, unit_text) or [1]
    reps = (target_len // len(unit)) + 2
    return (unit * reps)[:target_len]


@torch.no_grad()
def greedy_decode_cached(
    model,
    pkv: DynamicCache,
    last_logits: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    generated: list[int] = []
    nxt = int(last_logits[0, -1].argmax(-1).item())
    generated.append(nxt)
    if eos_token_id is not None and nxt == eos_token_id:
        return generated
    device = last_logits.device
    for _ in range(max_new_tokens - 1):
        inp = torch.tensor([[nxt]], dtype=torch.long, device=device)
        out = model(input_ids=inp, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        nxt = int(out.logits[0, -1].argmax(-1).item())
        generated.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return generated


def eval_arm1_item(
    model,
    tok,
    item: DeadEndItem,
    L_dead: int,
    pool_cap: int = 8,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> list[dict]:
    """Run controlled Orthogonal vs. Same-Subspace pollution and all rollback arms on one item."""
    prompt_ids = _enc(tok, item.prompt_text, add_bos=True)
    step_ids_list = [_enc(tok, st) for st in item.valid_prefix_steps]
    cue_ids = _enc(tok, item.continuation_cue)
    lemma_ids = _enc(tok, item.negative_lemma)[:28]

    prefix_ids_flat = list(prompt_ids)
    step_boundaries = [len(prompt_ids)]
    for s_ids in step_ids_list:
        prefix_ids_flat.extend(s_ids)
        step_boundaries.append(len(prefix_ids_flat))
    t_a = len(prefix_ids_flat)
    prefix_t = torch.tensor([prefix_ids_flat], dtype=torch.long, device=device)
    cue_t = torch.tensor([cue_ids], dtype=torch.long, device=device)
    lemma_cue_t = torch.tensor([lemma_ids + cue_ids], dtype=torch.long, device=device)

    rows: list[dict] = []

    # 0. Clean upper bound (L = 0)
    t0 = time.perf_counter()
    pkv_clean, _ = prefill_tokens(model, prefix_t)
    snap_a_ref = capture_recurrent_snapshot(pkv_clean, t_a)
    pkv_clean, last_lg = prefill_tokens(model, cue_t, past_key_values=pkv_clean)
    gen_clean = greedy_decode_cached(model, pkv_clean, last_lg, max_new_tokens, tok.eos_token_id)
    rows.append(
        dict(
            exp="arm1",
            i=item.idx,
            L=0,
            pollution="none",
            arm="clean",
            ok=answers_match(tok.decode(gen_clean), item.ground_truth),
            replay_tokens=0,
            total_tokens=t_a + len(cue_ids) + len(gen_clean),
            wall_sec=time.perf_counter() - t0,
        )
    )

    for pol_kind, unit_txt in [("orthogonal", item.orthogonal_haystack_unit), ("same_subspace", item.same_subspace_dead_end_unit)]:
        dead_ids = _tile_to_len(tok, unit_txt, L_dead)
        dead_t = torch.tensor([dead_ids], dtype=torch.long, device=device)
        full_probe_t = torch.cat([prefix_t, dead_t], dim=1)

        try:
            ov = measure_subspace_overlap(model, full_probe_t, (0, len(prompt_ids)), (t_a, t_a + L_dead))
        except ValueError:
            ov = {"mean_overlap": 0.0, "enrichment_ratio": 1.0}

        # Helper to build step-checkpointed state up to t_a + polluted dead_t
        def _build_polluted(policy: str):
            pool = OnlineCheckpointPool(capacity=pool_cap, policy=policy)
            pkv = None
            cur = 0
            for bnd in step_boundaries:
                pkv, _ = prefill_tokens(model, prefix_t[:, cur:bnd], past_key_values=pkv)
                pool.push(capture_recurrent_snapshot(pkv, bnd))
                cur = bnd
            # Step through dead-end in ~64-token pseudo-steps so ring_fifo vs min_span is tested realistically
            step_sz = 64
            for ds in range(0, L_dead, step_sz):
                chunk = dead_t[:, ds : ds + step_sz]
                pkv, _ = prefill_tokens(model, chunk, past_key_values=pkv)
                pool.push(capture_recurrent_snapshot(pkv, t_a + min(L_dead, ds + step_sz)))
            return pkv, pool

        # 1. Unrolled polluted continuation (no rollback)
        t0 = time.perf_counter()
        pkv_u, _ = _build_polluted("zero")
        drift = measure_recurrent_state_drift(snap_a_ref, pkv_u)
        pkv_u, lg_u = prefill_tokens(model, lemma_cue_t, past_key_values=pkv_u)
        gen_u = greedy_decode_cached(model, pkv_u, lg_u, max_new_tokens, tok.eos_token_id)
        rows.append(
            dict(
                exp="arm1",
                i=item.idx,
                L=L_dead,
                pollution=pol_kind,
                arm="unrolled",
                ok=answers_match(tok.decode(gen_u), item.ground_truth),
                replay_tokens=0,
                total_tokens=t_a + L_dead + lemma_cue_t.shape[1] + len(gen_u),
                wall_sec=time.perf_counter() - t0,
                subspace_overlap=ov["mean_overlap"],
                enrichment_ratio=ov["enrichment_ratio"],
                state_rel_drift=drift["mean_rel_drift"],
            )
        )

        if pol_kind == "orthogonal":
            # Rollback mechanism arms only need evaluation on same_subspace dead-ends.
            continue

        # 2. Attention-Only KV Rollback (desynchronized S_b + KV_a)
        t0 = time.perf_counter()
        pkv_kv, _ = _build_polluted("zero")
        rollback_attention_only_kv(pkv_kv, t_a)
        pkv_kv, lg_kv = prefill_tokens(model, lemma_cue_t, past_key_values=pkv_kv)
        gen_kv = greedy_decode_cached(model, pkv_kv, lg_kv, max_new_tokens, tok.eos_token_id)
        rows.append(
            dict(
                exp="arm1",
                i=item.idx,
                L=L_dead,
                pollution=pol_kind,
                arm="attn_kv_only",
                ok=answers_match(tok.decode(gen_kv), item.ground_truth),
                replay_tokens=0,
                total_tokens=t_a + L_dead + lemma_cue_t.shape[1] + len(gen_kv),
                wall_sec=time.perf_counter() - t0,
            )
        )

        # 3. Dumbest Strong Baseline #1: Fresh-Turn Prompt Restart (loses valid steps [len(prompt_ids), t_a))
        t0 = time.perf_counter()
        restart_t = torch.tensor([prompt_ids + lemma_ids], dtype=torch.long, device=device)
        pkv_pr, lg_pr = prefill_tokens(model, restart_t)
        gen_pr = greedy_decode_cached(model, pkv_pr, lg_pr, max_new_tokens + (t_a - len(prompt_ids)), tok.eos_token_id)
        rows.append(
            dict(
                exp="arm1",
                i=item.idx,
                L=L_dead,
                pollution=pol_kind,
                arm="prompt_restart",
                ok=answers_match(tok.decode(gen_pr), item.ground_truth),
                replay_tokens=len(prompt_ids),
                total_tokens=t_a + L_dead + restart_t.shape[1] + len(gen_pr),
                wall_sec=time.perf_counter() - t0,
            )
        )

        # 4, 5, 6: Synchronized Rollback across zero (full_replay), ring_fifo, and min_span (sync_binomial)
        for arm_name, policy in [("full_replay", "zero"), ("ring_fifo", "ring_fifo"), ("sync_binomial", "min_span")]:
            t0 = time.perf_counter()
            pkv_s, pool_s = _build_polluted(policy)
            rb = rollback_synchronized(model, pkv_s, prefix_t, pool_s, target_pos=t_a)
            pkv_s, lg_s = prefill_tokens(model, lemma_cue_t, past_key_values=pkv_s)
            gen_s = greedy_decode_cached(model, pkv_s, lg_s, max_new_tokens, tok.eos_token_id)
            rows.append(
                dict(
                    exp="arm1",
                    i=item.idx,
                    L=L_dead,
                    pollution=pol_kind,
                    arm=arm_name,
                    ok=answers_match(tok.decode(gen_s), item.ground_truth),
                    replay_tokens=rb.replay_tokens,
                    pool_nbytes=rb.pool_nbytes,
                    total_tokens=t_a + L_dead + rb.replay_tokens + lemma_cue_t.shape[1] + len(gen_s),
                    wall_sec=time.perf_counter() - t0,
                )
            )
    return rows


def boot_diff(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI of mean(a) - mean(b)."""
    g = torch.Generator().manual_seed(seed)
    d = torch.tensor(a, dtype=torch.float32) - torch.tensor(b, dtype=torch.float32)
    idx = torch.randint(0, len(d), (n_boot, len(d)), generator=g)
    means = d[idx].mean(dim=1)
    return float(d.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


def summarize(rows: list[dict], L_gate: int = 1024) -> dict[str, dict]:
    by_model: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    replay_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    subspace_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        if r.get("exp") != "arm1":
            continue
        m = r["model"]
        if r["L"] == 0 and r["arm"] == "clean":
            by_model[m]["clean"][r["i"]] = float(r["ok"])
        elif r["L"] == L_gate:
            key = f"{r['arm']}:{r['pollution']}" if r["arm"] == "unrolled" else r["arm"]
            by_model[m][key][r["i"]] = float(r["ok"])
            replay_stats[m][key].append(float(r.get("replay_tokens", 0)))
            if "subspace_overlap" in r:
                subspace_stats[m][r["pollution"]].append(float(r["subspace_overlap"]))

    out: dict[str, dict] = {}
    for m, arms in by_model.items():
        ids = sorted(arms["clean"])
        if not ids:
            continue
        s: dict[str, object] = {"clean_acc": 100.0 * sum(arms["clean"].values()) / len(ids)}
        for k, vals in arms.items():
            common = [i for i in ids if i in vals]
            if common:
                s[f"acc_{k}"] = 100.0 * sum(vals[i] for i in common) / len(common)
        # Key paired contrasts
        pairs = [
            ("same_vs_ortho_drop", "unrolled:orthogonal", "unrolled:same_subspace"),
            ("desync_penalty", "sync_binomial", "attn_kv_only"),
            ("gain_over_unrolled", "sync_binomial", "unrolled:same_subspace"),
            ("gain_over_restart", "sync_binomial", "prompt_restart"),
        ]
        for label, hi, lo in pairs:
            common = [i for i in ids if i in arms.get(hi, {}) and i in arms.get(lo, {})]
            if common:
                s[label] = tuple(100.0 * x for x in boot_diff([arms[hi][i] for i in common], [arms[lo][i] for i in common]))
        if replay_stats[m]["full_replay"] and replay_stats[m]["sync_binomial"]:
            fr = sum(replay_stats[m]["full_replay"]) / len(replay_stats[m]["full_replay"])
            sb = max(1.0, sum(replay_stats[m]["sync_binomial"]) / len(replay_stats[m]["sync_binomial"]))
            s["replay_speedup_ratio"] = fr / sb
        for pol in ("orthogonal", "same_subspace"):
            if subspace_stats[m][pol]:
                s[f"overlap_{pol}"] = sum(subspace_stats[m][pol]) / len(subspace_stats[m][pol])
        out[m] = s
    return out


def verdict(summary: dict[str, dict], transformer_control: str | None = None) -> tuple[str, dict]:
    """Pre-registered Gate from Section 3 of `RESEARCH_DECISION_2026-10-26.md`:
      - Phenomenon Gate: (hybrid same_vs_ortho_drop - transformer_control same_vs_ortho_drop >= 8.0 pts)
                         OR (desync_penalty = sync_binomial - attn_kv_only >= 6.0 pts).
      - Mechanism Gate : gain_over_unrolled >= 4.0 pts AND gain_over_restart >= 2.5 pts
                         AND replay_speedup_ratio >= 3.5x.
    Returns ('PASS_FULL' | 'PIVOT_DESYNC' | 'KILL', per_model_flags).
    """
    ctrl_drop = 0.0
    if transformer_control and transformer_control in summary:
        ctrl_drop = summary[transformer_control].get("same_vs_ortho_drop", (0.0,))[0]

    flags = {}
    any_full, any_phenom = False, False
    for m, s in summary.items():
        if m == transformer_control:
            continue
        same_excess = s.get("same_vs_ortho_drop", (0.0,))[0] - ctrl_drop
        desync = s.get("desync_penalty", (0.0,))[0]
        phenom_ok = (same_excess >= 8.0) or (desync >= 6.0)
        mech_ok = (
            s.get("gain_over_unrolled", (0.0,))[0] >= 4.0
            and s.get("gain_over_restart", (0.0,))[0] >= 2.5
            and float(s.get("replay_speedup_ratio", 0.0)) >= 3.5
        )
        flags[m] = {"phenomenon_ok": phenom_ok, "mechanism_ok": mech_ok, "same_excess": same_excess, "desync": desync}
        any_phenom = any_phenom or phenom_ok
        any_full = any_full or (phenom_ok and mech_ok)

    if any_full:
        return "PASS_FULL", flags
    if any_phenom:
        return "PIVOT_DESYNC", flags
    return "KILL", flags


def cmd_run(args) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()

    items = load_dead_end_suite(args.dataset, n=args.n, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for item in items:
            if args.mode in ("arm1", "both"):
                for L_dead in args.lengths:
                    rows = eval_arm1_item(
                        model,
                        tok,
                        item,
                        L_dead=L_dead,
                        pool_cap=args.pool_cap,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )
                    for r in rows:
                        f.write(json.dumps({"model": args.model, "dataset": args.dataset, **r}) + "\n")
                    f.flush()
            if args.mode in ("arm2", "both"):
                p_ids = _enc(tok, item.prompt_text, add_bos=True)
                for arm in ("uninterrupted", "attn_kv_only", "prompt_restart", "full_replay", "sync_binomial"):
                    t0 = time.perf_counter()
                    res = live_decode_with_rollback(
                        model,
                        tok,
                        p_ids,
                        arm=arm,
                        max_total_compute_tokens=args.live_budget,
                        pool_cap=args.pool_cap,
                        temperature=0.6,
                        seed=args.seed + item.idx,
                        device=device,
                    )
                    f.write(
                        json.dumps(
                            {
                                "model": args.model,
                                "dataset": args.dataset,
                                "exp": "arm2",
                                "i": item.idx,
                                "arm": arm,
                                "ok": answers_match(res.text, item.ground_truth),
                                "surviving_tokens": res.surviving_tokens,
                                "total_tokens": res.total_compute_tokens,
                                "replay_tokens": res.replay_tokens,
                                "backtracks": res.backtracks_triggered,
                                "wall_sec": time.perf_counter() - t0,
                            }
                        )
                        + "\n"
                    )
                f.flush()
            print(f"[{args.model}] item {item.idx + 1}/{len(items)} done", flush=True)


def cmd_report(args) -> None:
    rows = [json.loads(line) for line in Path(args.out).open() if line.strip()]
    summ = summarize(rows, L_gate=args.gate_length)
    for m, s in summ.items():
        parts = [f"{k}={v[0]:.1f}[{v[1]:.1f},{v[2]:.1f}]" if isinstance(v, tuple) else f"{k}={v:.2f}" for k, v in s.items()]
        print(f"{m:38s} | " + "  ".join(parts))
    v, flags = verdict(summ, transformer_control=args.control_model)
    print(f"\nPRE-REGISTERED VERDICT (L={args.gate_length}): {v}\nPer-model gate breakdown: {flags}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--model", required=True)
    pr.add_argument("--mode", default="both", choices=["arm1", "arm2", "both"])
    pr.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "synthetic"])
    pr.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048])
    pr.add_argument("--n", type=int, default=100)
    pr.add_argument("--pool-cap", type=int, default=8)
    pr.add_argument("--max-new-tokens", type=int, default=256)
    pr.add_argument("--live-budget", type=int, default=2048)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--device", default=None)
    pr.add_argument("--out", default=str(OUT_DEFAULT))

    pp = sub.add_parser("report")
    pp.add_argument("--gate-length", type=int, default=1024)
    pp.add_argument("--control-model", default=None)
    pp.add_argument("--out", default=str(OUT_DEFAULT))

    parsed = p.parse_args()
    cmd_run(parsed) if parsed.cmd == "run" else cmd_report(parsed)
