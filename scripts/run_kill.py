"""Kill experiment runner & pre-registered gate for hybrid-kv-cliff.

Usage:
  python scripts/run_kill.py run \
    --models Qwen/Qwen3-1.7B-Base Qwen/Qwen3.5-2B-Base ibm-granite/granite-4.0-h-1b tiiuae/Falcon-H1-1.5B-Base \
    --tasks ruler_single ruler_multikey ruler_vt multi_turn_shift longbench_v2 \
    --length 8192 --n 50
  python scripts/run_kill.py report --out results/kill.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybrid_kv_cliff.compress import (  # noqa: E402
    ARMS_DEFAULT,
    compress_kv_cache,
    forward_post_prefill,
    get_architecture_stats,
    greedy_decode_compressed,
    prefill_with_salience,
)
from hybrid_kv_cliff.evals import (  # noqa: E402
    build_ruler_and_query_shift_items,
    load_longbench_v2_items,
    measure_decode_tok_per_s,
    paired_bootstrap_ci,
    score_item,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "kill.jsonl"


def _load_model_and_tokenizer(model_id: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device).eval()
    return model, tok


def _load_task_items(tokenizer: Any, task: str, n: int, length: int, seed: int) -> list[dict[str, Any]]:
    if task == "longbench_v2":
        return load_longbench_v2_items(tokenizer, n=n, target_length=length, seed=seed)
    return build_ruler_and_query_shift_items(tokenizer, task=task, n=n, target_length=length, seed=seed)


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str, int, int, str]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["model"], r["task"], int(r["target_length"]), int(r["idx"]), r["arm"]))

    manifest = {
        "project": "hybrid-kv-cliff",
        "models": args.models,
        "tasks": args.tasks,
        "arms": args.arms,
        "target_length": args.length,
        "n_per_task": args.n,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dtype": str(dtype),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with out_path.open("a") as f:
        for model_id in args.models:
            print(f"\n[hybrid-kv-cliff] Loading {model_id} on {device} ({dtype})...", flush=True)
            model, tok = _load_model_and_tokenizer(model_id, device, dtype)
            arch = get_architecture_stats(model)
            print(f"  Arch stats: {arch}", flush=True)

            ran_preflight = False
            t_model_start = time.perf_counter()
            docs_completed = 0

            for task in args.tasks:
                items = _load_task_items(tok, task, n=args.n, length=args.length, seed=args.seed)
                print(f"  Task={task}: prepared {len(items)} real items (target_length={args.length})", flush=True)
                for idx, item in enumerate(items):
                    missing_arms = [
                        a for a in args.arms if (model_id, task, args.length, idx, a) not in done_keys
                    ]
                    if not missing_arms:
                        continue

                    doc_ids = item["doc_ids"].to(device)
                    post_ids = item["post_ids"].to(device)
                    bundle = prefill_with_salience(model, doc_ids, obs_window=args.obs_window)

                    if not ran_preflight:
                        c_full = compress_kv_cache(model, bundle, "full")
                        c_1x = compress_kv_cache(model, bundle, "snapkv_1x")
                        l_full = forward_post_prefill(model, c_full, post_ids)
                        l_1x = forward_post_prefill(model, c_1x, post_ids)
                        diff = float((l_full.float() - l_1x.float()).abs().max().item())
                        tol = 0.25 if dtype in (torch.bfloat16, torch.float16) else 1e-4
                        assert diff <= tol, f"Pre-flight 1x equivalence failed on {model_id}: max_diff={diff}"
                        print(f"  [Pre-Flight PASS] snapkv_1x == full (max_logit_diff={diff:.6f})", flush=True)
                        ran_preflight = True

                    for arm in missing_arms:
                        comp = compress_kv_cache(model, bundle, arm)
                        gen_ids = greedy_decode_compressed(
                            model,
                            comp,
                            prompt_suffix_ids=post_ids,
                            initial_last_logits=bundle.last_logits,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred_text = tok.decode(gen_ids, skip_special_tokens=True)
                        sc = score_item(item["metric"], pred_text, item["gold"])

                        # Measure synchronized decode tok/s on idx == 0 for hardware Pareto reporting
                        tok_s = None
                        if idx == 0:
                            comp_speed = compress_kv_cache(model, bundle, arm)
                            dummy_tok = post_ids[:, -1:]
                            base_pos = comp_speed.doc_len
                            tok_s = measure_decode_tok_per_s(
                                lambda step: forward_post_prefill(
                                    model, comp_speed, dummy_tok, logical_start_pos=base_pos + step
                                ),
                                n_steps=16,
                                device=device,
                            )

                        row = {
                            "model": model_id,
                            "task": task,
                            "target_length": args.length,
                            "actual_doc_len": bundle.doc_len,
                            "idx": idx,
                            "arm": arm,
                            "score": float(sc),
                            "pred": pred_text[:80],
                            "gold": item["gold"],
                            "compressed_kv_bytes": comp.compressed_kv_bytes,
                            "full_kv_bytes": comp.full_kv_bytes,
                            "effective_kv_ratio": round(comp.effective_kv_ratio, 4),
                            "decode_tok_s": round(tok_s, 2) if tok_s is not None else None,
                            **arch,
                        }
                        f.write(json.dumps(row) + "\n")
                        f.flush()

                    docs_completed += 1
                    if docs_completed == 10:
                        elapsed = time.perf_counter() - t_model_start
                        est_model_h = (elapsed / 10.0) * (len(args.tasks) * args.n) / 3600.0
                        print(
                            f"  [Throughput Check @ 10 docs] {elapsed:.1f}s -> Est {est_model_h:.2f} GPU-h for {model_id}",
                            flush=True,
                        )

            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    arch_meta: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = (r["model"], r["task"], int(r["target_length"]))
        grouped[key][r["arm"]][int(r["idx"])] = r
        arch_meta[r["model"]] = {
            "num_total_layers": r.get("num_total_layers"),
            "num_attn_layers": r.get("num_attn_layers"),
            "attn_layer_ratio": r.get("attn_layer_ratio"),
            "gqa_groups": r.get("gqa_groups"),
            "is_hybrid": r.get("is_hybrid", False),
            "is_sparse_attn_hybrid": r.get("is_sparse_attn_hybrid", False),
        }

    summary: dict[str, Any] = {}
    for (model, task, length), arms_map in sorted(grouped.items()):
        if "full" not in arms_map:
            continue
        ref_map = arms_map["full"]
        common_idxs = sorted(ref_map.keys())
        ref_scores = [ref_map[i]["score"] for i in common_idxs]
        entry: dict[str, Any] = {
            "n": len(common_idxs),
            "full_acc": 100.0 * sum(ref_scores) / max(1, len(ref_scores)),
            "full_kv_mb": sum(ref_map[i]["full_kv_bytes"] for i in common_idxs) / max(1, len(common_idxs)) / (1024**2),
            **arch_meta[model],
        }
        for arm, a_map in sorted(arms_map.items()):
            if arm == "full":
                continue
            shared = [i for i in common_idxs if i in a_map]
            if not shared:
                continue
            r_sc = [ref_map[i]["score"] for i in shared]
            a_sc = [a_map[i]["score"] for i in shared]
            drop_mean, drop_lo, drop_hi = paired_bootstrap_ci(r_sc, a_sc)
            eff_ratio = sum(a_map[i]["effective_kv_ratio"] for i in shared) / len(shared)
            entry[f"{arm}_acc"] = 100.0 * sum(a_sc) / len(shared)
            entry[f"{arm}_drop_ci"] = (drop_mean, drop_lo, drop_hi)
            entry[f"{arm}_kv_ratio"] = round(eff_ratio, 3)

        # Direct paired bootstrap of (KIVI-4bit minus SnapKV-4x) gain on identical items
        if "kivi_4bit" in arms_map and "snapkv_4x" in arms_map:
            shared_ks = [i for i in common_idxs if i in arms_map["kivi_4bit"] and i in arms_map["snapkv_4x"]]
            if shared_ks:
                k_sc = [arms_map["kivi_4bit"][i]["score"] for i in shared_ks]
                s_sc = [arms_map["snapkv_4x"][i]["score"] for i in shared_ks]
                gain_mean, gain_lo, gain_hi = paired_bootstrap_ci(k_sc, s_sc)
                entry["kivi4_over_snapkv4_gain_ci"] = (gain_mean, gain_lo, gain_hi)

        summary[f"{model}::{task}::L{length}"] = entry
    return summary


def verdict(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pre-registered Validity + Cliff + Quantization-Rescue Gate for hybrid-kv-cliff."""
    if not summary:
        return "INVALID_PIPELINE", {"reason": "Empty summary."}

    pure_tf_snapkv4_drops: list[float] = []
    sparse_snapkv4_drops: list[float] = []
    sparse_shift_drops: list[float] = []
    sparse_kivi4_drops: list[float] = []
    sparse_kivi_gains: list[float] = []
    sparse_kivi_gain_ci_los: list[float] = []

    for key, s in summary.items():
        _, task, _ = key.split("::")
        if not s.get("is_hybrid", False):
            # Pure-Transformer control validity check on single-turn tasks
            if task != "multi_turn_shift" and "snapkv_4x_drop_ci" in s:
                pure_tf_snapkv4_drops.append(s["snapkv_4x_drop_ci"][0])
        elif s.get("is_sparse_attn_hybrid", False):
            if "snapkv_4x_drop_ci" in s:
                sparse_snapkv4_drops.append(s["snapkv_4x_drop_ci"][0])
                if task == "multi_turn_shift":
                    sparse_shift_drops.append(s["snapkv_4x_drop_ci"][0])
            if "kivi_4bit_drop_ci" in s:
                sparse_kivi4_drops.append(s["kivi_4bit_drop_ci"][0])
            if "kivi4_over_snapkv4_gain_ci" in s:
                g_mean, g_lo, _ = s["kivi4_over_snapkv4_gain_ci"]
                sparse_kivi_gains.append(g_mean)
                sparse_kivi_gain_ci_los.append(g_lo)

    # Gate 0: Pure-Transformer Control Validity Gate (prevents buggy SnapKV/RoPE from faking a cliff)
    max_pure_drop = max(pure_tf_snapkv4_drops) if pure_tf_snapkv4_drops else 0.0
    if pure_tf_snapkv4_drops and max_pure_drop > 4.0:
        return "INVALID_PIPELINE", {
            "reason": "Pure-Transformer control failed SnapKV-4x sanity gate (>4.0 pts drop).",
            "max_pure_transformer_snapkv4_drop_pts": round(max_pure_drop, 3),
        }

    mean_sparse_evict_drop = sum(sparse_snapkv4_drops) / max(1, len(sparse_snapkv4_drops))
    max_sparse_evict_drop = max(sparse_snapkv4_drops) if sparse_snapkv4_drops else 0.0
    max_shift_drop = max(sparse_shift_drops) if sparse_shift_drops else 0.0
    mean_kivi4_drop = sum(sparse_kivi4_drops) / max(1, len(sparse_kivi4_drops))
    max_kivi_gain = max(sparse_kivi_gains) if sparse_kivi_gains else 0.0
    max_kivi_gain_lo = max(sparse_kivi_gain_ci_los) if sparse_kivi_gain_ci_los else -999.0

    details = {
        "max_pure_transformer_snapkv4_drop_pts": round(max_pure_drop, 3),
        "mean_sparse_hybrid_snapkv4_drop_pts": round(mean_sparse_evict_drop, 3),
        "max_sparse_hybrid_snapkv4_drop_pts": round(max_sparse_evict_drop, 3),
        "max_query_shift_snapkv4_drop_pts": round(max_shift_drop, 3),
        "mean_sparse_hybrid_kivi4_drop_pts": round(mean_kivi4_drop, 3),
        "max_kivi4_over_snapkv4_gain_pts": round(max_kivi_gain, 3),
        "max_kivi4_over_snapkv4_ci95_lo_pts": round(max_kivi_gain_lo, 3),
    }

    cliff_observed = (max_sparse_evict_drop >= 12.0) or (max_shift_drop >= 15.0)
    kivi_rescues = (mean_kivi4_drop <= 2.5) and (max_kivi_gain >= 10.0) and (max_kivi_gain_lo > 0.0)

    if cliff_observed and kivi_rescues:
        return "CONTINUE", details
    if pure_tf_snapkv4_drops and max_sparse_evict_drop <= 3.0 and max_shift_drop <= 3.0:
        return "H0_NO_CLIFF", details
    return "KILL", details


def report(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    summ = summarize(rows)
    decision, details = verdict(summ)
    print(json.dumps({"summary": summ, "verdict": decision, "gate_metrics": details}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen3-1.7B-Base",
            "Qwen/Qwen3.5-2B-Base",
            "ibm-granite/granite-4.0-h-1b",
            "tiiuae/Falcon-H1-1.5B-Base",
        ],
    )
    pr.add_argument(
        "--tasks",
        nargs="+",
        default=["ruler_single", "ruler_multikey", "ruler_vt", "multi_turn_shift", "longbench_v2"],
    )
    pr.add_argument("--arms", nargs="+", default=list(ARMS_DEFAULT))
    pr.add_argument("--length", type=int, default=8192)
    pr.add_argument("--n", type=int, default=50)
    pr.add_argument("--obs-window", type=int, default=64)
    pr.add_argument("--max-new-tokens", type=int, default=16)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default=str(DEFAULT_OUT))

    prep = sub.add_parser("report")
    prep.add_argument("--out", default=str(DEFAULT_OUT))

    args = p.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()

