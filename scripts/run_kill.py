"""Kill experiment runner & pre-registered gate for State-Free Modular Image Caching (zero-state-vlm-cache).

Usage:
  python scripts/run_kill.py run --model Qwen/Qwen3.5-2B --tasks docvqa chartqa mp_docvqa muirbench --n 150
  python scripts/run_kill.py report --out results/kill.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zero_state_vlm_cache.cache_ops import (  # noqa: E402
    encode_image_modular,
    encode_sink_prefix,
    find_visual_spans,
    forward_gdn_boundary_reset,
    forward_segmented_prefill,
    get_vlm_backbone,
    measure_cache_bytes,
    online_modular_prefill_and_decode,
)
from zero_state_vlm_cache.evals import (  # noqa: E402
    cuda_sync_timer,
    downsample_image_2x,
    load_vlm_task_items,
    paired_bootstrap_ci,
    score_prediction,
)

ARMS = ("full_joint", "gdn_boundary_reset", "zero_state_modular", "pooled_joint_2x2")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "kill.jsonl"


def _prepare_batch(processor: Any, images: list[Any], question: str, device: str) -> dict[str, torch.Tensor]:
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[prompt], images=images, return_tensors="pt")
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def _greedy_decode_from_cache(
    model: torch.nn.Module,
    last_logits: torch.Tensor,
    cache: Any,
    next_text_pos: int,
    next_3d_pos: torch.Tensor,
    max_new_tokens: int = 32,
) -> list[int]:
    _, lm, _ = get_vlm_backbone(model)
    device = last_logits.device
    next_tok = last_logits[:, -1:].argmax(dim=-1)
    gen_ids = [int(next_tok.item())]
    eos_id = getattr(model.config.get_text_config(), "eos_token_id", None)
    cur_t = next_text_pos
    cur_3d = next_3d_pos.clone()

    for _ in range(max_new_tokens - 1):
        if eos_id is not None and gen_ids[-1] == eos_id:
            break
        emb = model.get_input_embeddings()(next_tok)
        step_pos = torch.cat(
            [torch.tensor([[[cur_t]]], device=device, dtype=cur_3d.dtype), cur_3d],
            dim=0,
        )
        out = lm(inputs_embeds=emb, position_ids=step_pos, past_key_values=cache, use_cache=True)
        next_tok = model.lm_head(out.last_hidden_state[:, -1:]).argmax(dim=-1)
        gen_ids.append(int(next_tok.item()))
        cur_t += 1
        cur_3d = cur_3d + 1
    return gen_ids


@torch.no_grad()
def evaluate_item_arm(
    model: torch.nn.Module,
    processor: Any,
    item: dict[str, Any],
    arm: str,
    device: str,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    images = item["images"]
    if arm == "pooled_joint_2x2":
        images = [downsample_image_2x(im) for im in images]

    batch = _prepare_batch(processor, images, item["question"], device)
    input_ids = batch["input_ids"]
    pixel_values = batch.get("pixel_values")
    grid_thw = batch.get("image_grid_thw")
    mm_type = batch.get("mm_token_type_ids")
    if mm_type is None and hasattr(model.config, "image_token_id"):
        mm_type = (input_ids == model.config.image_token_id).int()

    if arm in ("full_joint", "pooled_joint_2x2"):
        t0 = cuda_sync_timer()
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            mm_token_type_ids=mm_type,
            use_cache=True,
        )
        t1 = cuda_sync_timer()
        ttft_ms = (t1 - t0) * 1000.0
        mem = measure_cache_bytes(out.past_key_values)

        vlm, _, _ = get_vlm_backbone(model)
        L = input_ids.shape[1]
        base_3d = torch.arange(L, L + 1, device=device).view(1, 1, 1).expand(3, 1, 1)
        if getattr(vlm, "rope_deltas", None) is not None:
            base_3d = base_3d + vlm.rope_deltas.to(device).unsqueeze(0)
        gen_ids = _greedy_decode_from_cache(model, out.logits, out.past_key_values, L, base_3d, max_new_tokens)

    elif arm == "gdn_boundary_reset":
        t0 = cuda_sync_timer()
        logits, cache = forward_gdn_boundary_reset(model, input_ids, pixel_values, grid_thw, mm_type)
        t1 = cuda_sync_timer()
        ttft_ms = (t1 - t0) * 1000.0
        mem = measure_cache_bytes(cache)
        vlm, _, _ = get_vlm_backbone(model)
        L = input_ids.shape[1]
        base_3d = torch.arange(L, L + 1, device=device).view(1, 1, 1).expand(3, 1, 1)
        if getattr(vlm, "rope_deltas", None) is not None:
            base_3d = base_3d + vlm.rope_deltas.to(device).unsqueeze(0)
        gen_ids = _greedy_decode_from_cache(model, logits, cache, L, base_3d, max_new_tokens)

    elif arm == "zero_state_modular":
        vs_id = model.config.vision_start_token_id
        ve_id = model.config.vision_end_token_id
        spans = find_visual_spans(input_ids, vs_id, ve_id)
        assert len(spans) == len(images), f"Expected {len(images)} visual spans, got {len(spans)}"

        sink_ids = input_ids[:, : spans[0][0]]
        sink_kv, _ = encode_sink_prefix(model, sink_ids)

        # Slice per-image pixel_values using grid_thw patch counts (T * H * W patches per image)
        patch_counts = [int(g.prod().item()) for g in grid_thw]
        pix_offset = 0
        image_kvs = []
        for idx_im, (s, e) in enumerate(spans):
            n_p = patch_counts[idx_im]
            im_pix = pixel_values[pix_offset : pix_offset + n_p]
            im_grid = grid_thw[idx_im : idx_im + 1]
            # Strip inter-image whitespace into adjacent question or keep contiguous image block
            next_s = spans[idx_im + 1][0] if idx_im + 1 < len(spans) else e
            seg_ids = input_ids[:, s:next_s]
            ikv, _ = encode_image_modular(model, sink_ids, seg_ids, im_pix, im_grid)
            image_kvs.append(ikv)
            pix_offset += n_p

        question_ids = input_ids[:, spans[-1][1] :]
        gen_ids, ttft_ms = online_modular_prefill_and_decode(
            model, sink_kv, image_kvs, question_ids, max_new_tokens=max_new_tokens
        )
        kv_b = sum(ikv.kv_bytes() for ikv in image_kvs)
        mem = {
            "kv_mb": kv_b / (1024**2),
            "recurrent_mb": 0.0,
            "total_mb": kv_b / (1024**2),
            "marconi_over_kv_ratio": 1.0,
        }
    else:
        raise ValueError(f"Unknown arm: {arm}")

    pred_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    score = score_prediction(item["metric"], pred_text, item["golds"])
    return {
        "score": float(score),
        "pred": pred_text,
        "ttft_ms": float(ttft_ms),
        "n_images": len(images),
        "target_page_pos": item.get("target_page_pos", "only"),
        **mem,
    }


def run(args: argparse.Namespace) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str, int, str]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["model"], r["task"], r["idx"], r["arm"]))

    print(f"[zero-state-vlm-cache] Loading {args.model} on {device} ({dtype})...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()

    manifest = {
        "project": "zero-state-vlm-cache",
        "model": args.model,
        "tasks": args.tasks,
        "n_per_task": args.n,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dtype": str(dtype),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ran_preflight = False
    t_run_start = time.perf_counter()
    completed_items = 0

    with out_path.open("a") as f:
        for task in args.tasks:
            items = load_vlm_task_items(task, args.n, seed=args.seed)
            print(f"  Loaded {len(items)} real items for task={task}", flush=True)
            for idx, item in enumerate(items):
                if not ran_preflight:
                    batch0 = _prepare_batch(processor, item["images"], item["question"], device)
                    mm0 = batch0.get("mm_token_type_ids")
                    if mm0 is None:
                        mm0 = (batch0["input_ids"] == model.config.image_token_id).int()
                    ref_out = model(
                        input_ids=batch0["input_ids"],
                        pixel_values=batch0.get("pixel_values"),
                        image_grid_thw=batch0.get("image_grid_thw"),
                        mm_token_type_ids=mm0,
                        use_cache=False,
                    ).logits
                    seg_logits, _ = forward_segmented_prefill(
                        model,
                        batch0["input_ids"],
                        batch0.get("pixel_values"),
                        batch0.get("image_grid_thw"),
                        mm0,
                        reset_at_vision_boundaries=False,
                    )
                    max_diff = float((ref_out[:, -1].float() - seg_logits[:, -1].float()).abs().max().item())
                    tol = 0.5 if dtype in (torch.bfloat16, torch.float16) else 1e-3
                    assert max_diff < tol, f"Pre-flight segmented-prefill mismatch: max_diff={max_diff:.4f} >= {tol}"
                    print(f"  [Pre-Flight PASS] Segmented prefill matches full_joint (max_logit_diff={max_diff:.5f})", flush=True)
                    ran_preflight = True

                for arm in ARMS:
                    if (args.model, task, idx, arm) in done_keys:
                        continue
                    res = evaluate_item_arm(model, processor, item, arm, device, max_new_tokens=args.max_new_tokens)
                    row = {"model": args.model, "task": task, "idx": idx, "arm": arm, **res}
                    f.write(json.dumps(row) + "\n")
                    f.flush()

                completed_items += 1
                if completed_items == 10:
                    elapsed = time.perf_counter() - t_run_start
                    total_est_h = (elapsed / 10.0) * (len(args.tasks) * args.n) / 3600.0
                    print(
                        f"  [Throughput Check @ 10 items] {elapsed:.1f}s -> Projected wall-clock: {total_est_h:.2f} GPU-h "
                        f"(~{total_est_h * 3.8:.1f} L4 CU)",
                        flush=True,
                    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_task: dict[tuple[str, str], dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        by_model_task[(r["model"], r["task"])][r["arm"]][r["idx"]] = r

    summary: dict[str, Any] = {}
    for (model, task), arms_map in sorted(by_model_task.items()):
        base = arms_map.get("full_joint", {})
        if not base:
            continue
        common_idxs = sorted(set.intersection(*(set(arms_map[a].keys()) for a in ARMS if a in arms_map)))
        if not common_idxs:
            continue

        ref_scores = [base[i]["score"] for i in common_idxs]
        ref_ttft = sum(base[i]["ttft_ms"] for i in common_idxs) / len(common_idxs)
        ref_mem = sum(base[i]["total_mb"] for i in common_idxs) / len(common_idxs)
        kv_mem = sum(base[i]["kv_mb"] for i in common_idxs) / len(common_idxs)

        entry: dict[str, Any] = {
            "n": len(common_idxs),
            "full_joint_acc": 100.0 * sum(ref_scores) / len(ref_scores),
            "full_joint_ttft_ms": ref_ttft,
            "marconi_total_mb": ref_mem,
            "kv_only_mb": kv_mem,
            "memory_reduction_ratio": (ref_mem / kv_mem) if kv_mem > 0 else 1.0,
        }
        for arm in ("gdn_boundary_reset", "zero_state_modular", "pooled_joint_2x2"):
            if arm not in arms_map:
                continue
            a_scores = [arms_map[arm][i]["score"] for i in common_idxs]
            a_ttft = sum(arms_map[arm][i]["ttft_ms"] for i in common_idxs) / len(common_idxs)
            drop_mean, drop_lo, drop_hi = paired_bootstrap_ci(ref_scores, a_scores)
            entry[f"{arm}_acc"] = 100.0 * sum(a_scores) / len(a_scores)
            entry[f"{arm}_drop_ci"] = (drop_mean, drop_lo, drop_hi)
            entry[f"{arm}_ttft_ms"] = a_ttft
            entry[f"{arm}_speedup"] = ref_ttft / max(a_ttft, 1e-6)

        # Stratified early-vs-late page check on multi-page tasks (excludes recency shortcut)
        for pos_tag in ("early", "late"):
            p_idxs = [i for i in common_idxs if base[i].get("target_page_pos") == pos_tag]
            if p_idxs and "zero_state_modular" in arms_map:
                entry[f"full_{pos_tag}_acc"] = 100.0 * sum(base[i]["score"] for i in p_idxs) / len(p_idxs)
                entry[f"modular_{pos_tag}_acc"] = (
                    100.0 * sum(arms_map["zero_state_modular"][i]["score"] for i in p_idxs) / len(p_idxs)
                )

        summary[f"{model}::{task}"] = entry
    return summary


def verdict(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pre-registered kill gate for zero-state-vlm-cache."""
    if not summary:
        return "INVALID", {"reason": "No completed task summaries found."}

    doc_drops_hi = []
    reset_drops = []
    modular_vs_pool_gains = []
    multi_speedups = []
    mem_ratios = []
    muir_drops = []

    for key, s in summary.items():
        _, task = key.split("::", 1)
        mem_ratios.append(s["memory_reduction_ratio"])
        if "gdn_boundary_reset_drop_ci" in s:
            reset_drops.append(s["gdn_boundary_reset_drop_ci"][0])
        if "zero_state_modular_drop_ci" in s and "pooled_joint_2x2_acc" in s:
            modular_vs_pool_gains.append(s["zero_state_modular_acc"] - s["pooled_joint_2x2_acc"])
        if task in ("docvqa", "chartqa", "mp_docvqa") and "zero_state_modular_drop_ci" in s:
            doc_drops_hi.append(s["zero_state_modular_drop_ci"][2])  # 95% CI upper bound!
        if task == "muirbench" and "zero_state_modular_drop_ci" in s:
            muir_drops.append(s["zero_state_modular_drop_ci"][0])
        if task in ("mp_docvqa", "muirbench") and "zero_state_modular_speedup" in s:
            multi_speedups.append(s["zero_state_modular_speedup"])

    max_reset_drop = max(reset_drops) if reset_drops else 0.0
    max_doc_ci_hi = max(doc_drops_hi) if doc_drops_hi else 999.0
    mean_pool_gain = sum(modular_vs_pool_gains) / max(1, len(modular_vs_pool_gains))
    mean_multi_speedup = sum(multi_speedups) / max(1, len(multi_speedups)) if multi_speedups else 1.0
    mean_mem_ratio = sum(mem_ratios) / max(1, len(mem_ratios))
    max_muir_drop = max(muir_drops) if muir_drops else 0.0

    details = {
        "max_doc_drop_95ci_upper_pts": round(max_doc_ci_hi, 3),
        "max_gdn_reset_drop_pts": round(max_reset_drop, 3),
        "mean_gain_over_pooled2x2_pts": round(mean_pool_gain, 3),
        "mean_multi_image_ttft_speedup": round(mean_multi_speedup, 3),
        "mean_memory_reduction_vs_marconi": round(mean_mem_ratio, 3),
        "max_muirbench_drop_pts": round(max_muir_drop, 3),
    }

    if max_reset_drop > 3.0 or (multi_speedups and mean_multi_speedup < 3.0):
        return "KILL", details
    if (
        max_doc_ci_hi <= 1.5
        and max_muir_drop <= 3.0
        and mean_pool_gain >= 6.0
        and (not multi_speedups or mean_multi_speedup >= 4.5)
        and mean_mem_ratio >= 5.0
    ):
        return "CONTINUE", details
    if max_reset_drop <= 1.5 and max_muir_drop > 3.0:
        return "PIVOT_MECHANISTIC", details
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
    pr.add_argument("--model", default="Qwen/Qwen3.5-2B")
    pr.add_argument("--tasks", nargs="+", default=["docvqa", "chartqa", "mp_docvqa", "muirbench"])
    pr.add_argument("--n", type=int, default=150)
    pr.add_argument("--max-new-tokens", type=int, default=24)
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

