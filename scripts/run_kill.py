"""Kill experiment runner & pre-registered gate for tsfm-scale-identifiability (`fev-bench` 46 covariate tasks).

Usage:
  python scripts/run_kill.py run --model amazon/chronos-2 --max-tasks 46 --out results/kill.jsonl
  python scripts/run_kill.py report --out results/kill.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsfm_identifiability.residuals import (  # noqa: E402
    paired_bootstrap_rel_reduction_ci,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "kill.jsonl"
ARMS_TSFM = (
    "seasonal_naive",
    "chronos2_univariate",
    "chronos2_covariates",
    "uni_chronos2_ridge",
    "uni_chronos2_qlinear",
    "uni_chronos2_lgbm",
    "uni_chronos2_scale_rescue",
)
CAUSAL_HEADS = ("uni_chronos2_ridge", "uni_chronos2_qlinear", "uni_chronos2_lgbm")


def run(args: argparse.Namespace) -> None:
    import torch
    from chronos import Chronos2Pipeline
    from tsfm_identifiability.fev_runner import (
        FEV_BENCH_YAML,
        evaluate_fev_task,
        load_fev_covariate_tasks,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_task_idxs: set[int] = set()
    if out_path.exists():
        # Count distinct arms written per task_idx so interrupted partial tasks are re-run cleanly
        arms_per_task: dict[int, set[str]] = defaultdict(set)
        kept_lines: list[str] = []
        raw_lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
        for ln in raw_lines:
            r = json.loads(ln)
            arms_per_task[int(r["task_idx"])].add(r["arm"])
        complete_t = {t_i for t_i, a_set in arms_per_task.items() if len(a_set) == len(ARMS_TSFM)}
        if len(complete_t) < len(arms_per_task):
            for ln in raw_lines:
                r = json.loads(ln)
                if int(r["task_idx"]) in complete_t:
                    kept_lines.append(ln)
            out_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
        done_task_idxs = complete_t

    print(f"[tsfm-scale-identifiability] Loading {args.model} on {device} ({dtype})...", flush=True)
    try:
        pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map=device, torch_dtype=dtype)
    except (TypeError, ValueError):
        pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map=device, dtype=dtype)

    tasks = load_fev_covariate_tasks(yaml_url=args.yaml_url, max_tasks=args.max_tasks)
    print(f"  Discovered {len(tasks)} dynamic-covariate tasks in fev-bench.", flush=True)

    manifest = {
        "project": "tsfm-scale-identifiability",
        "model": args.model,
        "num_covariate_tasks": len(tasks),
        "max_series_per_window": args.max_series_per_window,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dtype": str(dtype),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

    t_start = time.perf_counter()
    newly_done = 0

    with out_path.open("a") as f:
        for t_idx, task in enumerate(tasks):
            if t_idx in done_task_idxs:
                print(f"  [Skip completed task {t_idx + 1}/{len(tasks)}]", flush=True)
                continue
            t_task0 = time.perf_counter()
            t_name = getattr(task, "dataset_config", None) or getattr(task, "dataset_path", f"task_{t_idx}")
            print(f"  [{t_idx + 1}/{len(tasks)}] Evaluating task={t_name} (H={task.horizon})...", flush=True)
            rows = evaluate_fev_task(
                pipeline=pipeline,
                task=task,
                task_idx=t_idx,
                max_series_per_window=args.max_series_per_window,
                max_context_len=args.max_context_len,
            )
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            newly_done += 1
            dt_t = time.perf_counter() - t_task0
            print(f"    -> wrote {len(rows)} rows in {dt_t:.1f}s", flush=True)

            if newly_done == 5:
                elapsed = time.perf_counter() - t_start
                est_total_h = (elapsed / 5.0) * len(tasks) / 3600.0
                print(
                    f"  [Throughput Check @ 5 tasks] {elapsed:.1f}s -> Projected full {len(tasks)}-task run: {est_total_h:.2f} GPU-h",
                    flush=True,
                )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate SQL & MASE overall and on the Quiescent-to-Active Shock Slice (rho_w >= 2.5)."""
    # Pair by unique `(task_idx, window_idx, series_id)` key
    by_item: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get("sql") is None or not np.isfinite(r["sql"]):
            continue
        item_key = (int(r["task_idx"]), int(r["window_idx"]), str(r["series_id"]))
        by_item[item_key][r["arm"]] = r

    valid_items = [k for k, amap in by_item.items() if "chronos2_covariates" in amap and "chronos2_univariate" in amap]
    shock_items = [k for k in valid_items if by_item[k]["chronos2_covariates"].get("is_shock", False)]

    def _slice_stats(keys: list[tuple[int, int, str]]) -> dict[str, Any]:
        if not keys:
            return {"n_series_windows": 0}
        out: dict[str, Any] = {"n_series_windows": len(keys)}
        mv_sql = [by_item[k]["chronos2_covariates"]["sql"] for k in keys]
        uni_sql = [by_item[k]["chronos2_univariate"]["sql"] for k in keys]

        for arm in ARMS_TSFM:
            shared = [k for k in keys if arm in by_item[k]]
            if not shared:
                continue
            a_sql = [by_item[k][arm]["sql"] for k in shared]
            a_mase = [by_item[k][arm]["mase"] for k in shared if by_item[k][arm].get("mase") is not None]
            a_ms = [by_item[k][arm].get("wall_ms_per_series", 0.0) for k in shared]
            out[f"{arm}_mean_sql"] = round(float(np.mean(a_sql)), 4)
            out[f"{arm}_median_sql"] = round(float(np.median(a_sql)), 4)
            out[f"{arm}_mean_mase"] = round(float(np.mean(a_mase)), 4) if a_mase else None
            out[f"{arm}_wall_ms"] = round(float(np.mean(a_ms)), 3)

            if arm != "chronos2_covariates":
                ref_sh = [by_item[k]["chronos2_covariates"]["sql"] for k in shared]
                g_mean, g_lo, g_hi = paired_bootstrap_rel_reduction_ci(ref_sh, a_sql, n_boot=10_000, seed=0)
                out[f"{arm}_vs_mv_sql_reduction_pct_ci"] = (round(g_mean, 3), round(g_lo, 3), round(g_hi, 3))

            if arm in CAUSAL_HEADS:
                uni_sh = [by_item[k]["chronos2_univariate"]["sql"] for k in shared]
                u_mean, u_lo, u_hi = paired_bootstrap_rel_reduction_ci(uni_sh, a_sql, n_boot=10_000, seed=1)
                out[f"{arm}_vs_uni_sql_reduction_pct_ci"] = (round(u_mean, 3), round(u_lo, 3), round(u_hi, 3))
        return out

    return {
        "n_distinct_tasks": len({k[0] for k in valid_items}),
        "overall": _slice_stats(valid_items),
        "shock_slice_rho_ge_2_5": _slice_stats(shock_items),
    }


def verdict(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pre-registered CONTINUE / KILL gate for tsfm-scale-identifiability."""
    overall = summary.get("overall", {})
    shock = summary.get("shock_slice_rho_ge_2_5", {})
    if overall.get("n_series_windows", 0) == 0:
        return "INVALID", {"reason": "No valid series-windows evaluated."}

    best_head = None
    best_shock_gain = -999.0
    for h in CAUSAL_HEADS:
        ci_key = f"{h}_vs_mv_sql_reduction_pct_ci"
        if ci_key in shock and shock[ci_key][0] > best_shock_gain:
            best_shock_gain = shock[ci_key][0]
            best_head = h

    if best_head is None:
        return "KILL", {"reason": "No causal residual head evaluated on shock slice (or shock slice empty)."}

    ov_gain, ov_lo, ov_hi = overall[f"{best_head}_vs_mv_sql_reduction_pct_ci"]
    sh_gain, sh_lo, sh_hi = shock[f"{best_head}_vs_mv_sql_reduction_pct_ci"]
    uni_sh_gain, _, _ = shock.get(f"{best_head}_vs_uni_sql_reduction_pct_ci", (0.0, 0.0, 0.0))

    details = {
        "best_causal_head": best_head,
        "overall_sql_reduction_vs_mv_pct": ov_gain,
        "overall_ci95_pct": (ov_lo, ov_hi),
        "shock_sql_reduction_vs_mv_pct": sh_gain,
        "shock_ci95_pct": (sh_lo, sh_hi),
        "shock_sql_reduction_vs_univariate_pct": uni_sh_gain,
        "n_shock_windows": shock.get("n_series_windows", 0),
    }

    # Pre-registered gate:
    # 1) Overall parity or better vs native Chronos-2 Group Attention (ov_gain >= -0.5%, kill if < -1.5%)
    # 2) Shock slice (rho_w >= 2.5) SQL reduction >= 8.0% with 10,000-resample 95% CI lower bound > 0.0%
    # 3) Shock slice beats pure univariate Chronos-2 (proving covariates genuinely carry signal)
    if ov_gain < -1.5:
        return "KILL", details
    if ov_gain >= -0.5 and sh_gain >= 8.0 and sh_lo > 0.0 and uni_sh_gain >= 2.0:
        return "CONTINUE", details
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
    pr.add_argument("--model", default="amazon/chronos-2")
    pr.add_argument(
        "--yaml-url",
        default="https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/fev_bench/tasks.yaml",
    )
    pr.add_argument("--max-tasks", type=int, default=46)
    pr.add_argument("--max-series-per-window", type=int, default=128)
    pr.add_argument("--max-context-len", type=int, default=1024)
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
