"""Official fev-bench covariate-task evaluator for Chronos-2 Group Attention vs. Causal Residual Decoupling."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .residuals import (
    QUANTILE_LEVELS,
    CausalResidualConfig,
    build_causal_covariate_features,
    compute_covariate_shock_ratio,
    enforce_monotone_quantiles,
    fit_predict_lgbm_quantile,
    fit_predict_quantile_linear,
    fit_predict_ridge_gcv,
    generate_rolling_origin_folds,
    mase_score,
    scaled_quantile_loss,
    seasonal_naive_forecast,
)

FEV_BENCH_YAML = "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/fev_bench/tasks.yaml"

ARMS_TSFM = (
    "seasonal_naive",
    "chronos2_univariate",
    "chronos2_covariates",
    "uni_chronos2_ridge",
    "uni_chronos2_qlinear",
    "uni_chronos2_lgbm",
    "uni_chronos2_scale_rescue",
)


def _cuda_sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _extract_numeric_covariates(
    past_row: dict[str, Any],
    fut_row: dict[str, Any],
    cov_cols: list[str],
    t_past: int,
    h_fut: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert numeric or categorical known dynamic covariates into [T_past, D] and [H, D] float64 arrays."""
    past_cols = []
    fut_cols = []
    for c in cov_cols:
        vp = list(past_row[c])
        vf = list(fut_row[c])
        s_all = pd.Series(vp + vf)
        num = pd.to_numeric(s_all, errors="coerce")
        if num.notna().sum() >= 0.5 * len(s_all):
            arr = num.to_numpy(dtype=np.float64)
        else:
            # Categorical known covariate (e.g., holiday/event code): map categories seen in past, unseen -> -1
            past_cats = pd.Series(vp).astype(str)
            uniq = {val: float(idx) for idx, val in enumerate(dict.fromkeys(past_cats))}
            arr = np.array([uniq.get(str(v), -1.0) for v in (vp + vf)], dtype=np.float64)
        past_cols.append(arr[:t_past])
        fut_cols.append(arr[t_past : t_past + h_fut])
    if not past_cols:
        return np.zeros((t_past, 0), dtype=np.float64), np.zeros((h_fut, 0), dtype=np.float64)
    return np.column_stack(past_cols), np.column_stack(fut_cols)


def _unpack_chronos_quantiles(pred_df: pd.DataFrame, id_col: str, quantiles: tuple[float, ...]) -> dict[str, np.ndarray]:
    """Map `item_id` -> `[H, len(quantiles)]` numpy array sorted by timestamp."""
    q_cols = []
    for q in quantiles:
        # Chronos2Pipeline names quantile columns as str(q) e.g. '0.1', '0.5', '0.9'
        if str(q) in pred_df.columns:
            q_cols.append(str(q))
        elif f"{q:.1f}" in pred_df.columns:
            q_cols.append(f"{q:.1f}")
        elif q in pred_df.columns:
            q_cols.append(q)
        else:
            raise KeyError(f"Quantile column for {q} not found in {list(pred_df.columns)}")

    out: dict[str, np.ndarray] = {}
    for item_id, grp in pred_df.groupby(id_col, sort=False):
        out[str(item_id)] = enforce_monotone_quantiles(grp[q_cols].to_numpy(dtype=np.float64))
    return out


def load_fev_covariate_tasks(yaml_url: str = FEV_BENCH_YAML, max_tasks: int | None = None) -> list[Any]:
    import fev

    bench = fev.Benchmark.from_yaml(yaml_url)
    cov_tasks = [t for t in bench.tasks if getattr(t, "known_dynamic_columns", None)]
    if max_tasks is not None:
        cov_tasks = cov_tasks[:max_tasks]
    return cov_tasks


def evaluate_fev_task(
    pipeline: Any,
    task: Any,
    task_idx: int,
    max_series_per_window: int = 128,
    max_context_len: int = 1024,
    cfg: CausalResidualConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all 7 arms on a real `fev-bench` covariate task with strict pre-cutoff causality."""
    import fev

    cfg = cfg or CausalResidualConfig()
    id_col = getattr(task, "id_column", "id")
    ts_col = getattr(task, "timestamp_column", "timestamp")
    target_col = getattr(task, "target", "target")
    if isinstance(target_col, list):
        target_col = target_col[0]
    cov_cols = list(getattr(task, "known_dynamic_columns", []))
    horizon = int(task.horizon)
    seasonality = int(getattr(task, "seasonality", 1) or 1)
    task_name = getattr(task, "dataset_config", None) or getattr(task, "dataset_path", f"task_{task_idx}")

    results_rows: list[dict[str, Any]] = []

    for win_idx, window in enumerate(task.iter_windows()):
        past_ds, fut_ds = window.get_input_data()
        n_total_series = len(past_ds)
        if n_total_series > max_series_per_window:
            rng = np.random.default_rng(win_idx)
            sel_indices = sorted(rng.choice(n_total_series, size=max_series_per_window, replace=False).tolist())
            if hasattr(past_ds, "select"):
                past_ds = past_ds.select(sel_indices)
                fut_ds = fut_ds.select(sel_indices)
            elif hasattr(past_ds, "iloc"):
                past_ds = past_ds.iloc[sel_indices].reset_index(drop=True)
                fut_ds = fut_ds.iloc[sel_indices].reset_index(drop=True)
            elif isinstance(past_ds, list):
                past_ds = [past_ds[i] for i in sel_indices]
                fut_ds = [fut_ds[i] for i in sel_indices]
        else:
            sel_indices = list(range(n_total_series))

        # Build clean long-format DataFrames for Multivariate Chronos-2 and Univariate Causal Heads
        cov_past_frames = []
        cov_fut_frames = []
        uni_frames = []
        series_meta: list[dict[str, Any]] = []

        for s_i in range(len(past_ds)):
            p_row = past_ds[s_i]
            f_row = fut_ds[s_i]
            raw_id = str(p_row[id_col])
            y_raw = p_row[target_col]
            y_full = np.asarray(
                pd.to_numeric(pd.Series(y_raw), errors="coerce").interpolate().bfill().ffill().fillna(0.0),
                dtype=np.float64,
            )
            ts_full = list(p_row[ts_col])
            if len(y_full) > max_context_len:
                y_past = y_full[-max_context_len:]
                ts_past = ts_full[-max_context_len:]
                p_row_clamped = {c: list(p_row[c])[-max_context_len:] for c in cov_cols}
            else:
                y_past = y_full
                ts_past = ts_full
                p_row_clamped = p_row

            # Extract numeric covariates [T_past, D] and [H, D]
            x_past, x_fut = _extract_numeric_covariates(p_row_clamped, f_row, cov_cols, len(y_past), horizon)

            # Extract future timestamps (strictly len == horizon)
            ts_fut_raw = list(f_row[ts_col])
            if len(ts_fut_raw) >= horizon:
                ts_fut = ts_fut_raw[:horizon]
            elif len(ts_fut_raw) > 1:
                try:
                    step = ts_fut_raw[1] - ts_fut_raw[0]
                    last_t = ts_fut_raw[-1]
                    ts_fut = ts_fut_raw + [last_t + step * (k + 1) for k in range(horizon - len(ts_fut_raw))]
                except Exception:
                    ts_fut = list(range(horizon))
            else:
                ts_fut = list(range(horizon))

            # 1. Past multivariate frame
            p_dict = {id_col: raw_id, ts_col: ts_past, target_col: y_past}
            for c_i, c_name in enumerate(cov_cols):
                p_dict[c_name] = x_past[:, c_i]
            cov_past_frames.append(pd.DataFrame(p_dict))

            # 2. Future covariate frame (strictly horizon rows)
            f_dict = {id_col: raw_id, ts_col: ts_fut}
            for c_i, c_name in enumerate(cov_cols):
                f_dict[c_name] = x_fut[:, c_i]
            cov_fut_frames.append(pd.DataFrame(f_dict))

            # 3. Univariate folds for Causal Residual Heads
            rho_w = compute_covariate_shock_ratio(x_past, x_fut)
            folds = generate_rolling_origin_folds(
                t_cutoff=len(y_past),
                horizon=horizon,
                max_folds=cfg.max_rolling_folds,
                min_subcontext=cfg.min_subcontext_len,
            )

            test_key = f"{raw_id}__test"
            uni_frames.append(pd.DataFrame({id_col: test_key, ts_col: ts_past, target_col: y_past}))
            for f_k, (orig_t, _end_t) in enumerate(folds):
                fold_key = f"{raw_id}__fold{f_k}"
                uni_frames.append(
                    pd.DataFrame({id_col: fold_key, ts_col: ts_past[:orig_t], target_col: y_past[:orig_t]})
                )

            series_meta.append(
                {
                    "s_idx": sel_indices[s_i],
                    "raw_id": raw_id,
                    "y_past": y_past,
                    "x_past": x_past,
                    "x_fut": x_fut,
                    "rho_w": rho_w,
                    "folds": folds,
                }
            )

        cov_past_df = pd.concat(cov_past_frames, ignore_index=True)
        cov_fut_df = pd.concat(cov_fut_frames, ignore_index=True)

        t0 = _cuda_sync_time()
        mv_pred_df = pipeline.predict_df(
            cov_past_df[[id_col, ts_col, target_col] + cov_cols],
            future_df=(cov_fut_df[[id_col, ts_col] + cov_cols] if cov_cols else None),
            prediction_length=horizon,
            quantile_levels=list(QUANTILE_LEVELS),
            id_column=id_col,
            timestamp_column=ts_col,
            target=target_col,
        )
        mv_wall_s = _cuda_sync_time() - t0
        mv_map = _unpack_chronos_quantiles(mv_pred_df, id_col=id_col, quantiles=QUANTILE_LEVELS)

        batched_uni_df = pd.concat(uni_frames, ignore_index=True)
        batched_uni_df[target_col] = (
            pd.to_numeric(batched_uni_df[target_col], errors="coerce")
            .interpolate(method="linear")
            .bfill()
            .ffill()
            .fillna(0.0)
            .astype(np.float64)
        )
        t1 = _cuda_sync_time()
        uni_pred_df = pipeline.predict_df(
            batched_uni_df,
            future_df=None,
            prediction_length=horizon,
            quantile_levels=list(QUANTILE_LEVELS),
            id_column=id_col,
            timestamp_column=ts_col,
            target=target_col,
        )
        uni_map = _unpack_chronos_quantiles(uni_pred_df, id_col=id_col, quantiles=QUANTILE_LEVELS)

        # 3. Fit strictly pre-cutoff residual heads in physical units (plus pooled ScaleRatioRescue diagnostic)
        med_idx = QUANTILE_LEVELS.index(0.5)
        per_series_preds: dict[str, dict[str, np.ndarray]] = {}
        pooled_X_tr, pooled_r_tr = [], []
        pooled_meta_te: dict[str, np.ndarray] = {}

        for m in series_meta:
            rid = m["raw_id"]
            yp = m["y_past"]
            xp, xf = m["x_past"], m["x_fut"]
            uni_q = uni_map[f"{rid}__test"]
            snaive_q = seasonal_naive_forecast(yp, horizon=horizon, seasonality=seasonality)

            arm_out = {
                "seasonal_naive": snaive_q,
                "chronos2_univariate": uni_q,
                "chronos2_covariates": mv_map[rid],
            }

            folds = m["folds"]
            if not folds or xp.shape[1] == 0:
                arm_out["uni_chronos2_ridge"] = uni_q
                arm_out["uni_chronos2_qlinear"] = uni_q
                arm_out["uni_chronos2_lgbm"] = uni_q
            else:
                X_past_f, X_fut_f = build_causal_covariate_features(xp, xf)
                X_rows, r_med_rows = [], []
                for f_k, (orig_t, end_t) in enumerate(folds):
                    f_uni = uni_map[f"{rid}__fold{f_k}"]
                    actual_seg = yp[orig_t:end_t]
                    X_rows.append(X_past_f[orig_t:end_t])
                    r_med_rows.append(actual_seg - f_uni[:, med_idx])

                X_tr = np.vstack(X_rows)
                r_med = np.concatenate(r_med_rows)

                if len(r_med) < cfg.min_residual_samples:
                    arm_out["uni_chronos2_ridge"] = uni_q
                    arm_out["uni_chronos2_qlinear"] = uni_q
                    arm_out["uni_chronos2_lgbm"] = uni_q
                else:
                    d_ridge = fit_predict_ridge_gcv(X_tr, r_med, X_fut_f)
                    d_qlin = fit_predict_quantile_linear(X_tr, r_med, X_fut_f, l2_reg=0.25)
                    d_lgbm = fit_predict_lgbm_quantile(X_tr, r_med, X_fut_f)
                    arm_out["uni_chronos2_ridge"] = enforce_monotone_quantiles(uni_q + d_ridge)
                    arm_out["uni_chronos2_qlinear"] = enforce_monotone_quantiles(
                        uni_q[:, med_idx : med_idx + 1] + d_qlin
                    )
                    arm_out["uni_chronos2_lgbm"] = enforce_monotone_quantiles(
                        uni_q[:, med_idx : med_idx + 1] + d_lgbm
                    )

                    # Collect normalized RevIN covariates + explicit log(1 + sigma_x / sigma_y) rescue feature!
                    sig_y = max(float(np.std(yp[np.isfinite(yp)])), 1e-6)
                    mu_x = np.mean(X_tr, axis=0, keepdims=True)
                    sig_x = np.maximum(np.std(X_tr, axis=0, keepdims=True), 1e-6)
                    X_tr_norm = np.clip((X_tr - mu_x) / sig_x, -15.0, 15.0)
                    X_te_norm = np.clip((X_fut_f - mu_x) / sig_x, -15.0, 15.0)
                    log_scale_ratio = np.log1p(sig_x / sig_y)
                    pooled_X_tr.append(np.hstack([X_tr_norm, X_tr_norm * log_scale_ratio]))
                    pooled_r_tr.append(r_med / sig_y)
                    pooled_meta_te[rid] = (np.hstack([X_te_norm, X_te_norm * log_scale_ratio]), sig_y)

            per_series_preds[rid] = arm_out

        # Fit cross-series Scale-Ratio Rescue diagnostic (strictly t <= T_cutoff across series in this window)
        if pooled_X_tr and len(pooled_X_tr) >= 2:
            P_X = np.vstack(pooled_X_tr)
            P_r = np.concatenate(pooled_r_tr)
            for m in series_meta:
                rid = m["raw_id"]
                uni_q = per_series_preds[rid]["chronos2_univariate"]
                if rid in pooled_meta_te:
                    X_te_aug, sig_y = pooled_meta_te[rid]
                    d_resc = fit_predict_ridge_gcv(P_X, P_r, X_te_aug) * sig_y
                    per_series_preds[rid]["uni_chronos2_scale_rescue"] = enforce_monotone_quantiles(uni_q + d_resc)
                else:
                    per_series_preds[rid]["uni_chronos2_scale_rescue"] = uni_q
        else:
            for m in series_meta:
                rid = m["raw_id"]
                per_series_preds[rid]["uni_chronos2_scale_rescue"] = per_series_preds[rid]["uni_chronos2_ridge"]

        decoupled_wall_s = _cuda_sync_time() - t1

        # 4. ONLY NOW read ground-truth horizon targets `y_true` to score all 7 arms!
        if hasattr(window, "get_ground_truth"):
            gt_ds = window.get_ground_truth()
        elif hasattr(window, "get_output_data"):
            gt_ds = window.get_output_data()
        elif hasattr(window, "_get_ground_truth"):
            gt_ds = window._get_ground_truth()
        else:
            raise AttributeError(f"Cannot extract ground truth from EvaluationWindow: {dir(window)}")
        if n_total_series > max_series_per_window:
            if hasattr(gt_ds, "select"):
                gt_ds = gt_ds.select(sel_indices)
            elif hasattr(gt_ds, "iloc"):
                gt_ds = gt_ds.iloc[sel_indices].reset_index(drop=True)
            elif isinstance(gt_ds, list):
                gt_ds = [gt_ds[i] for i in sel_indices]

        n_s = max(1, len(series_meta))
        for idx_local, m in enumerate(series_meta):
            rid = m["raw_id"]
            yp = m["y_past"]
            if hasattr(gt_ds, "iloc"):
                gt_row = gt_ds.iloc[idx_local]
            else:
                gt_row = gt_ds[idx_local]
            y_raw = gt_row[target_col]
            y_true = np.asarray(pd.to_numeric(pd.Series(y_raw), errors="coerce").fillna(0.0), dtype=np.float64)
            min_h = min(len(y_true), horizon)
            y_true = y_true[:min_h]
            rho_w = m["rho_w"]
            for arm in ARMS_TSFM:
                qp = per_series_preds[rid][arm][:min_h]
                sql = scaled_quantile_loss(yp, y_true, qp, seasonality=seasonality)
                mase = mase_score(yp, y_true, qp[:, med_idx], seasonality=seasonality)
                wall_ms = (
                    (mv_wall_s * 1000.0 / n_s)
                    if arm == "chronos2_covariates"
                    else ((decoupled_wall_s * 1000.0 / n_s) if arm.startswith("uni_chronos2") else 1.0)
                )
                results_rows.append(
                    {
                        "task": str(task_name),
                        "task_idx": int(task_idx),
                        "window_idx": int(win_idx),
                        "series_id": rid,
                        "arm": arm,
                        "sql": float(sql) if np.isfinite(sql) else None,
                        "mase": float(mase) if np.isfinite(mase) else None,
                        "rho_w": round(float(rho_w), 4),
                        "is_shock": bool(rho_w >= 2.5),
                        "wall_ms_per_series": round(float(wall_ms), 3),
                    }
                )

    return results_rows
