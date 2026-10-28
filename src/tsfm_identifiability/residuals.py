"""Zero-leakage causal rolling-origin residual heads, RevIN non-identifiability diagnostics, and fev metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class CausalResidualConfig:
    max_rolling_folds: int = 4
    min_subcontext_len: int = 16
    min_residual_samples: int = 12
    feature_clip_z: float = 15.0
    l2_quantile_reg: float = 2.0


def enforce_monotone_quantiles(q_preds: np.ndarray) -> np.ndarray:
    """Enforce non-crossing quantiles q_0.1 <= q_0.2 <= ... <= q_0.9 along last axis."""
    return np.maximum.accumulate(np.asarray(q_preds, dtype=np.float64), axis=-1)


def _causal_impute_2d(x: np.ndarray, lookback_len: int) -> np.ndarray:
    """Impute NaN/Inf in covariates using strictly causal forward-fill and lookback-only medians."""
    arr = np.array(x, dtype=np.float64, copy=True)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr[~np.isfinite(arr)] = np.nan
    t_total, d = arr.shape
    for j in range(d):
        col_past = arr[:lookback_len, j]
        finite_past = col_past[np.isfinite(col_past)]
        fill_val = float(np.median(finite_past)) if finite_past.size > 0 else 0.0
        last = fill_val
        for t in range(t_total):
            if np.isfinite(arr[t, j]):
                last = arr[t, j]
            else:
                arr[t, j] = last
    return arr


def compute_covariate_shock_ratio(x_past: np.ndarray, x_future: np.ndarray, eps: float = 1e-6) -> float:
    """Compute window covariate shock ratio rho_w = max_j std(x_{j,future}) / max(std(x_{j,past}), eps)."""
    xp = np.asarray(x_past, dtype=np.float64)
    xf = np.asarray(x_future, dtype=np.float64)
    if xp.size == 0 or xf.size == 0:
        return 1.0
    if xp.ndim == 1:
        xp = xp[:, None]
    if xf.ndim == 1:
        xf = xf[:, None]
    t_past = xp.shape[0]
    joined = _causal_impute_2d(np.vstack([xp, xf]), lookback_len=t_past)
    std_past = np.std(joined[:t_past], axis=0)
    std_fut = np.std(joined[t_past:], axis=0)
    # Ignore strictly constant dummy columns where both past and future std are ~0
    active_mask = (std_past > eps) | (std_fut > eps)
    if not np.any(active_mask):
        return 1.0
    ratios = std_fut[active_mask] / np.maximum(std_past[active_mask], eps)
    return float(np.max(ratios))


def per_variate_revin_normalize(
    y_past: np.ndarray,
    x_past: np.ndarray,
    x_future: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Reference per-variate RevIN operator demonstrating erasure of physical cross-variate scale sigma_x / sigma_y."""
    yp = np.asarray(y_past, dtype=np.float64)
    mu_y = float(np.mean(yp))
    sig_y = float(max(np.std(yp), eps))
    y_norm = (yp - mu_y) / sig_y

    xp = np.asarray(x_past, dtype=np.float64)
    xf = np.asarray(x_future, dtype=np.float64)
    if xp.ndim == 1:
        xp = xp[:, None]
    if xf.ndim == 1:
        xf = xf[:, None]
    mu_x = np.mean(xp, axis=0, keepdims=True)
    sig_x = np.maximum(np.std(xp, axis=0, keepdims=True), eps)
    xp_norm = (xp - mu_x) / sig_x
    xf_norm = (xf - mu_x) / sig_x
    return y_norm, xp_norm, xf_norm, (mu_y, sig_y)


def build_causal_covariate_features(x_past: np.ndarray, x_future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build strictly causal lag/difference covariate features `[x_t, x_{t-1}, x_t - x_{t-1}]` for past & future."""
    xp = np.asarray(x_past, dtype=np.float64)
    xf = np.asarray(x_future, dtype=np.float64)
    if xp.ndim == 1:
        xp = xp[:, None]
    if xf.ndim == 1:
        xf = xf[:, None]
    t_past = xp.shape[0]
    x_all = _causal_impute_2d(np.vstack([xp, xf]), lookback_len=t_past)
    x_lag1 = np.vstack([x_all[0:1], x_all[:-1]])
    x_diff = x_all - x_lag1
    feats_all = np.hstack([x_all, x_lag1, x_diff])
    return feats_all[:t_past], feats_all[t_past:]


def generate_rolling_origin_folds(
    t_cutoff: int,
    horizon: int,
    max_folds: int = 4,
    min_subcontext: int = 16,
) -> list[tuple[int, int]]:
    """Return strictly pre-cutoff `(origin_idx, end_idx)` pairs with `end_idx <= t_cutoff`.

    Guarantees ZERO access to any observation at `t >= t_cutoff`.
    Returns `[]` when `t_cutoff < max(min_subcontext, horizon) + horizon` (`< 2H` rule).
    """
    eff_min_ctx = max(min_subcontext, horizon)
    if t_cutoff < eff_min_ctx + horizon:
        return []
    folds: list[tuple[int, int]] = []
    for k in range(1, max_folds + 1):
        end_idx = t_cutoff - (k - 1) * horizon
        origin_idx = end_idx - horizon
        if origin_idx < eff_min_ctx:
            break
        folds.append((origin_idx, end_idx))
    folds.reverse()
    return folds


def _standardize_X_only(
    X_train: np.ndarray,
    X_test: np.ndarray,
    clip_z: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize X using strictly training-fold statistics while leaving target residuals in physical units."""
    mu = np.mean(X_train, axis=0, keepdims=True)
    sig = np.std(X_train, axis=0, keepdims=True)
    active = (sig.reshape(-1) > 1e-8)
    if not np.any(active):
        return np.zeros((X_train.shape[0], 0)), np.zeros((X_test.shape[0], 0)), active
    sig_safe = np.where(sig > 1e-8, sig, 1.0)
    X_tr = np.clip((X_train[:, active] - mu[:, active]) / sig_safe[:, active], -clip_z, clip_z)
    X_te = np.clip((X_test[:, active] - mu[:, active]) / sig_safe[:, active], -clip_z, clip_z)
    return X_tr, X_te, active


def fit_predict_ridge_gcv(
    X_train: np.ndarray,
    r_train: np.ndarray,
    X_test: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILE_LEVELS,
    alphas: np.ndarray | None = None,
    clip_z: float = 15.0,
) -> np.ndarray:
    """Closed-form SVD Leave-One-Out Generalized Cross-Validation Ridge regression in physical target units.

    Shifts all univariate TSFM quantiles by the predicted covariate mean effect `delta_hat(x)`, preserving
    the foundation model's calibrated uncertainty spread while restoring physical covariate sensitivity!
    """
    h_out = X_test.shape[0]
    n_q = len(quantiles)
    r = np.asarray(r_train, dtype=np.float64).reshape(-1)
    finite = np.isfinite(r)
    if np.sum(finite) < 6:
        return np.zeros((h_out, n_q), dtype=np.float64)

    X_tr, X_te, _ = _standardize_X_only(X_train[finite], X_test, clip_z=clip_z)
    r = r[finite]
    if X_tr.shape[1] == 0:
        return np.zeros((h_out, n_q), dtype=np.float64)

    r_mean = float(np.mean(r))
    r_c = r - r_mean
    n_samples = float(X_tr.shape[0])

    if alphas is None:
        alphas = np.logspace(-1.0, 5.0, 25)

    u, s, vt = np.linalg.svd(X_tr, full_matrices=False)
    uty = u.T @ r_c
    s2 = s**2

    best_gcv = float("inf")
    best_alpha = float(alphas[-1])
    # Also allow alpha -> infinity (zero slope shrinkage) if covariates are pure noise!
    null_gcv = float(np.mean(r_c**2))

    for alpha in alphas:
        filt = s2 / (s2 + alpha)
        df = float(np.sum(filt))
        denom = max(1e-4, (1.0 - df / n_samples) ** 2)
        # Residual norm ||r_c - U diag(filt) U^T r_c||^2
        proj_diff = r_c - u @ (filt * uty)
        gcv = float(np.mean(proj_diff**2)) / denom
        if gcv < best_gcv:
            best_gcv = gcv
            best_alpha = float(alpha)

    if null_gcv <= best_gcv:
        return np.zeros((h_out, n_q), dtype=np.float64)

    w = vt.T @ ((s / (s2 + best_alpha)) * uty)
    delta_pred = X_te @ w + r_mean
    return np.repeat(delta_pred[:, None], n_q, axis=1)


def fit_predict_quantile_linear(
    X_train: np.ndarray,
    r_train_per_q: np.ndarray,
    X_test: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILE_LEVELS,
    l2_reg: float = 2.0,
    clip_z: float = 15.0,
) -> np.ndarray:
    """Convex L2-regularized smooth pinball-loss linear quantile regression per quantile level."""
    h_out = X_test.shape[0]
    n_q = len(quantiles)
    r_mat = np.asarray(r_train_per_q, dtype=np.float64)
    if r_mat.ndim == 1:
        r_mat = np.repeat(r_mat[:, None], n_q, axis=1)

    valid = np.all(np.isfinite(r_mat), axis=1)
    if np.sum(valid) < 8:
        return np.zeros((h_out, n_q), dtype=np.float64)

    X_tr, X_te, _ = _standardize_X_only(X_train[valid], X_test, clip_z=clip_z)
    r_mat = r_mat[valid]
    if X_tr.shape[1] == 0:
        return np.zeros((h_out, n_q), dtype=np.float64)

    n_samples, d = X_tr.shape
    preds = np.zeros((h_out, n_q), dtype=np.float64)

    for q_i, q in enumerate(quantiles):
        target = r_mat[:, q_i]
        scale_r = max(float(np.std(target)), 1e-3)
        kappa = 1e-2 * scale_r

        def _obj_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:-1]
            b = params[-1]
            err = target - (X_tr @ w + b)
            # Smooth Huberized pinball loss for exact L-BFGS-B convergence
            abs_e = np.abs(err)
            quad = np.minimum(abs_e, kappa)
            lin = abs_e - quad
            huber = 0.5 * (quad**2) / kappa + lin
            weight = np.where(err >= 0.0, q, 1.0 - q)
            loss = float(np.mean(weight * huber)) + 0.5 * (l2_reg / n_samples) * float(np.sum(w**2))

            # d/d(err) of weight * huber
            dh_derr = weight * np.where(abs_e <= kappa, err / kappa, np.sign(err))
            grad_w = -(X_tr.T @ dh_derr) / n_samples + (l2_reg / n_samples) * w
            grad_b = -float(np.mean(dh_derr))
            return loss, np.append(grad_w, grad_b)

        init = np.zeros(d + 1, dtype=np.float64)
        res = minimize(_obj_and_grad, init, jac=True, method="L-BFGS-B", options={"maxiter": 80})
        w_hat, b_hat = res.x[:-1], res.x[-1]
        preds[:, q_i] = X_te @ w_hat + b_hat

    return enforce_monotone_quantiles(preds)


def fit_predict_lgbm_quantile(
    X_train: np.ndarray,
    r_train_per_q: np.ndarray,
    X_test: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILE_LEVELS,
    clip_z: float = 15.0,
) -> np.ndarray:
    """Ridge-detrended shallow LightGBM quantile residual booster (native C++ lgb.train API).

    Detrending with GCV Ridge first ensures out-of-support covariate shocks (rho_w >= 2.5) extrapolate
    cleanly instead of saturating at the outermost historical tree leaf, while shallow quantile trees
    capture nonlinear thresholds and heteroscedastic uncertainty on the detrended residuals.
    """
    import lightgbm as lgb

    h_out = X_test.shape[0]
    n_q = len(quantiles)
    r_mat = np.asarray(r_train_per_q, dtype=np.float64)
    if r_mat.ndim == 1:
        r_mat = np.repeat(r_mat[:, None], n_q, axis=1)

    valid = np.all(np.isfinite(r_mat), axis=1)
    if np.sum(valid) < 16:
        return fit_predict_quantile_linear(X_train, r_mat, X_test, quantiles=quantiles, clip_z=clip_z)

    X_tr, X_te, _ = _standardize_X_only(X_train[valid], X_test, clip_z=clip_z)
    r_mat = r_mat[valid]
    if X_tr.shape[1] == 0:
        return np.zeros((h_out, n_q), dtype=np.float64)

    med_col = len(quantiles) // 2
    base_tr = fit_predict_ridge_gcv(X_tr, r_mat[:, med_col], X_tr, quantiles=(0.5,), clip_z=clip_z)[:, 0:1]
    base_te = fit_predict_ridge_gcv(X_tr, r_mat[:, med_col], X_te, quantiles=(0.5,), clip_z=clip_z)[:, 0:1]
    detrended = r_mat - base_tr

    min_leaf = max(4, min(12, X_tr.shape[0] // 5))
    preds = np.zeros((h_out, n_q), dtype=np.float64)
    for q_i, q in enumerate(quantiles):
        dtrain = lgb.Dataset(X_tr, label=detrended[:, q_i], free_raw_data=False)
        params = {
            "objective": "quantile",
            "alpha": float(q),
            "max_depth": 3,
            "num_leaves": 7,
            "learning_rate": 0.08,
            "min_data_in_leaf": min_leaf,
            "lambda_l2": 1.0,
            "seed": 0,
            "num_threads": 1,
            "verbose": -1,
        }
        booster = lgb.train(params, dtrain, num_boost_round=30)
        preds[:, q_i] = base_te[:, 0] + booster.predict(X_te)
    return enforce_monotone_quantiles(preds)


def fit_predict_causal_residuals(
    y_past: np.ndarray,
    x_past: np.ndarray,
    x_future: np.ndarray,
    univariate_forecaster: Callable[[np.ndarray, int], np.ndarray],
    head: str = "ridge",
    quantiles: tuple[float, ...] = QUANTILE_LEVELS,
    cfg: CausalResidualConfig | None = None,
) -> np.ndarray:
    """End-to-end strictly causal rolling-origin univariate TSFM + physical-unit residual predictor.

    Parameters
    ----------
    y_past : shape [T_cutoff] (STRICTLY <= T_cutoff; post-cutoff y_future is NEVER passed!)
    x_past : shape [T_cutoff, D_cov]
    x_future : shape [H, D_cov]
    univariate_forecaster : fn(y_subcontext_1d, H) -> quantile forecast [H, len(quantiles)]
    head : 'ridge' | 'qlinear' | 'lgbm'
    """
    cfg = cfg or CausalResidualConfig()
    yp = np.asarray(y_past, dtype=np.float64).reshape(-1)
    h_out = x_future.shape[0]

    # Forecast final test horizon first with univariate TSFM on y_past
    uni_test_q = enforce_monotone_quantiles(univariate_forecaster(yp, h_out))

    folds = generate_rolling_origin_folds(
        t_cutoff=len(yp),
        horizon=h_out,
        max_folds=cfg.max_rolling_folds,
        min_subcontext=cfg.min_subcontext_len,
    )
    if not folds:
        return uni_test_q

    X_past_feat, X_fut_feat = build_causal_covariate_features(x_past, x_future)

    X_rows = []
    r_med_rows = []
    med_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    for orig_t, end_t in folds:
        sub_y = yp[:orig_t]
        fold_uni_q = enforce_monotone_quantiles(univariate_forecaster(sub_y, end_t - orig_t))
        actual_fold_y = yp[orig_t:end_t]
        X_rows.append(X_past_feat[orig_t:end_t])
        r_med_rows.append(actual_fold_y - fold_uni_q[:, med_idx])

    X_train = np.vstack(X_rows)
    r_med = np.concatenate(r_med_rows)

    if len(r_med) < cfg.min_residual_samples:
        return uni_test_q

    if head == "ridge":
        delta_q = fit_predict_ridge_gcv(X_train, r_med, X_fut_feat, quantiles=quantiles, clip_z=cfg.feature_clip_z)
        return enforce_monotone_quantiles(uni_test_q + delta_q)
    elif head == "qlinear":
        delta_q = fit_predict_quantile_linear(
            X_train, r_med, X_fut_feat, quantiles=quantiles, l2_reg=0.25, clip_z=cfg.feature_clip_z
        )
        return enforce_monotone_quantiles(uni_test_q[:, med_idx : med_idx + 1] + delta_q)
    elif head == "lgbm":
        delta_q = fit_predict_lgbm_quantile(
            X_train, r_med, X_fut_feat, quantiles=quantiles, clip_z=cfg.feature_clip_z
        )
        return enforce_monotone_quantiles(uni_test_q[:, med_idx : med_idx + 1] + delta_q)
    else:
        raise ValueError(f"Unknown residual head: {head}")


def seasonal_naive_forecast(
    y_past: np.ndarray,
    horizon: int,
    seasonality: int = 1,
    quantiles: tuple[float, ...] = QUANTILE_LEVELS,
) -> np.ndarray:
    """Physical seasonal persistence baseline with empirical seasonal-difference quantile intervals."""
    yp = np.asarray(y_past, dtype=np.float64).reshape(-1)
    yp_clean = yp[np.isfinite(yp)]
    if yp_clean.size == 0:
        return np.zeros((horizon, len(quantiles)), dtype=np.float64)

    m = max(1, int(seasonality))
    if yp_clean.size < m:
        m = 1

    last_season = yp_clean[-m:]
    reps = (horizon + m - 1) // m
    base = np.tile(last_season, reps)[:horizon]

    if yp_clean.size > m:
        diffs = yp_clean[m:] - yp_clean[:-m]
        offsets = np.quantile(diffs, quantiles)
    else:
        offsets = np.zeros(len(quantiles), dtype=np.float64)

    return enforce_monotone_quantiles(base[:, None] + offsets[None, :])


def _seasonal_scale(y_past: np.ndarray, seasonality: int = 1, eps: float = 1e-6) -> float:
    yp = np.asarray(y_past, dtype=np.float64).reshape(-1)
    yp = yp[np.isfinite(yp)]
    m = max(1, int(seasonality))
    if yp.size <= m:
        m = 1
    if yp.size <= 1:
        return 1.0
    denom = float(np.mean(np.abs(yp[m:] - yp[:-m])))
    return max(denom, eps)


def scaled_quantile_loss(
    y_past: np.ndarray,
    y_true: np.ndarray,
    q_preds: np.ndarray,
    seasonality: int = 1,
    quantiles: tuple[float, ...] = QUANTILE_LEVELS,
) -> float:
    """Official fev-bench Scaled Quantile Loss (SQL = mean 2*pinball_loss / in-sample seasonal naive MAE)."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    qp = np.asarray(q_preds, dtype=np.float64)
    mask = np.isfinite(yt[:, 0])
    if not np.any(mask):
        return float("nan")
    yt = yt[mask]
    qp = qp[mask]
    qs = np.asarray(quantiles, dtype=np.float64)[None, :]
    err = yt - qp
    pinball_2x = 2.0 * np.maximum(qs * err, (qs - 1.0) * err)
    scale = _seasonal_scale(y_past, seasonality=seasonality)
    return float(np.mean(pinball_2x) / scale)


def mase_score(
    y_past: np.ndarray,
    y_true: np.ndarray,
    median_pred: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error (MASE) on the median (q=0.5) forecast."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mp = np.asarray(median_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt)
    if not np.any(mask):
        return float("nan")
    scale = _seasonal_scale(y_past, seasonality=seasonality)
    return float(np.mean(np.abs(yt[mask] - mp[mask])) / scale)


def paired_bootstrap_rel_reduction_ci(
    baseline_losses: list[float] | np.ndarray,
    candidate_losses: list[float] | np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """10,000-resample paired bootstrap 95% CI of relative % loss reduction `100 * (base - cand) / base`."""
    b = np.asarray(baseline_losses, dtype=np.float64)
    c = np.asarray(candidate_losses, dtype=np.float64)
    valid = np.isfinite(b) & np.isfinite(c) & (b > 1e-8)
    b, c = b[valid], c[valid]
    if b.size == 0:
        return 0.0, 0.0, 0.0
    # Use log-ratio / clipped per-series relative gain or ratio-of-means (ratio of geometric/trimmed means in fev)
    # Ratio of winsorized means prevents a single near-zero denominator series from dominating:
    b_cap = np.clip(b, 1e-4, np.quantile(b, 0.99) * 2.0 if b.size > 10 else 1e6)
    c_cap = np.clip(c, 1e-4, np.quantile(c, 0.99) * 2.0 if c.size > 10 else 1e6)

    rng = np.random.default_rng(seed)
    point = 100.0 * (float(np.mean(b_cap)) - float(np.mean(c_cap))) / max(float(np.mean(b_cap)), 1e-8)
    idx = rng.integers(0, b_cap.size, size=(n_boot, b_cap.size))
    b_means = np.mean(b_cap[idx], axis=1)
    c_means = np.mean(c_cap[idx], axis=1)
    gains = 100.0 * (b_means - c_means) / np.maximum(b_means, 1e-8)
    return float(point), float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))
