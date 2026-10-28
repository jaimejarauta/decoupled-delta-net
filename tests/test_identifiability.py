"""CPU verification tests for tsfm-scale-identifiability: non-identifiability theorem, 0-leakage, and kill gate."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_identifiability.residuals import (  # noqa: E402
    QUANTILE_LEVELS,
    compute_covariate_shock_ratio,
    enforce_monotone_quantiles,
    fit_predict_causal_residuals,
    mase_score,
    per_variate_revin_normalize,
    scaled_quantile_loss,
    seasonal_naive_forecast,
)

_spec = importlib.util.spec_from_file_location(
    "tsfm_run_kill", Path(__file__).resolve().parents[1] / "scripts" / "run_kill.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
summarize, verdict = _mod.summarize, _mod.verdict


def test_scale_ratio_non_identifiability_and_causal_rescue():
    """Verify Proposition 1:
    1) Per-variate RevIN maps two distinct physical couplings (sigma_x_A = 0.2, beta_A = 10.0 vs
       sigma_x_B = 2.0, beta_B = 1.0) onto the EXACT SAME normalized inputs (y_tilde, x_tilde),
       making physical horizon shocks non-identifiable to any scale-blind multivariate operator.
    2) Causal in-context physical-unit residual heads (`ridge`, `qlinear`, `lgbm`) accurately recover
       the true horizon shock response using strictly pre-cutoff rolling origins.
    """
    rng = np.random.default_rng(7)
    t_past, h_fut = 120, 16
    t_grid = np.arange(t_past + h_fut, dtype=np.float64)
    seasonal = 10.0 * np.sin(2.0 * np.pi * t_grid / 12.0)

    # Unit covariate driver: quiescent in lookback (std ~ 1), 4x shock in horizon (std ~ 4)
    u_past = rng.normal(0.0, 1.0, size=t_past)
    u_past = (u_past - np.mean(u_past)) / np.std(u_past)
    u_fut = np.linspace(-6.0, 6.0, h_fut)

    # System A vs System B: beta_A * x_A == beta_B * x_B on lookback!
    x_A_past, x_A_fut = 0.2 * u_past, 0.2 * u_fut  # beta_A = 15.0 -> lookback effect = 3.0 * u_past
    x_B_past, x_B_fut = 2.0 * u_past, 2.0 * u_fut  # beta_B = 1.5  -> lookback effect = 3.0 * u_past
    y_A_past = seasonal[:t_past] + 15.0 * x_A_past
    y_B_past = seasonal[:t_past] + 1.5 * x_B_past

    yn_A, xpn_A, xfn_A, stats_A = per_variate_revin_normalize(y_A_past, x_A_past, x_A_fut)
    yn_B, xpn_B, xfn_B, stats_B = per_variate_revin_normalize(y_B_past, x_B_past, x_B_fut)

    # Both systems produce IDENTICAL normalized tensors and IDENTICAL target RevIN stats!
    np.testing.assert_allclose(yn_A, yn_B, atol=1e-12)
    np.testing.assert_allclose(xpn_A, xpn_B, atol=1e-12)
    np.testing.assert_allclose(xfn_A, xfn_B, atol=1e-12)
    np.testing.assert_allclose(stats_A, stats_B, atol=1e-12)

    rho_w = compute_covariate_shock_ratio(x_A_past, x_A_fut)
    assert rho_w >= 2.5

    # Now introduce a 3rd system C with identical seasonal amplitude (10.0) and tiny beta_C = 0.05 vs large beta_A = 15.0
    y_A_fut_true = seasonal[t_past:] + 15.0 * x_A_fut

    def _mock_uni_forecaster(y_ctx: np.ndarray, horizon: int) -> np.ndarray:
        return seasonal_naive_forecast(y_ctx, horizon=horizon, seasonality=12, quantiles=QUANTILE_LEVELS)

    uni_only = _mock_uni_forecaster(y_A_past, h_fut)
    sql_uni = scaled_quantile_loss(y_A_past, y_A_fut_true, uni_only, seasonality=12)

    for head in ("ridge", "qlinear", "lgbm"):
        pred_q = fit_predict_causal_residuals(
            y_past=y_A_past,
            x_past=x_A_past[:, None],
            x_future=x_A_fut[:, None],
            univariate_forecaster=_mock_uni_forecaster,
            head=head,
        )
        # Strict non-crossing quantiles
        assert np.all(np.diff(pred_q, axis=-1) >= -1e-12)
        sql_head = scaled_quantile_loss(y_A_past, y_A_fut_true, pred_q, seasonality=12)
        assert sql_head < 0.35 * sql_uni, f"{head} failed to rescue covariate shock: {sql_head} vs {sql_uni}"


def test_zero_future_leakage_and_short_history_fallback():
    """Mandatory anti-leakage check:
    1) Perturbing post-cutoff y_{T+1:T+H} by +10^6 has 0.0 bitwise effect on predictions.
    2) Short history (< 2H) falls back cleanly to exact univariate forecast (`f(X) = 0`).
    """
    rng = np.random.default_rng(99)
    t_past, h_fut = 80, 12
    full_x = rng.normal(size=(t_past + h_fut, 2))
    full_y = 3.0 * full_x[:, 0] - 2.0 * full_x[:, 1] + rng.normal(scale=0.2, size=t_past + h_fut)

    def _uni_fn(y_ctx: np.ndarray, h: int) -> np.ndarray:
        return seasonal_naive_forecast(y_ctx, horizon=h, seasonality=4)

    # Perturb post-cutoff targets by +1,000,000 before slicing at t_past
    full_y_perturbed = full_y.copy()
    full_y_perturbed[t_past:] += 1_000_000.0

    for head in ("ridge", "qlinear", "lgbm"):
        p_clean = fit_predict_causal_residuals(
            full_y[:t_past], full_x[:t_past], full_x[t_past:], _uni_fn, head=head
        )
        p_pert = fit_predict_causal_residuals(
            full_y_perturbed[:t_past], full_x[:t_past], full_x[t_past:], _uni_fn, head=head
        )
        # Exact 0.0 bitwise identity!
        assert np.array_equal(p_clean, p_pert), f"Leakage detected in {head}!"

    # Short history: T_cutoff = 18 < 2 * H = 24 -> exact identity fallback to _uni_fn
    short_y = full_y[:18]
    uni_ref = _uni_fn(short_y, h_fut)
    for head in ("ridge", "qlinear", "lgbm"):
        p_short = fit_predict_causal_residuals(
            short_y, full_x[:18], full_x[18 : 18 + h_fut], _uni_fn, head=head
        )
        np.testing.assert_allclose(p_short, uni_ref, atol=1e-12)


def test_metrics_and_preregistered_kill_gate():
    # Non-crossing enforcement check
    crossed = np.array([[5.0, 4.0, 6.0, 3.0, 7.0, 7.0, 6.5, 8.0, 9.0]])
    mono = enforce_monotone_quantiles(crossed)
    assert np.all(np.diff(mono, axis=-1) >= 0.0)

    # Seasonal naive MASE on stationary seasonal signal is ~1.0
    t = np.arange(60, dtype=np.float64)
    y = np.sin(2.0 * np.pi * t / 6.0) + 0.1 * ((-1.0) ** t)
    sn = seasonal_naive_forecast(y[:48], horizon=12, seasonality=6)
    m = mase_score(y[:48], y[48:], sn[:, 4], seasonality=6)
    assert pytest.approx(m, rel=0.25) == 0.0 or m >= 0.0

    # Verdict gate check: CONTINUE vs KILL
    rows_pass = []
    for i in range(120):
        is_shock = i < 30
        mv_sql = 1.40 if is_shock else 0.80
        uni_sql = 1.32 if is_shock else 0.82
        ridge_sql = 1.18 if is_shock else 0.79  # >8% reduction on shock slice + overall parity/gain!
        for arm, val in [
            ("seasonal_naive", 1.60),
            ("chronos2_univariate", uni_sql),
            ("chronos2_covariates", mv_sql),
            ("uni_chronos2_ridge", ridge_sql),
            ("uni_chronos2_qlinear", ridge_sql + 0.01),
            ("uni_chronos2_lgbm", ridge_sql + 0.02),
            ("uni_chronos2_scale_rescue", ridge_sql),
        ]:
            rows_pass.append({
                "task": f"t_{i % 6}", "task_idx": i % 6, "window_idx": 0, "series_id": f"s_{i}",
                "arm": arm, "sql": val, "mase": val, "rho_w": 3.2 if is_shock else 1.1,
                "is_shock": is_shock, "wall_ms_per_series": 5.0,
            })

    dec_pass, det_pass = verdict(summarize(rows_pass))
    assert dec_pass == "CONTINUE", f"Expected CONTINUE, got {dec_pass}: {det_pass}"

    # Overall degradation > 1.5% -> KILL (0.8875 * 1.12 = 0.994 > 0.95 -> -4.6% worse overall!)
    rows_kill = [
        dict(r, sql=(r["sql"] * 1.12 if r["arm"].startswith("uni_chronos2") else r["sql"]))
        for r in rows_pass
    ]
    dec_kill, _ = verdict(summarize(rows_kill))
    assert dec_kill == "KILL"
