# `tsfm-scale-identifiability`: Scale-Ratio Non-Identifiability in Universal TSFMs

Formal and empirical audit of **per-variate reversible instance normalization (RevIN)** in covariate-aware Time-Series Foundation Models (`amazon/chronos-2` Group Attention) across all **46 real dynamic-covariate tasks** in [`fev-bench`](https://github.com/autogluon/fev).

## Core Falsifiable Claim
Independent per-variate lookback standardization $\tilde{y}_t = (y_t - \mu_y)/\sigma_y$ and $\tilde{x}_{j,t} = (x_{j,t} - \mu_{x_j,\text{ctx}})/\sigma_{x_j,\text{ctx}}$ erases the physical cross-variate sensitivity $\beta_j = \partial y_t / \partial x_{j,t} \mapsto \tilde{\beta}_j = \beta_j (\sigma_{x_j,\text{ctx}} / \sigma_y)$. On **Quiescent-to-Active Covariate Shocks** ($\rho_w = \max_j \sigma_{x_j,\text{hor}} / \sigma_{x_j,\text{ctx}} \ge 2.5$), native multivariate TSFM attention suffers severe variance/quantile miscalibration, whereas decoupling **Univariate `Chronos-2` + strictly causal in-context `RidgeCV` / `QuantileLinear` / `LightGBM` residual regression in physical units** matches or beats native Group Attention overall and reduces Scaled Quantile Loss (`SQL`) by $\ge 8\%$ on the shock slice.

## Evaluated Arms (Zero Future Leakage)
1. `seasonal_naive`: Physical seasonality floor.
2. `chronos2_univariate`: Zero-shot univariate `amazon/chronos-2` ignoring exogenous covariates.
3. `chronos2_covariates`: Official multivariate `amazon/chronos-2` Group Attention with known dynamic covariates.
4. `uni_chronos2_ridge`: Univariate `Chronos-2` + analytical SVD LOO-GCV `RidgeCV` on strictly causal rolling-origin (`t <= T_cutoff`) historical residuals.
5. `uni_chronos2_qlinear`: Univariate `Chronos-2` + L2-regularized pinball-loss quantile regression (`q=0.1..0.9`) with exact monotonization.
6. `uni_chronos2_lgbm`: Univariate `Chronos-2` + shallow `LightGBM` quantile residual booster.
7. `uni_chronos2_scale_rescue`: Diagnostic cross-series pooled residual model conditioning explicitly on $\log(1 + \sigma_{x,\text{ctx}} / \sigma_y)$.

## Quickstart
```bash
pytest -q
python scripts/run_kill.py run --max-tasks 46 --out results/kill.jsonl
python scripts/run_kill.py report --out results/kill.jsonl
```

