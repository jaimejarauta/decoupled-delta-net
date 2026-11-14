# RESULTS - redispatch-break

Generated 2026-11-13 17:01 by run_all.py. Data through 2026-11-13.

## Overall verdict: **MIXED**

Validity: V1_matches_redata=True, V2_positive_control=True, V3_placebo=True, V4_coverage=True
Hypotheses: H1_break=True, H2_breakdown=False, H2_adaptation=False, H3_skill=True

## Registered numbers

- H1 actual / old-regime counterfactual, 2025-05-01..2026-04-30: 2.03 [1.95, 2.15] (28206 vs 13881 GWh); ridge counterfactual ratio 1.80. Placebo ratio 1.11.
- H2 breakdown: `wx|gbm_d|expanding` MAE in the shock window / its MAE in the last 180 dev days = 1.16.
- H2 adaptation: days to recover vs `naive_last`: expanding 66, ew30 64.
- H3 pre-market skill, post window (n=489): `online_t0` vs `naive_last` = 0.104 [0.037, 0.170], DM t -2.92, p 0.0035.
- Not gated: post-market `online_t1` skill vs the same baseline 0.270.

## MAE (GWh/day) by period

| forecast | dev | shock | post |
|---|---|---|---|
| naive_last | 7.12 | 6.30 | 9.98 |
| naive_week | 10.31 | 16.87 | 13.31 |
| naive_m7 | 8.28 | 8.67 | 10.44 |
| online_naive | 7.37 | 6.30 | 9.88 |
| online_t0 | 6.04 | 7.02 | 8.94 |
| online_t1 | 5.32 | 7.55 | 7.29 |
| lags|gbm_d|dummy | 6.21 | 6.65 | 9.51 |
| lags|gbm_d|ew30 | 6.72 | 6.45 | 9.77 |
| lags|gbm_d|ew90 | 6.22 | 6.11 | 9.60 |
| lags|gbm_d|expanding | 6.21 | 6.67 | 9.53 |
| lags|gbm_d|roll365 | 6.30 | 6.69 | 9.73 |
| lags|gbm_d|roll90 | 6.33 | 6.12 | 9.93 |
| lags|gbm|dummy | 6.79 | 12.19 | 10.25 |
| lags|gbm|ew30 | 8.37 | 12.81 | 10.20 |
| lags|gbm|ew90 | 7.20 | 11.98 | 10.06 |
| lags|gbm|expanding | 6.79 | 12.38 | 10.21 |
| lags|gbm|roll365 | 6.98 | 14.96 | 10.54 |
| lags|gbm|roll90 | 7.44 | 13.80 | 10.85 |
| lags|ridge|dummy | 6.06 | 6.70 | 9.76 |
| lags|ridge|ew30 | 6.59 | 10.25 | 9.56 |
| lags|ridge|ew90 | 6.04 | 9.79 | 9.37 |
| lags|ridge|expanding | 6.06 | 8.12 | 9.76 |
| lags|ridge|roll365 | 6.13 | 10.09 | 9.80 |
| lags|ridge|roll90 | 7.05 | 10.54 | 11.18 |
| mkt|gbm_d|dummy | 5.39 | 6.79 | 8.90 |
| mkt|gbm_d|ew30 | 5.91 | 6.39 | 8.35 |
| mkt|gbm_d|ew90 | 5.64 | 6.04 | 8.11 |
| mkt|gbm_d|expanding | 5.39 | 6.87 | 8.99 |
| mkt|gbm_d|roll365 | 5.55 | 6.79 | 8.59 |
| mkt|gbm_d|roll90 | 5.91 | 5.65 | 8.65 |
| mkt|gbm|dummy | 5.94 | 14.54 | 9.46 |
| mkt|gbm|ew30 | 7.09 | 9.43 | 8.19 |
| mkt|gbm|ew90 | 5.93 | 12.71 | 8.55 |
| mkt|gbm|expanding | 5.94 | 15.29 | 9.54 |
| mkt|gbm|roll365 | 6.03 | 15.57 | 9.56 |
| mkt|gbm|roll90 | 6.78 | 12.43 | 9.09 |
| mkt|ridge|dummy | 5.46 | 6.33 | 8.63 |
| mkt|ridge|ew30 | 5.88 | 9.20 | 7.13 |
| mkt|ridge|ew90 | 5.45 | 8.46 | 7.09 |
| mkt|ridge|expanding | 5.46 | 8.47 | 8.57 |
| mkt|ridge|roll365 | 5.55 | 9.21 | 7.38 |
| mkt|ridge|roll90 | 6.89 | 10.60 | 8.49 |
| wx|gbm_d|dummy | 5.75 | 6.67 | 9.48 |
| wx|gbm_d|ew30 | 6.44 | 7.15 | 9.12 |
| wx|gbm_d|ew90 | 5.79 | 6.21 | 9.11 |
| wx|gbm_d|expanding | 5.75 | 6.65 | 9.44 |
| wx|gbm_d|roll365 | 5.84 | 6.58 | 9.41 |
| wx|gbm_d|roll90 | 6.07 | 6.17 | 9.13 |
| wx|gbm|dummy | 6.38 | 12.86 | 10.04 |
| wx|gbm|ew30 | 7.86 | 12.34 | 9.48 |
| wx|gbm|ew90 | 6.62 | 11.77 | 9.46 |
| wx|gbm|expanding | 6.38 | 12.98 | 10.12 |
| wx|gbm|roll365 | 6.53 | 14.99 | 10.60 |
| wx|gbm|roll90 | 7.33 | 13.74 | 10.18 |
| wx|ridge|dummy | 6.08 | 6.90 | 9.64 |
| wx|ridge|ew30 | 6.56 | 9.62 | 9.13 |
| wx|ridge|ew90 | 6.13 | 9.49 | 9.25 |
| wx|ridge|expanding | 6.08 | 9.02 | 9.64 |
| wx|ridge|roll365 | 6.15 | 10.22 | 9.59 |
| wx|ridge|roll90 | 7.23 | 9.71 | 11.51 |

Scored days: dev 481, shock 60, post 489. Mean target: dev 37.1, shock 76.2, post 75.9 GWh/day.

## Recovery days after the break, by scheme (model family selected on dev)

- `wx|gbm_d|expanding`: 66
- `wx|gbm_d|roll90`: 62
- `wx|gbm_d|roll365`: 64
- `wx|gbm_d|ew30`: 64
- `wx|gbm_d|ew90`: 31
- `wx|gbm_d|dummy`: 67

## Phase-I upward redispatch, GWh/day, 2024-05-01..2025-04-27 vs 2025-05-01..2026-04-30

| split | key | before | after |
|---|---|---|---|
| type | rtd | 5.60 | 9.92 |
| type | sot | 12.51 | 19.71 |
| type | sca | 8.26 | 14.26 |
| type | dan | 0.01 | 0.33 |
| type | scb | 0.08 | 0.16 |
| type | cot | 8.92 | 26.19 |
| type | pev | 0.07 | 0.00 |
| type | rsi | 0.00 | 6.68 |
| tech | carbon | 5.30 | 0.97 |
| tech | ciclo_combinado | 25.30 | 61.98 |
| tech | consumo_bombeo | 0.04 | 0.17 |
| tech | cogeneracion | 0.09 | 0.15 |
| tech | nuclear | 4.68 | 9.45 |
| tech | fuel_gas | 0.00 | 4.47 |

## Gate (verbatim from README)

## PRE-REGISTERED GATE (frozen 2026-11-13, before any post-break forecast was scored; never edited afterwards)

All numbers are computed by `run_all.py test` (`verdict`). Thresholds were set before the development run and
were not changed by it.

- **Validity** (else INVALID, nothing is interpreted):
  - V1: in at least 95% of complete months, the parsed Phase-I upward energy is within 2% of REData series 701.
  - V2 (positive control): on the dev period, the best `mkt` configuration has skill >= 0.15 over the best naive
    baseline. The market schedule must be informative, or the pipeline is suspect.
  - V3: the placebo counterfactual ratio (actual / predicted, 2024-05-01..2025-04-27) lies in [0.80, 1.25].
  - V4: the target exists on at least 95% of days in 2025-05-01..2026-10-31.
- **H1 (break, a check against published estimates)**: actual / old-regime counterfactual over
  2025-05-01..2026-04-30 is >= 1.5 and the lower end of its 95% block-bootstrap interval is > 1.2.
- **H2 (breakdown and adaptation)**, for the model family and information set selected on dev at the pre-market
  issue time (`lags` or `wx`), both required:
  - breakdown: with the `expanding` scheme, MAE in the shock window is >= 1.5 x its MAE in the last 180 dev days;
  - adaptation: the `ew30` scheme recovers at least 14 days sooner than `expanding`.
- **H3 (pre-market skill after the break)**: on the post window, `online_t0` (online selection over all `lags`
  and `wx` configurations) has skill >= 0.10 over the naive baseline that is best on that same window (chosen in
  hindsight, which favours the baseline), with Diebold-Mariano p < 0.05 in its favour.
- **Overall**: INVALID if any validity condition fails. Otherwise PASS if H1, H2 and H3 hold, FAIL if none holds,
  MIXED otherwise. Each hypothesis is reported separately whatever the overall label.
- Reported, not gated: post-market (`mkt`) skill on the post window; recovery days of every scheme; MAE of every
  configuration in every period; Phase-I energy by restriction-type code and by technology before and after;
  ridge counterfactual as a robustness check.
- Predictions written before any post-break forecast was scored: H1 holds (ratio about 2; public totals already
  suggest it). H2 breakdown holds for `gbm`, less clearly for `ridge` and `gbm_d`, whose lag terms follow a level
  shift; adaptation gain uncertain. H3 uncertain: after the break the target may be steadier and persistence
  harder to beat.
