# redispatch-break

Question: after the Iberian blackout of 28 April 2025, Red Eléctrica changed how it schedules security
redispatch ("operación reforzada"). (1) How do day-ahead forecasts of Spain's daily Phase-I upward
technical-constraint energy break down at that regime change, and which adaptation scheme recovers fastest?
(2) How much skill over naive baselines is available before the day-ahead market clears, and how much more once
the market schedule is public? (3) As a check against published estimates, how large is the break relative to an
old-regime counterfactual?

Target venue: IEEE PowerTech 2027 (Bruges), 6 pages, full paper due 2027-02-28. Topics "Forecasting, detecting and
managing congestion" and "Artificial intelligence and data analytics".

What the code does: builds one row per delivery day from public REE files, produces rolling-origin forecasts for
54 model configurations (3 information sets x 3 models x 6 training schemes) plus 3 naive baselines, and scores
them before, during and after the break.

## Prior art (checked 2026-11-13, before any code; second pass by a separate search agent the same day)

| Work | What it did | Delta to this project |
|---|---|---|
| Davi-Arderius & Graf, "Iberian Peninsula Blackout: A Trade-Off Between Operational Risk and Consumer Cost", SSRN 5429395 (2025, rev. 2026) | event study on hourly data 2019-2025: more gas procured, more curtailment, higher redispatch cost and CO2 after the blackout | **the break is already measured.** No forecasting, no adaptation schemes, no information-timing analysis. Our part (3) is a check against it, not a new claim |
| Davi-Arderius, Jamasb & Rosellon, Applied Energy 382 (2025) | Spanish redispatch volumes x2 and costs x9 over 2019-2023; voltage control about half of volume | pre-blackout, descriptive |
| Ventosa Pontes, "Modelos de predicción de restricciones técnicas", Comillas ICAI master's thesis (2021) | RF / KNN / MLP forecasts of system constraint volume | **direct predecessor.** Pre-blackout, not open, no regime change |
| Shamim & Shamim, arXiv 2610.08337 (2026) | probabilistic forecasts of German daily redispatch energy under reporting delays; rolling calibration helps under gradual drift | gradual drift, no discrete policy break, no comparison of retraining schemes. Germany |
| S. A. Mansouri, GitHub "Spain-Generation-Redispatch-Forecasting" (2026) | per-unit up/down LightGBM classifier, 60-day half-life weights, trained from Oct 2025 only | no baselines, no pre-break data, no paper |
| Industry estimates (PwC, NERA, REE statements, Ember) | cost of reinforced operation, EUR 0.5-1.1 bn per year depending on the definition | no forecasting, no formal counterfactual |

Not opened in full: the SSRN paper (abstract and slides only), the 2021 thesis (abstract only). No redispatch
forecasting paper was found in PowerTech 2025, PSCC 2026, EEM or ISGT Europe, but search coverage of those
proceedings is weak; check IEEE Xplore before submission.

Honest position: the size of the break is published. The new parts are the forecasting benchmark across a
discrete operating-policy change, the comparison of adaptation schemes by recovery time, and the value of the
market schedule as information. The method is standard; the contribution is the measurement and an open,
token-free data pipeline.

## Data (all public, no token)

- **Target `y`**: Phase-I upward energy of the PDBF technical-constraint solution (P.O. 3.2 code `UPOPVPV` and any
  other non-`ECO` upward row), GWh per delivery day, from sheet "redispatch by technology and restriction type" of
  the ESIOS `I3DIA` workbook (archive 32).
- **Market schedule**: PDBF energy by technology (market + bilateral) from the same workbook. Scheduled
  generation `gen` is the sum of the generation rows; the demand rows changed name in May 2024 and are not used.
- **Cross-checks**: ESIOS `totalrpdvpprec` XML (archive 19; total upward energy of Phase I + II and its
  publication timestamp); REE REData series 701 "Energía a subir restricciones técnicas al PDBF (fase I)", monthly.
- **Weather**: Open-Meteo previous-runs archive, forecasts made two days before the valid time, mean of 8 points.
  Wind and radiation exist only from 2024; earlier rows are missing values.
- Period: delivery days from 2023-01-01. Scored forecasts start 2024-01-01.
- File format: REE serves I3DIA as Excel workbooks until September 2026 and as one CSV per sheet from October
  2026; both are read. From October 2025 the real-time sheet has the same columns as the PDBF sheet, so rows
  are assigned by their P.O. 3.2 code (`UPOPVP*` = Phase I upward).

Information sets (row D uses only what is public at the issue time; `test_rb.py` checks this):
- `lags`, issue D-1 10:00: calendar, target up to D-1, PDBF by technology up to D-1.
- `wx`: `lags` plus the weather forecast for D.
- `mkt`, issue D-1 13:30, after the PDBF is published and before the constraint solution (14:45): `wx` plus
  the PDBF of D by technology.

Assumptions to state in the paper:
- The I3DIA workbook appears at D+3, but the numbers used are public earlier: the Phase-I solution of day D-1
  is published on D-2 (XML timestamps confirm it), and the PDBF is published per programming unit by 13:30 on
  D-1. Whether ESIOS also publishes the PDBF *by technology* at that time was not confirmed; a forecaster can
  aggregate units with the public unit list.
- The meaning of the restriction-type codes (RTD, RTT, SOT, SCA, SCB, COT, PEV, RSI and others) was not found
  in P.O. 3.2 or P.O. 9. They are reported as raw codes and not interpreted.

## Design

- Models: `ridge` (standardised, alpha 1), `gbm` (sklearn HistGradientBoosting, absolute-error loss, 300 trees,
  8 leaves, fixed), `gbm_d` (same trees on the change from yesterday, because trees cannot extrapolate a level shift).
- Training schemes: `expanding`; `roll90`, `roll365` (last N days); `ew30`, `ew90` (weight halves every N days);
  `dummy` (expanding plus a post-break indicator).
- Rolling origin: refit every 7 days on days strictly before the block; lag features update daily.
- Naive baselines: yesterday (`naive_last`), same weekday last week (`naive_week`), mean of the last 7 days.
- `online_*`: each day follow the configuration with the lowest mean absolute error over the previous 56 days.
- Counterfactual: `gbm` on calendar + the day's PDBF by technology (no target lags, which carry the regime),
  trained on days up to 2025-04-27, applied to 2025-05-01..2026-04-30. Placebo: train to 2024-04-27, apply to
  2024-05-01..2025-04-27.
- Periods: dev 2024-01-01..2025-04-27; shock 2025-05-01..2025-06-29; post 2025-06-30..2026-10-31. The blackout
  day and the two days after it are never scored.
- Metrics: MAE (GWh/day); skill = 1 - MAE / MAE of the baseline; 95% moving-block bootstrap intervals (7-day
  blocks); Diebold-Mariano on absolute errors with a Newey-West variance (7 lags).
- Recovery day of a forecast: days after 2025-05-01 until its trailing-14-day MAE is no worse than that of
  `naive_last` for 14 days in a row (365 if never).

Development rule: `run_all.py dev` loads the table truncated at 2025-04-27. Everything chosen by looking at
data (model family, baseline) is chosen there and written to `data/dev.json`.

What was already known about the post-break period when the gate was written (public aggregates only): REData
monthly cost components (constraint cost per MWh roughly doubled after April 2025 and stayed high in 2026), press
and Substack totals, and one sample day (2025-06-10, 69 GWh) used to check the file format.

## Development run (2026-11-13, delivery days up to 2025-04-27 only; written before the test run)

- Data check on the whole table: 43 of 43 complete months match REData series 701 within 2% (largest gap
  0.009%). Getting there needed two fixes to the target definition (rows by P.O. 3.2 code, CSV reader); these
  used monthly totals of the post-break period, never a forecast or an error.
- Missing target days: 2024-06-26, 2024-11-09, 2025-04-29, and the last days not yet published.
- Scored days 481, mean target 37.1 GWh/day. MAE (GWh/day): `naive_last` 7.12, `naive_m7` 8.28, `naive_week` 10.31.
- Selected on dev: naive `naive_last`; pre-market `wx|gbm_d|expanding` (MAE 5.75, skill 0.192); post-market
  `mkt|gbm_d|expanding` (MAE 5.39, skill 0.243). V2 holds (0.243 >= 0.15).
- Placebo counterfactual ratio: gbm 1.106 [1.063, 1.153], ridge 1.040 [1.007, 1.072]. V3 holds, with a 4-11%
  upward drift between consecutive years that the H1 thresholds (1.5 and 1.2) have to clear.
- Consequence for H2: the family selected on dev is `gbm_d`, so H2 compares `wx|gbm_d|expanding` with `wx|gbm_d|ew30`.

## Files
- `rb.py` - download, parsers, daily table, features, models, backtest, metrics.
- `run_all.py` - stages `data`, `dev`, `test`; gate logic in `verdict`; writes `RESULTS.md`.
- `test_rb.py` - pytest (no network): parsers, no-look-ahead checks, metrics, gate rules.
- Run: `uv run --no-project --python 3.12 --with-requirements requirements.txt python run_all.py data|dev|test`.

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
