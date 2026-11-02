# Colab campaign (README gate + amendment A9)

Run: open `notebooks/colab_runner_grok_thermostat.ipynb` in Colab on an L4 (or A100) → Runtime → Run all.
Locally / any GPU box: `python campaign.py` (writes `results/`). Pipeline check: `SMOKE=1 python campaign.py`
(CPU, d=32, T=100, 2 seeds, writes `results_smoke/`; not evidence).

## Stages (all decisions automatic, no human look in between)

| # | Stage | Run(s) | Models × steps | Decision rule |
|---|---|---|---|---|
| 1 | Calibration | `cal_{task}` | 5 × 10k per task (WD=1, dev seeds 100–104, frac 0.3 / parity n=1000) | median steps-to-95% ≤ T, else frac += 0.1 (parity n=1500) once |
| 2 | Dev WD grid | `dev_{task}` | 20 × 10k per task (WD {0.1,0.3,1,3} × dev seeds) | dev-picked B1 (secondary only) |
| 3 | H1 selection | (uses `dev_add`) | – | signal × κ maximizing fraction in useful window |
| 4 | Controller selection | `devctrl_add` | 50 × 10k (10 configs × dev seeds) | lowest dev median, ties → first listed; frozen in `selection.json` |
| 5 | Eval | `eval_{task}` | 50 × 10k per task (B1×4, B2, B3×4, ctrl × seeds 0–4) | – |
| 6 | Analysis | – | – | `verdict.json`, `RESULTS.md` |

Resume: each run is saved atomically to `results/<run>.json` when it finishes; on restart finished runs are
skipped and an unfinished one restarts from step 0 (partial checkpoints every 2000 steps carry `"partial": true`
and are ignored). Decisions are recomputed deterministically from the saved runs. In Colab, `results/` is a
symlink to `MyDrive/moonshot_results/grok-thermostat`, so just "Run all" again after a disconnect.

## Expected GPU time (L4)

13 batched runs, ≤ 10k steps each (≈130k batched steps upper bound; runs stop early once all models grok).
At d=128 a step is ~25 GFLOP for the largest runs (M=50 models × batch 512), i.e. a few ms of TF32 compute;
the hand-written per-parameter AdamW + Grokfast make the step launch/Python-bound at roughly 5–15 ms on L4.
Logging (full train + test eval and CPU SVD every 50 steps) adds ~1 ms/step amortized.
**Estimate: ≈0.5 h expected, ≤1.5 h pessimistic on an L4** (an A100 is not much faster: launch-bound).
Peak memory is < 3 GB (measured on CPU for M=50, d=128, full-train eval), far below 24 GB.
This is an estimate, not a GPU measurement: each run logs its ms/step (`[done] ... ms/step`) in
`results/campaign.log`; the first `cal_add` line tells you the real rate within a minute.

## Output files (`results/`)

| File | Meaning |
|---|---|
| `cal_{task}.json`, `dev_{task}.json`, `devctrl_add.json`, `eval_{task}.json` | raw batched runs: per-model config (seeds, wds, gf, ctrl) and logs every 50 steps (train loss/acc, test acc, ‖θ‖, embedding erank, grad norm, wd) |
| `calibration.json` | per task: median WD=1 steps, whether frac was raised, final frac |
| `selection.json` | frozen choices before eval: fracs, dev-picked B1 WDs, H1 signal/κ + dev table, controller config + dev medians |
| `verdict.json` | H1 and H2 verdicts, per-task arm medians, best baseline, R, bootstrap CI, secondary R vs dev-picked B1, H1 held-out score and lead fractions |
| `RESULTS.md` | human-readable summary of `verdict.json` |
| `campaign.log` | stdout of the run (Colab) |

## Gate (unchanged; README "PRE-REGISTERED GATE" as amended by A9)

- **H1 PASS** iff on held-out tasks div, quad, parity (eval-seed B1 grid runs, eligible = groks within T and
  D = t_grok − t_mem ≥ 500) ≥ 80% of eligible runs fire in [t_mem + 0.2D, t_grok − 0.2D] with the signal/κ
  chosen on add dev runs. Otherwise FAIL.
- **H2**: R = median(ctrl) / median(best of B1 ∪ B2 ∪ B3), censored = 2T, bootstrap CI (10k resamples).
  PASS iff R ≤ 0.5 and CI upper < 1 on ≥ 3 of 4 tasks; MIXED iff R < 1 and CI upper < 1 on ≥ 3 of 4; else FAIL.

## Decision tree

- **H2 PASS** → the result contradicts the expectation that tuned constant WD / Grokfast are hard to beat.
  Before any claim: re-run eval with 10 fresh seeds (e.g. 10–19) with the frozen config, same code, and
  check the result holds; then write up against CvAdamW (arXiv 2607.20552) with the 4-task, strong-baseline
  framing. Only then consider scaling (deeper models, non-modular tasks).
- **H2 MIXED** → controller is reliably faster but < 2×. Report as a negative-for-the-pre-registered-bar
  result with the effect size; a closed-loop WD paper is not justified on this alone. Optionally one
  follow-up: a new pre-registration (new date, new seeds) for the secondary dev-picked-B1 comparison,
  which is the practical setting.
- **H2 FAIL** → kill the controller line (as planned for the moonshot portfolio). Publish the negative result
  in short form (pre-registration + table): closed-loop WD from train loss / norm contraction does not beat
  tuned constant WD or Grokfast across 4 tasks.
- **H1** is reported independently: PASS → note the signal as a replication of prior early-warning work;
  FAIL → report as a negative replication. Neither changes the H2 decision.
- If calibration leaves a task where no arm groks (all R = 1 [1, 1]), that task counts against PASS by design
  (A9 §4); report it, do not re-tune.
