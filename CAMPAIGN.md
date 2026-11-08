# CAMPAIGN - vla-reachability (Colab L4, one "Run all")

Notebook: `notebooks/colab_runner_vla_reachability.ipynb` -> `run_all.py`. Results: Drive
`MyDrive/moonshot_results/vla-reachability/` (`$VR_RESULTS`); HF cache on Drive (`HF_HOME`) so checkpoints download
once. Re-running resumes: every eval / replicate is a separate JSON and is skipped when present. Gate: README.md
"FROZEN GATE" (2026-11-08).

GPU: **L4 24 GB** (pi0.5 3B in bf16 ~7 GB + SmolVLA 0.45B; A100 not justified: the run is CPU-simulation and
latency bound). The L4 VM's vCPUs run the MuJoCo/EGL workers (one env per process, `min(16, cpu_count)`).

## Stages

| # | stage | content | outputs |
|---|---|---|---|
| 0 | installs | `lerobot[libero,smolvla,pi]==0.7.0` (+ LIBERO assets ~0.4 GB on first env build) | - |
| 1 | pytest pre-flight, nothing skipped | exact cross-env restore, predicates on a real scene, pool, SmolVLA forward on GPU (seeded + stochastic) | `pytest.json` |
| 2 | pilot | SmolVLA 10 tasks x 10 ep; SmolVLA-HF and pi0.5 2 tasks x 8 ep; 1 splitting replicate (N=20); drop rule; validity; abandon flags; budget ladder | `pilot.json` |
| 3 | stage A | every unit: 100 standard episodes + 2 splitting replicates (N=50, 6 levels) | `eval_*.json`, `split_*.json` |
| 4 | stage B | calibration units -> 4 replicates (+300 MC top-up if < 3 units have >= 5 events); efficiency set (4 SmolVLA-family units) -> 10 replicates | `topup_*.json`, `split_*.json`, `sets.json` |
| 5 | analysis | frozen gate | `verdict.json`, `RESULTS.md` |

## Budget (to be replaced by the pilot's measured numbers in `pilot.json["plan"]`)

Measured locally (M4, loaded machine): LIBERO step with physics + two 64x64 renders ~20-30 ms per env per step.
Assumed on Colab: ~15 ms/step/process with EGL, 8-12 worker processes -> ~400-600 env steps/s from the simulator;
SmolVLA ~0.1 s per batched decision (batch 48), pi0.5 ~1 s per decision (batch 16) -> pi0.5 is the bottleneck
(~250 env steps/s at K = 20).

Per unit: standard eval 100 x ~150 steps = 15k steps; one splitting replicate ~6 stages x 50 particles x ~80 steps
= ~24k steps -> stage A ~63k steps per unit. 30 units ~1.9M steps -> ~1.5-2.5 h; stage B ~0.5 M steps (SmolVLA) ->
~0.3 h. Installs + downloads ~0.3 h, pilot ~0.4 h.
**Total estimate: 2.5-3.5 h L4 = 10-13 CU (cap 15 CU).** The pilot projects the main run from measured sec/step and
applies the pre-registered ladder (drop SmolVLA-HF, then tasks 0-4) to stay <= 11 CU for main; if even the last
rung exceeds it, `run_all.py` stops before the main run.

## Decision tree

- **INVALID** (pytest or pilot SR < 0.5): check `pytest.json`, the pilot SR per task in `pilot.json`; usual suspects are
  observation conventions (image flip, rename map `image -> camera1` for `lerobot/smolvla_libero`), action
  post-processing. Fix, dated README note, fresh `VR_RESULTS`.
- **Abandon signal at pilot** (`pilot.json["abandon"]`): `not_rare` -> severe events are common at these thresholds;
  splitting cannot win; report MC numbers and stop (no main). `uninformative` -> the score never approaches 1; the
  hypothesis "rare events hidden by standard eval" is untestable at these predicates; stop and report.
- **PASS**: splitting is a calibrated >= 5x estimator and severe risk ranks units differently from SR. Next: libero_10
  (long horizon), add pi0 / OpenVLA-OFT, adaptive levels (AMS) and a learned score (value function on the VLA
  features) to raise E; write-up as "reachability evaluation for VLAs".
- **MIXED**: report which leg failed. G2 WEAK -> score xi is the bottleneck (inspect per-stage q in `split_*.json`;
  stages with q > 0.9 waste budget, q < 0.05 starve); G3 FAIL with G2 PASS -> the method is useful but severe risk
  tracks failure rate; frame as an efficient estimator only.
- **FAIL**: G1 FAIL -> restore/branching is not faithful (should not happen given the exact-restore test; check halting
  and success termination); kill rule -> severe reachability is the failure rate and splitting does not help: close.

## Scale-up path (A6000 / Blackwell, after a PASS or MIXED with G2 >= WEAK)

Same code; raise worker count (CPU-bound simulation) and policy batch size; add LIBERO-10 and LIBERO-Spatial suites,
pi0 / OpenVLA-OFT checkpoints (>= 6 policies) so that rho is estimated over >= 60 units; N = 100 particles, 10
replicates on every unit. Expected ~4x the compute of this campaign.
