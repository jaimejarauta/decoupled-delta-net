# grok-thermostat

Question: (H1) does a statistic computed only from training data/weights give a usable early warning of
the test-accuracy jump in grokking, and (H2) can a closed-loop weight-decay controller driven only by
training signals reach 95% test accuracy faster than tuned baselines, with one config shared across tasks?

## Prior art (checked 2026-11-02) and delta

| Work | What it does | Delta here |
|---|---|---|
| Thermodynamic Weight Decay / CvAdamW (arXiv 2607.20552, 2026) | Closed-loop WD from attention-logit variance ("specific heat"). One task (a+b mod 97), compared to an untuned optimizer; 6% mean speedup, p=0.049, 10 seeds. | **Closest collision for H2** (closed-loop WD for grokking already exists). This project is not a new idea; it is a stricter test of the idea class: 4 tasks, oracle-tuned constant-WD and Grokfast(+WD) baselines, one frozen config, pre-registered gate. Our controller uses train loss / norm contraction, not attention variance. |
| Omnigrok (Liu et al. 2023); "Weight Norm Sets the Grokking Timescale" (arXiv 2606.13753) | Grokking time governed by weight norm; constraining/pinning the norm changes delay (exponential in pinned norm). | Norm-rate controller arm (C-norm) is a feedback version of norm steering; we test whether it beats tuned constant WD. |
| Grokfast (Lee et al. 2024) | Amplify slow (EMA) gradient components; with WD. | Used as baseline (recommended hp, and with tuned WD). |
| Notsawo et al. 2023 "Predicting grokking long before it happens"; commutator-defect / spectral-edge early-warning papers (arXiv 2602.16967 and companions); "Distributional spectral diagnostics" | Early-warning signals of grokking, mostly loss-curve spectra or weight-space geometry. Lead times reported but not with a pre-registered non-trivial window. | H1 here is a replication-style check of cheap train-only candidates with a window that excludes trivially firing at memorization. Not novel. |
| Weight-decay regimes / cheap online diagnostics (arXiv 2605.20441) | Attention-head diagnostics track WD regimes; no controller. | — |

Honest framing: neither H1 nor H2 is novel. The value is a pre-registered, strong-baseline falsification
test of "closed-loop WD beats tuned constant WD / Grokfast" across tasks.

## Setup (fixed before main experiment)

- Model (modular tasks): 1-layer transformer, d=64, 4 heads, MLP 256 (see amendment A1), no LayerNorm, tokens [a, b, =],
  readout at '='. Parity: 1-hidden-layer MLP, width 512, inputs ±1.
- Optimizer: AdamW written by hand (decoupled WD), lr 1e-3, betas (0.9, 0.98), minibatch 512, per-model WD.
  All arms × seeds are trained as one batched tensor program (identical code path).
- Tasks: add (a+b mod 97), div (a·b⁻¹ mod 97), quad (a²+ab+b² mod 97), parity (n=40, k=3).
  Train fractions / sizes: see "Calibration" below. Test: up to 2000 held-out examples per seed.
- Logged every 50 steps: full-train loss & acc, test acc, ‖θ‖₂, effective rank of embedding, minibatch grad norm, WD.
- steps-to-95% = first logged step with test acc ≥ 0.95. Never reached within budget T → censored (= ∞ for medians).
- Seeds: dev seeds 100–104 (dev task `add` only: H1 threshold selection and controller selection);
  eval seeds 0–4 on all 4 tasks (separate data splits and inits).

## Calibration rule (done before freezing; uses baseline WD runs only, never the controller)

T and train fractions are chosen so the default baseline (constant WD=1) groks within budget on dev seeds:
start at frac 0.3 (add, div), 0.3 (quad), n_train=1000 (parity); if median wd=1 run on dev seeds 100–104 does
not reach 95% within T, raise frac by 0.1 (parity: n_train ×1.5) once. Values used are recorded below
before any controller run on eval seeds.

## Calibration results and amendments (2026-11-02 ~18:55 CET, written BEFORE any controller run)

No controller run has been started at the time of writing. Gate thresholds below are unchanged.

A1. **Model width.** d=128 / MLP 512 was ~4x too slow on the shared M4 (load average 25–90 from other
agents). All runs use d=64, 4 heads, MLP 256. Runs use CPU with 2 threads (default threading was 10x
slower due to oversubscription).

A2. **Calibration at frac 0.3 (dev seeds 100–104, T=10000).** Runs on MPS stalled at ~1.6 s/step under
machine load and were stopped at step 6000 (logs: `runs/cal_{add,div,quad}.log`):
- add, WD grid {0.1,0.3,1,3}×5: 1/20 models at ≥95% test acc by step 6000; median train loss 0.0025.
- div, WD=1: 0/5 at 95%; median train loss 0.36 (not yet memorized); test acc 0.019.
- quad, WD=1: 0/5 at 95%; median train loss 0.20; test acc 0.17.
Applying the rule conservatively (median WD=1 run not at 95% by step 6000), frac is raised once to **0.4**
for add, div, quad. T stays 10000.
- parity, n_train=1000, WD=1: 0/5 at 95% in 10k steps (test acc 0.50). Raised to 1500: 0/5 (test acc 0.556).
  **Parity is excluded.**

A3. **Task exclusion rule (added because of A2).** A modular task at frac 0.4 is excluded only if no
constant-WD arm reaches 95% test acc in median on dev seeds within T. Excluded tasks are reported as such.

A4. **H2 gate with parity excluded:** H2 PASS iff R ≤ 0.5 and CI upper < 1.0 on **3 of 3** remaining tasks
(add, div, quad), minus any task excluded by A3 (then all remaining tasks). MIXED / FAIL definitions unchanged
(R < 1 with CI upper < 1 on all remaining tasks → MIXED).

A5. **Reduced arms (compute).** Per task, eval seeds 0–4: controller, B1, B2, B3 (20 models/task).
- B1 = constant WD, value picked by median steps-to-95% on **dev seeds 100–104** from {0.1,0.3,1,3}
  (dev-seed WD-grid run per task at frac 0.4), not on eval seeds.
- B2 = Grokfast-EMA α=0.98, λ=2, WD=0.005.
- B3 = Grokfast-EMA α=0.98, λ=2 with B1's dev-picked WD (no separate Grokfast×WD grid).
Best baseline = lowest eval median among B1, B2, B3 (this last choice still uses eval seeds, which favours
the baselines).

A6. **H1 held-out pool:** with parity excluded, held-out tasks are div and quad. Pool = their dev-seed
constant-WD grid runs (never used for selection) + eval-seed B1 runs. Signal and κ chosen on add dev-seed
WD-grid runs as pre-registered.

A7. Controller selection grid (10 configs) unchanged; run on add, frac 0.4, dev seeds.


## PRE-REGISTERED GATE (dated 2026-11-02; not to be edited after the main experiment starts)

### H1 – prediction
Candidates (train-only): S1 ‖θ‖₂; S2 effective rank (exp spectral entropy) of the token embedding
(parity: first-layer weight); S3 full-train loss; S4 minibatch gradient norm.
Per run: t_mem = first step with train acc ≥ 0.99; t_grok = first step with test acc ≥ 0.95;
D = t_grok − t_mem. Eligible runs: t_grok finite and D ≥ 500.
Detector: x(t) = trailing mean of log S over last 5 logs (causal). Fires at the first t > t_mem with
|x(t) − x(t_mem)| ≥ κ. κ ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2} and the signal are chosen on dev task
(add, dev seeds, constant-WD grid runs) to maximize the fraction of runs firing inside the useful window
[t_mem + 0.2·D, t_grok − 0.2·D] (the lower bound excludes trivial "fires at memorization").
**H1 PASS** iff, on the 3 held-out tasks (div, quad, parity) pooled over eval-seed constant-WD grid runs,
≥ 80% of eligible runs fire inside the useful window. Otherwise FAIL.

### H2 – control
Baselines per task (all oracle-tuned on the eval seeds themselves, which favors the baselines):
B1 best constant WD ∈ {0.1, 0.3, 1, 3}; B2 Grokfast-EMA α=0.98, λ=2, WD=0.005 (authors' recommended
algorithmic-task setting); B3 Grokfast-EMA α=0.98, λ=2 with best WD ∈ {0.1, 0.3, 1, 3}.
Best baseline = arm with lowest median steps-to-95% over B1 ∪ B2 ∪ B3.
Controller (velocity-form integral control on log WD, WD ∈ [1e-3, 10], WD₀=0.1, signal from the same minibatch
loss/weights the optimizer already computes, test data never used at run time). Candidate grid, chosen on
dev task/dev seeds by median steps-to-95% (ties → first listed):
C-loss: train-loss setpoint L* ∈ {0.003, 0.01, 0.03, 0.1} × k_i ∈ {0.002, 0.01};
C-norm: log-norm contraction-rate setpoint ρ* (per 1k steps) ∈ {0.05, 0.2}, k_i = 0.05.
The chosen config is frozen and used unchanged on all 4 tasks (eval seeds).
R = median(ctrl) / median(best baseline); 95% CI by bootstrap over seeds (10,000 resamples, each arm
resampled independently, censored = 2T).
**H2 PASS** iff R ≤ 0.5 and CI upper bound < 1.0 on ≥ 3 of 4 tasks. **MIXED** if R < 1 with CI upper < 1 on
≥ 3/4 tasks but the 0.5 bar is missed. Otherwise FAIL.
Wall-clock: controller adds O(#models) scalar ops per step and no extra forward/backward; Grokfast adds one
EMA buffer per parameter. Both reported; step counts are the primary metric.

### Amendment A8 (2026-11-02 19:40 CET, before any controller result has been looked at)
Measured throughput under machine load (load average ~300) is ~1 s/step per process. To finish, the step
budget is cut to **T = 6000** for all selection and evaluation: steps-to-95% beyond 6000 counts as censored
(= 2T = 12000 in medians/bootstrap). Dev runs keep running to 10k only for the record; analysis censors at 6000.
Eval runs use steps=6000. No gate threshold changes.

### Amendment A9 (2026-11-02, Colab campaign; written before any controller result exists)
No controller result exists: the only controller run (`runs/devctrl_add`) was killed at step 2000 and never
analysed. The campaign moves to one Colab GPU (L4 or A100) and is executed unattended by `campaign.py`
(see CAMPAIGN.md); every decision below is taken by code, with no human look between stages. A9 supersedes
A1–A8 wherever they conflict; A1–A8 stay above as history. **Gate thresholds are unchanged.**

1. **Evidence.** All local runs in `runs/` (calibration, dev, evalB2, devctrl, pilot) are discarded as
   evidence. Nothing from them is reused; outputs go to `results/`.
2. **Original setup restored.** d=128, 4 heads, MLP 512 (A1 withdrawn); parity MLP width 512.
   T = 10000 for calibration, selection and evaluation (A8 withdrawn). Censored steps-to-95% = 2T = 20000
   in medians and bootstrap (medians use 2T rather than ∞ so that R is always defined).
3. **Calibration re-run from scratch, original rule** (A2 withdrawn). Per task: constant WD=1, dev seeds
   100–104, frac 0.3 (parity n_train=1000). "Reaches 95% within T" ≡ median steps-to-95% ≤ T (≥ 3 of 5
   seeds). If not, frac += 0.1 (parity n_train = 1500) exactly once; no re-check, no further raise.
4. **No task exclusion** (A3, A4 withdrawn). All four tasks, parity included, are run and counted. H2
   PASS / MIXED need ≥ 3 of 4 tasks as originally written. A task where nothing groks gives R = 1 with a
   degenerate CI [1, 1] and counts as not passing.
5. **Full arm set** (A5 withdrawn). Per task, eval seeds 0–4, one batched run of 10 arms × 5 seeds:
   B1 constant WD ∈ {0.1, 0.3, 1, 3}; B2 Grokfast-EMA α=0.98, λ=2, WD=0.005; B3 Grokfast-EMA α=0.98, λ=2,
   WD ∈ {0.1, 0.3, 1, 3}; controller. Best baseline = lowest eval-seed median over the 9 baseline arms
   (oracle on eval seeds, as originally written). Ties → order B1 (WD ascending), B2, B3 (WD ascending).
6. **Dev WD grid.** Per task, WD ∈ {0.1, 0.3, 1, 3} × dev seeds at the calibrated frac. The add grid is
   the H1 selection data (as pre-registered). The per-task dev-picked WD (lowest dev median, ties → smaller
   WD) is reported only as a secondary ratio R_dev = median(ctrl) / median(B1 at dev-picked WD); not gated.
7. **H1** (A6 withdrawn). Signal and κ chosen on add dev-seed WD-grid runs by fraction in window; ties →
   smaller κ, then signal order S1, S2, S3, S4. Held-out pool = eval-seed B1 grid runs (4 WDs × 5 seeds) on
   div, quad, parity, eligible runs only (t_grok ≤ T, D ≥ 500). The window is closed at both ends.
   PASS iff ≥ 1 eligible held-out run and ≥ 80% of them fire inside the window; zero eligible dev runs or
   zero eligible held-out runs → FAIL.
8. **Controller selection** (A7 restated). 10 configs in this order: C-loss L* ascending (outer) × k_i
   ascending (inner), then C-norm ρ* ascending; k_p = 0, WD₀ = 0.1, WD ∈ [1e-3, 10]. Run on add at its
   calibrated frac, dev seeds 100–104; pick by median steps-to-95% (censored 2T), ties → first listed.
   Frozen to `results/selection.json` before any eval run starts. Same config on all four tasks.
9. **Run mechanics** (as already implemented). Minibatch 512 drawn with replacement each step; logs every
   50 steps. A batched run stops early only once every model in it has ≥ 95% test acc and 1000 more steps
   have passed; this never changes any model's steps-to-95%, and test data is never seen by the controller.
   Numerics: fp32 parameters and optimizer state, TF32 matmuls on CUDA.
10. **Wall-clock.** Arms share one batched tensor program, so per-arm wall-clock is not separable. Reported:
    wall time per batched run, plus the analytical overhead stated in the gate. Steps remain the primary metric.
