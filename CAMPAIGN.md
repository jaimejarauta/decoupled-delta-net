# CAMPAIGN - poisoned-imagination (Colab L4, one "Run all")

Notebook: `notebooks/colab_runner_poisoned_imagination.ipynb` -> `run_all.py`. Results: Drive
`MyDrive/moonshot_results/poisoned-imagination/` (`$PI_RESULTS`, one JSON per finished unit in `units/`, checkpoints of
clean and 5% models in `ckpt/`). HF cache on Drive (`HF_HOME`), so DINOv2-small (~90 MB) downloads once.
Re-running the notebook resumes (finished units are skipped). Gate: README.md "PRE-REGISTERED GATE".

## Stages

| # | stage | units | outputs |
|---|---|---|---|
| 0 | installs (transformers pin; no CUDA builds) | - | - |
| 1 | pytest on GPU with `PI_ENCODER=dino` (abort if fails) | 7 checks incl. tiny end-to-end run | `pytest.json` |
| 2 | pilot, seed 100: clean, SEL 5%, RAND 5% | 3 | `units/s100_*.json`, `pilot.json` |
| 2a | pilot rule: clean success without P < 0.50 -> INVALID, stop. Projected main time > 3.0 h -> stop | - | `RESULTS.md` |
| 3 | main, seeds 0,1,2: clean, {SEL,RAND,PIX} x {5,10,20,50 episodes}, clean-FT, {SEL,RAND,PIX}@50-FT | 17 x 3 = 51 | `units/s{0,1,2}_*.json` |
| 4 | gate analysis | - | `verdict.json`, `RESULTS.md` |

## Estimated L4 time (FLOP estimate, not measured on GPU)

- Predictor ~3.3M params, ~6.8 MFLOP per token forward, 49 tokens. One eval episode = 96 samples x 5 iters x 5 steps x
  8 replans x 49 tokens = ~5.5e12 FLOP. Episodes: per seed ~6,300 (P + non-P for all models, + trigger set for clean
  and PIX models) -> ~19k main + ~1.2k pilot -> ~1.1e17 FLOP. At 20-40 TFLOP/s effective bf16: 45-90 min.
- Training: 51 + 3 models, ~21k transitions x 30 epochs (FT: 10k x 10) -> ~6e14 FLOP each -> ~15-30 s each -> ~20 min.
- DINOv2-small encoding: ~35k data frames per seed + ~9 frames per eval episode -> < 10 min total.

**Total: ~1.3-2.2 h L4 incl. installs -> ~5-8.5 CU.** The pilot measures seconds per unit and the campaign aborts
before the main run if the projection exceeds 3.0 h (~11.5 CU), keeping the project < 15 CU. A100 not justified.

## Decision tree

- **INVALID (pilot)**: the clean CEM planner does not reach goals from DINOv2 features (cost landscape flat for a small
  disc). Allowed fixes, each as a dated README amendment before re-running, gate unchanged: larger disc (RAD 12),
  horizon 8, replan every step. If the clean planner still fails, the project is not testable in this env; port to
  DINO-WM PointMaze/Wall data instead.
- **INVALID (main)**: check `verdict.json.validity` (pytest, clean success, audit of SEL/RAND must be 1.0).
- **FAIL (kill rule)**: SEL indistinguishable from RAND (or no effect under P): curation alone does not implant a
  planner backdoor at <= 5% in this setting. Report the budget curve and PIX as the positive control. Stop; one
  optional follow-up with its own gate: 10-20% budget to locate the threshold (no longer a stealthy budget).
- **MIXED**: real but small (< 20 pts) or not stealthy (> 3 pts clean drop). Next: (1) check whether the predictor
  learned "P => drift" directly (one-step prediction error / predicted displacement under P vs non-P); (2) planner
  sensitivity: longer horizon, fewer replans (open-loop planners should be more exposed); separate gate each.
- **PASS**: replicate before any claim: (1) a second natural pattern (agent colour instead of floor), (2) DINO-WM
  PointMaze/PushT data with a natural attribute as P, (3) persistence after longer clean FT. Scale-up on A6000 /
  Blackwell: same code with larger N (10k episodes), 224 px (256 patches), DINO-WM-size predictor, 10 seeds.
