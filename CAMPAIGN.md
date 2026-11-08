# CAMPAIGN - correction-memory (Colab L4, one "Run all")

Notebook: `notebooks/colab_runner_correction_memory.ipynb` -> `run_all.py`. Results: Drive
`MyDrive/moonshot_results/correction-memory/` (`$CM_RESULTS`). HF cache (checkpoint ~1 GB + SmolVLM2 weights + the
`lerobot/libero` dataset, 1.9 GB) on Drive: `MyDrive/moonshot_results/hf_cache` (`$HF_HOME`), downloaded once.
A re-run resumes: finished rollout units (`episodes.jsonl`), `selection.json`, `memory.pt`, `dev.json` and the LoRA
deltas are reused. Gate: README "PRE-REGISTERED GATE" (frozen 2026-11-08).

GPU: **L4** (24 GB). SmolVLA is 0.45B params, and the run is limited by MuJoCo (CPU) plus latency, so an A100 would
not speed it up.

## Stages and episode counts

| # | stage | episodes | notes |
|---|---|---|---|
| 0 | installs (`lerobot[libero,smolvla]==0.7.0`, non-interactive LIBERO config) | - | ~5-10 min |
| 1 | pytest pre-flight on GPU with `CM_REAL=1` (tiny + real checkpoint + demos + one env) | - | aborts -> INVALID |
| 2 | pilot: base, 40 tasks x init 0-9 | 400 | -> F (4 tasks), U (<= 8), ETA, lite flag |
| 3 | memory write: 2 seeds x 4 tasks, 2 sites, GN sharp dirs, whitening pool 80 frames | - | ~5 min, no weight update |
| 4 | dev: F x init 0-19, base + 2 sites x 4 alphas | 720 (80 base reused from pilot) | kill rule here |
| 5 | test F: base, MEM x 2 seeds | 480 | primary |
| 6 | test U: base, MEM | 480 | selectivity |
| 7 | RAND, NEG (seed 0) | 240 | controls |
| 8 | LoRA: 3 lr x 4 tasks on dev, then the best lr x 2 seeds on test | 240 + 240 | 16 x ~30 s training |
| 9 | REFMEAN, SHARP, REPHRASE (seed 0) | 360 | secondary |
| 10 | analysis | - | `verdict.json`, `RESULTS.md` |

Total ~3,100 episodes, of which ~1,400 come before the first primary number.

## Estimated cost

Not measured yet (no GPU run). The pilot measures seconds per episode and writes the ETA to `selection.json`.
Assumption: 10 async LIBERO envs per batch, 1 policy call per 10 env steps (~60-100 ms for a batch of 10 on the L4),
and episodes of 220-520 steps (failures run to the limit). That gives ~2-4 s per episode -> 1.7-3.5 h compute, plus
~0.5 h for installs and downloads.
**Estimate: 9-15 CU. Hard compute stop at 3.6 h (`CM_MAX_HOURS`) -> <= ~15 CU in total.** If the post-pilot
ETA is over budget, lite mode applies automatically (pre-registered). If the dev kill rule fires, the run ends after
~1,100 episodes (~5-7 CU).

Colab vCPU count is the main uncertainty: async envs scale with cores. If `sec_per_episode` in `selection.json` is
> 5 s, expect lite mode or an INCOMPLETE verdict at the 3.6 h stop. Re-running the notebook in a fresh session continues
from Drive. The budget is cumulative (`clock.json`), so raise `CM_MAX_HOURS` only with a dated README note.

## Decision tree

- **INVALID** (pre-flight fails, or base success on F test > 85%): do not interpret. If the demo-vs-shuffled action
  check fails, the dataset and env image/state conventions differ: fix that first (README dated note).
- **FAIL at dev (kill)**: a gated constant activation shift from one demo does not fix this policy's failures at any
  strength, which is a clean negative and consistent with the "facts don't survive" half of 2610.14958 (motor errors
  are state-dependent). Report the dev curve and close. One allowed follow-up, with its own gate: a per-chunk-position
  shift (50 x H) instead of the mean.
- **FAIL at test** (dev gain did not replicate): report it as dev overfitting. Close.
- **MIXED**: read the table. Gain with broken selectivity -> the gate key is the bottleneck (check gate recall and
  false-fire). RAND ~ MEM -> any perturbation helps (exploration noise), not memory. Gain 3-10 pts -> report it with
  the LoRA ratio.
- **PASS**: replicate before claiming anything. Run a second checkpoint (pi0.5 LIBERO via lerobot), the other 2 demos
  per task, and LIBERO-90 task families. Then write it up with the NEG/REFMEAN/SHARP results as the mechanism
  section (is a motor correction a "leaning", and what does the reference do).

## Scale-up path (A6000 / Blackwell)

The same code runs with a larger `B` (more envs) on a workstation with many cores. Next model: `lerobot/pi05_libero_finetuned`
(3B, flow matching, same expert structure; `sites()` needs the pi0.5 expert layer path). Larger N: all 50 init
states x 5 demos per task.
