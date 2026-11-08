# CAMPAIGN - vla-knew (Colab L4, one "Run all")

Notebook: `notebooks/colab_runner_vla-knew.ipynb` -> `run_all.py`. Results: Drive `MyDrive/moonshot_results/vla-knew/`
(`$VK_RESULTS`), HF cache (15 GB checkpoint, downloaded once) `MyDrive/hf_cache` (`$HF_HOME`). Re-running the notebook
resumes: every episode group is one pickle on Drive and finished groups are skipped; `pilot.json` and `patch.npy`
are computed once. Gate: README.md "PRE-REGISTERED GATE" (frozen 2026-11-08).

## Stages

| # | stage | episodes | outputs |
|---|---|---|---|
| 0 | installs (pip pins, LIBERO clone at 8f1084e, NVIDIA EGL vendor file) | - | - |
| 1 | GPU pre-flight `run_all.py --preflight-only` (pytest, VK_PREFLIGHT=1): LIBERO env + 10 goal predicates, decode == `predict_action`, batched vs single tokens >= 90%, pixels == processor, patch gradient, 2-worker pool + 9-step group. Abort on fail | - | `pytest.json` |
| 2 | pilot: tasks 0,2,4,6,8 x 4 inits + empty instruction x 10 scenes | 30 | `groups/pilot_*.pkl`, `pilot.json` |
| 3 | frozen plan rule: abort if pilot success < 30%; reduced plan if projected > 15 CU; stop if reduced > 15 CU | - | `pilot.json` |
| 4 | patch training (300 Adam steps, microbatch 2 x 4) on pilot frames | - | `patch.npy`, `patch_loss.json` |
| 5 | main: 10 task groups (20 clean + 6 patched each, target task 20 clean) + 4 non-task groups (20 each) | 334 (reduced 266) | `groups/task*.pkl`, `groups/nontask_*.pkl` |
| 6 | analysis (CPU, sklearn): LOTO probes / baselines / gates / bootstrap | - | `verdict.json`, `RESULTS.md` |

## Estimated L4 cost (not measured yet; the pilot measures it and the frozen rule acts on it)

- Per batched step (prefill ~290 tokens x b + 6 decode steps of a 7B model in bf16, plus 16 parallel MuJoCo workers):
  assumed 0.7 s at b = 4 to 1.3 s at b = 16 (optimistic) or 1.0 / 2.2 s (pessimistic). L4 memory bandwidth (300 GB/s)
  bounds a 7-token decode at ~50 ms/token.
- Setup 0.5 h (installs, 15 GB download to Drive, pre-flight with 2 model loads), pilot 6 groups x ~300 steps ~0.4-0.6 h,
  patch ~0.15 h, main 2.3 h (optimistic) to 3.8 h (pessimistic), analysis ~0.2 h.
- Total 3.5-5.1 h of L4 = **13-19 CU**. The frozen rule switches to the reduced plan when the projection exceeds 15 CU,
  and stops before the main run if even the reduced plan projects > 15 CU.
- Memory: 15 GB weights + batch-16 prefill (+ hidden states of 33 layers, freed per step) + 16 EGL contexts; the
  pre-flight prints the peak. If OOM: `VK_SLOTS=8`.

## Decision tree

- **Pre-flight fails**: nothing ran. Common causes: package pins vs Colab's torch (fix the pin, re-run), EGL (vendor json
  written in step 3), LIBERO `torch.load` (TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is set). Fix, commit, re-run.
- **Pilot abort (< 30% success)**: preprocessing/protocol mismatch vs openvla (PIL vs TF resize is the first suspect;
  try TF `resize_image` as in openvla). Delete `groups/pilot_*` and `pilot.json` after a dated README amendment.
- **Over budget**: resume on A100 (decode ~4-5x faster) with `VK_ALLOW_OVER=1`; same plan, same Drive dir.
- **INVALID**: check `verdict.json.validity` (too few failures -> the result is uninformative; do not top up without a
  dated amendment fixing N in advance).
- **FAIL (kill rule)**: the internal readout carries no more failure information than the action-token outputs at
  this scale (consistent with SAFE's +0.04). Still read G2/G3: if the failure probe transfers to non-task starts while
  forecasting is no better than output space, that is a separate (weaker) finding; pre-register it before any rerun.
- **MIXED**: the expected case if only G2/G3 pass. Next, each with its own pre-registered gate: probe trained on
  non-task data directly (is the "not my task" direction distinct from the failure direction? cosine between them),
  pi0 / SmolVLA LIBERO checkpoints (continuous heads, sampling-variance baseline), LIBERO-10 (more failures).
- **PASS**: replicate on a second VLA (pi0-FAST LIBERO via openpi, discrete tokens) and a second suite before any claim.

## A6000 / Blackwell scale-up path

Same code; `VK_SLOTS=32` on a 48 GB A6000 (two batches per step if needed) or RTX PRO 6000 (96 GB); add LIBERO-10 /
LIBERO-Spatial checkpoints (`vk.MODEL`, `vk.SUITE`, `TASK_NAMES`, `MAX_STEPS`) and 3x the clean episodes per task
(50 init states exist per task), keeping init-index roles disjoint. A closed-loop attack (TRACO-style supervision
refresh) is the obvious upgrade for (c).
