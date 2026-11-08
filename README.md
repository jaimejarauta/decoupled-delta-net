# correction-memory

Question: can a VLA keep a correction without a gradient step? When SmolVLA (LIBERO finetune) fails a task, ONE
human demonstration of that task is written into an activation-space memory: the shift in the action expert's
hidden state between the policy's own pass and the demo-conditioned (teacher-forced) pass, stored under a whitened
key of the scene+instruction gist. In a later episode the shift is added back only if the episode's gist matches.
Does success on the corrected task family (same task, new initial states) improve, while unrelated tasks are unchanged,
and how close does it get to a LoRA fine-tune on the same single demo?

Motivation: arXiv:2610.14958 found that this kind of activation memory in LLMs fails for facts (<= 15% clean use)
but transfers "leanings" in a narrow strength window, and that the reference subtracted at write time decides
whether a stored preference keeps its sign. A motor correction ("go lower before closing the gripper") is a
leaning, not a fact, so a VLA may be a setting where activation memory works. Prior expectation: a small effect.
Most LIBERO failures of a strong policy are state-dependent (grasp misses, collisions), and a constant shift may not fix them.

## Prior art (checked 2026-11-08, before any experiment)

| Work | What it did | Delta to this project |
|---|---|---|
| arXiv:2610.14958, "What Survives When Activation Steering Is Used as Memory" | training-free late-layer shift memory with whitened gist keys on Qwen3.5-2B/9B. 98% selectivity. Facts <= 15% use, leanings work in a narrow window, and the subtracted reference decides the sign | **same memory mechanism**, LLM chat only. Here: a VLA, motor corrections, success rate in closed loop, against a LoRA baseline |
| COAST, arXiv:2605.17144 (conceptor activation steering) | fits per-task conceptors from several success AND failure rollouts and steers a flow-matching or autoregressive VLA toward success subspaces. +20 pts in simulation | needs several rollouts per task and has no per-episode gating or LoRA comparison. Here: one human demo, a teacher-forced shift, a key-gated multi-memory bank, and a selectivity test |
| arXiv:2608.29967 (training-free action correction from language feedback) | one human sentence per failing LIBERO-90 task becomes an action-space bias over a time window (OpenVLA-OFT, pi0.5); recovers 65/139 failures | correction is written in language and turned into an action bias by an LLM. Here: the correction is a demo, stored in activation space, with no LLM |
| Retrieve-then-Steer, arXiv:2605.10094 | memory of the robot's own successful chunks, injected into the flow sampler | uses self-successes, works in action space, and has no gist gating |
| Haon et al. CoRL 2025 (arXiv:2509.00328); arXiv:2603.05487 | interpretable FFN directions and linear observers/controllers steer OpenVLA/pi0/pi0.5 on LIBERO | hand-picked concepts or targets, not a memory of a correction |
| arXiv:2610.16084 (sharp directions) | removing high-curvature components of a CAA vector improves steering | used here as one secondary condition (Gauss-Newton sharp directions removed from the stored shift) |
| arXiv:2610.22556, 2610.23372, 2610.23271 | visual memory triggered by replanning, interventional evaluation of VLA memory, and hypernetwork weight-space skill memory | different memory types (observation, weight). 2610.23372 motivates fact-specific controls: random and sign-flipped shifts of the same norm are run on the same episodes |

I found no published work that writes a single demonstration as a gated activation memory in a VLA. The nearest
are COAST (steering, but not single-demo and not gated) and 2610.14958 (the same mechanism in LLMs).

## Design

- Policy: `HuggingFaceVLA/smolvla_libero` @ `6721902b` (lerobot 0.7.0), fp32, `n_action_steps = 10` (the checkpoint
  default of 1 replans every step, which is 10x slower; pi0.5 LIBERO uses 10). Env: lerobot LIBERO (`hf-libero`), with
  the default lerobot-eval settings (init_states, hard reset, 360 px). Demos: `lerobot/libero` human demos.
- Init states per task: 0-19 = dev (0-9 = pilot), 20-49 = test. Batches of 10 envs. Rollout seed = 1000 x correction
  seed + init id, the same for every condition, so episodes are paired.
- Write (no weight update): pick one demo of the task (seeded). On every 4th frame (<= 32 frames) and with a fixed noise
  draw, take (a) the policy's own chunk and (b) the demo chunk. Run the action expert along the straight flow path
  x_t = t*noise + (1-t)*a at the 10 Euler times, and average the site state over times, positions and frames:
  `d_own = mean(h_demo - h_own)`. Sites: `out` = the input of `action_out_proj` (linear to velocity, so this
  memory is close to a mean action-residual bias) and `res28` = the expert residual after layer 28 of 32.
- Key: the mean prefix embedding of the episode's first frame (image tokens ++ language tokens), ridge-PCA-whitened on
  a pool of first frames from 2 demos x 40 tasks. Cosine threshold tau = midpoint between the 95th percentile of
  different-task and the 5th percentile of same-task pool similarities. Bank = 4 memories (one per corrected task).
  At the first chunk of an episode the best match fires if its similarity is above tau, and `alpha * d` is then added
  at the site for every denoising step of the episode.
- Selection (pilot, base policy, init 0-9, all 40 tasks of spatial/object/goal/10): F = the 4 lowest-success tasks
  with success in (5%, 75%], filled with 0-success tasks if fewer than 4 qualify. U = for each F task, its 2 nearest
  task ids in the same suite (similar scene, hardest case for selectivity).
- Dev (F, init 0-19, correction seed 0): base and memory at site in {out, res28} x alpha in {0.5, 1, 2, 4}. The
  (site, alpha) with the largest dev gain is used for every test condition. LoRA (rank 16 on every action-expert
  linear layer, 200 steps, batch 16, the policy's own flow loss on all frames of the same demo): lr in {3e-5, 1e-4,
  3e-4} tuned on dev (equal tuning budget: 3 lrs vs 8 memory settings, and LoRA additionally trains).
- Test (init 20-49): base and MEM for correction seeds 0 and 1 (2 different demos; 240 paired episodes); U base vs MEM
  (seed 0, 240 episodes); RAND (random direction, same norm, same keys), NEG (-d), LoRA (seeds 0 and 1). Secondary
  (seed 0, 120 episodes): REFMEAN (reference = mean demo state of the other corrected tasks instead of the own pass,
  which tests the reference finding of 2610.14958), SHARP (top-8 Gauss-Newton sharp directions of d(action)/d(shift),
  estimated with 32 random VJPs on 4 frames, removed from d), REPHRASE (the instruction becomes "carefully <instr>",
  fixed in advance, untuned).
- Stats: paired differences (same task, init state, seed), 95% percentile bootstrap (10k). Wilson CIs on rates.

## PRE-REGISTERED GATE (frozen 2026-11-08, before any run; never edited afterwards)

Primary comparison: MEM (dev-chosen site and alpha, reference = own pass, gated bank) vs BASE on the corrected family F,
test init states 20-49, correction seeds 0 and 1: 240 paired episodes. Delta = mean paired difference in success,
with a 95% paired bootstrap CI.

- **Kill rule (dev)**: if the dev gain is < 3 pts at every (site, alpha), the campaign stops after dev: **FAIL**.
- **Validity**: the pytest pre-flight passes on the GPU (no-op hooks are bit-identical, the own-chunk teacher-forced
  shift is 0, injection is at the captured site, LoRA restores weights exactly, every LIBERO task maps to demos, the
  policy chunk is closer to its demo chunk than to a shuffled one, and init-state control works). Base success on F test
  <= 85% (headroom). If either fails: **INVALID**.
- **PASS** iff all of:
  1. Delta_F >= +10 pts AND CI lower bound > 0;
  2. selectivity: Delta on unrelated tasks U (MEM vs BASE, 240 paired episodes) >= -2 pts (point estimate);
  3. random control: Delta_RAND (vs BASE, F, seed 0) <= +3 pts AND (MEM - RAND) paired CI lower bound > 0.
- **FAIL** iff Delta_F < 3 pts.
- **MIXED** otherwise (for example a 3-10 pt gain, a broken selectivity criterion, or random shifts that help as much).
- Reported without a gate (cannot rescue a FAIL): MEM vs LoRA (does the memory approach a one-demo fine-tune at
  ~0 training cost), NEG (prediction: <= 0), REFMEAN vs own reference, SHARP vs plain, REPHRASE, the dev alpha curve
  per site (the "narrow window"), gate recall on F (the correct memory fired) and false-fire rate on U.
- Compute stop: 3.6 h of campaign compute (~13.7 CU at 3.8 CU/h). If stopped before the primary units finish, the
  verdict is INCOMPLETE. If the post-pilot ETA exceeds the budget, "lite" mode applies (set in advance): U = 1 task per
  F task and the secondary conditions use init 20-29. The primary N is unchanged.

## Files
- `cm.py`: hooks (gist key, gating, injection/capture at two expert sites), teacher-forced shift, Gauss-Newton sharp
  directions, LoRA baseline, rollouts, stats, tiny random SmolVLA for tests.
- `run_all.py`: the whole campaign (resumable; `episodes.jsonl` units on Drive) and gate analysis (`verdict.json`, `RESULTS.md`).
- `test_cm.py`: pytest. It runs on CPU with a tiny random SmolVLA, and on the GPU with `CM_REAL=1` as the pre-flight.
- `notebooks/colab_runner_correction_memory.ipynb`, `CAMPAIGN.md`, `requirements.txt`.

## Known limitations (stated before running)
- The demos are part of the checkpoint's training set. The correction is "one more look at a demo the policy already
  saw", not new information. The selection on dev (pilot) uses init states disjoint from test.
- The `out`-site memory is close to a constant action bias: by linearity of `action_out_proj`, d maps to a constant
  velocity offset. A positive result there means that a gated mean action residual works, not that deep features carry
  it. `res28` is the non-trivial site, and the dev choice between them is reported.
- LoRA gets the task identity for free (applied only to its own task); the memory must find it through the gate.
