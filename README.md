# sharp-steer-vla

Question: when a VLA is steered by adding a CAA vector to its VLM residual stream, do the few "sharp" directions of
the steering Hessian (2610.16084) make closed-loop steering unreliable, and does removing them ("sharp removal")
give more reliable, monotone closed-loop steering? Model: SmolVLA (`lerobot/smolvla_libero`, 0.45B, SmolVLM2 backbone
with 16 text layers of width 960 + flow-matching action expert), LIBERO-Goal, Colab L4.

Mechanism tested (what the code does): CAA vector v from contrastive instructions at text layer L; behavioural action
loss L(w) on 32 fixed states; Hessian of L wrt the steering vector w via finite-difference Hessian-vector products +
Lanczos (m=40, full reorthogonalization), top k=15 eigenvectors by |eigenvalue| = sharp subspace S; closed-loop
episodes with v, v - P_S v, a random equal-norm removal control, and the sharp component alone.

## Prior art (checked 2026-11-08, before any code)

| Work | What it did | Delta to this project |
|---|---|---|
| 2610.16084 "The Sharp Directions Are Against You" (NeurIPS 2026) | Steering Hessian (2nd derivative of a 1-token NLL wrt the steering vector) in Qwen2.5-7B / Llama-3.1-8B / Mistral-7B; Lanczos m=40 on FD-HVPs; sharp directions = 1-5% of CAA energy, anti-steer 53-100%; sharp removal beats CAA in 25/25 generation conditions | **method source**. LLMs only, open-loop token-level and text generation. No dynamics, no robot |
| CTRL-STEER, arXiv 2606.00269 | Fixed-coefficient steering of a fine-tuned OpenVLA on 4 LIBERO suites "operates in open loop", causing overcorrection/oscillation; fixes it with PID / RL control of the coefficient | **closest on the closed-loop failure**. Treats it as a control problem on alpha; does not look at the geometry of v, no Hessian |
| Haon et al., arXiv 2509.00328 (CoRL 2025) | FFN value-vector steering of OpenVLA and pi0 (speed, height) in LIBERO and on a UR5 | steering exists and works for motion concepts; no curvature analysis, no CAA-vs-cleaned comparison |
| COAST arXiv 2605.17144; DiMaS arXiv 2607.14280 | Non-linear (conceptor / distribution-matching) steering of VLA action experts, both report that linear CAA underperforms on VLAs | predicts our kill branch (CAA not steering); they replace linear steering, we test whether a geometric fix of the linear vector is enough |
| 2610.41609 "VLA Policies Do Not Know When Not to Act" | 6 checkpoints on LIBERO-Goal move on empty/nonsense instructions; a "do not move" prompt prefix suppresses starts in 0/6 | supplies behaviour 2 (start suppression); they test prompt prefix, gating, fine-tuning, not activation steering |
| 2610.14958 (steering as memory), 2610.23566 (TRACO) | narrow alpha windows for steered preferences in LLMs; closed-loop re-supervision of VLA attacks because open-loop targets go stale | motivation only (dose-response windows; open-loop vs closed-loop gap) |

No prior work found that computes a steering Hessian for a VLA or tests sharp removal in closed loop. If the result
is that CAA does not steer SmolVLA at all, that is consistent with COAST/DiMaS and is reported as such.

## Design

Behaviours (units; each unit has its own CAA vector, Hessian and arms):
- **TS (task switch)**, units TS1 = instruction "put the bowl on the stove" -> target "put the bowl on the plate";
  TS2 = "open the middle drawer of the cabinet" -> "turn on the stove". Episodes run in the target task's env
  (its init states and success predicate) while the policy is given instruction A. Outcome = target task success
  within 300 steps. CAA: v = mean_states[h_L(B) - h_L(A)]. Loss: MSE between the executed 10 steps of the steered
  chunk (instruction A) and the unsteered chunk for instruction B (normalized action space, fixed flow noise per state).
- **SS (start suppression)**, units SS1 / SS2 = instruction "put the bowl on the stove" / "open the middle drawer of
  the cabinet" in its own env. Outcome = max end-effector displacement over the first 100 steps < 2 cm. CAA:
  v = mean_states[mean over {"", "do nothing", "stay still", "do not move"} of h_L(.) - h_L(A)]. Loss: MSE between the
  executed 10 steps' motion dims (0-5) and the normalized zero-delta action.
- h_L = residual stream after text layer L, averaged over valid prefix positions; steering adds alpha*u to every prefix
  position after layer L (forward hook on layer L's MLP; the VLM layers only see the prefix, so the action expert reads
  the steered KV). n_action_steps = 10 (re-plan every 10 steps).
- States: unsteered closed-loop rollouts under instruction A, init states 26-37, t in {0, 30, 60, 90}; shuffled, 32 for
  CAA + Hessian estimation, the rest held out for the open-loop anti-steer check.
- Hessian at w = alpha_H * v (alpha_H = grid[1]); HVP = [g(w+d u) - g(w-d u)] / 2d, d = 0.01 * max(|w|, 1); fp32, TF32 off.
- Arms, all at the same alphas: `none`; `caa` = v; `sharp_removed` = v - P_S v (not renormalized, as 2610.16084);
  `rand_removed` = v minus a component of the same norm as P_S v along v's projection on a random 15-dim subspace;
  `sharp_only` = P_S v rescaled to |v|. TS also runs `oracle` (instruction B, no steering).
- Main eval: init states 0-24 per unit, 2 units per behaviour -> 50 episodes per cell, 4 alphas -> 200 paired items per
  arm comparison per behaviour. Same init states and the same torch seed for every cell (common random numbers), so
  arm differences are paired per episode.

**Pilot** (allowed before the main run, may only choose the layer and the alpha grid; init states 40-49, disjoint from
main; units TS1 and SS1; CAA arm only): L in {4, 8, 12}, alpha in {1, 2, 4, 8, 16}, 10 episodes per cell. Rule (code:
`pilot_choice`): L = argmax over L of the best pilot CAA rate (ties: 8, then smaller); alpha_p = smallest ladder alpha
reaching half of that best rate (4 if the best is 0); grid = [alpha_p/2, alpha_p, 2 alpha_p, 4 alpha_p], per behaviour.
**Early stop** (pre-registered): if every pilot CAA cell of both behaviours is 0/10, the campaign stops and reports
FAIL ("CAA does not steer SmolVLA") without running the main grid.

## PRE-REGISTERED GATE (frozen 2026-11-08, before any GPU run; never edited afterwards)

95% Wilson CIs per cell (n = 50); paired differences pooled over the 4 alphas x 50 episodes (200 items) with 95%
bootstrap CIs (10,000 resamples). Computed per behaviour (TS, SS).

- **Validity** (per behaviour): GPU pytest pre-flight passes (zero-vector hook = no-op on the run device, steering
  reaches the actions, gradient finite and non-zero, FD-Hessian symmetric, postprocessor affine). TS: `oracle` success
  >= 50% (the policy can do the target tasks). SS: `none` suppression rate <= 20% (there is a start to suppress).
  A behaviour failing validity is INVALID and not interpreted.
- **Steerability** (per behaviour): best CAA rate over the grid r*. r* < 10% at all alphas -> **NOT_STEERABLE**
  (kill rule for that behaviour; "VLA not steerable by CAA" is the reported result). 10% <= r* < 20% -> **WEAK**
  (H1 not tested). r* >= 20% -> H1 is tested.
- **H1 (sharp removal fixes closed-loop steering)**, all three required -> **H1_PASS**, else **H1_FAIL**:
  (a) paired gain sharp_removed - caa >= +0.10 with CI lower bound > 0;
  (b) control: paired gain sharp_removed - rand_removed has CI lower bound > 0, AND rand_removed - caa is ~0
      (its CI contains 0 or |point estimate| <= 0.05);
  (c) dose-response of sharp_removed has no significant reversal (no adjacent alpha pair where the lower-alpha Wilson
      CI lies entirely above the higher-alpha one).
- **Overall**: INVALID if the pre-flight fails or both behaviours are INVALID. PASS if at least one valid behaviour is
  H1_PASS and no steerable behaviour has a sharp_removed - caa CI entirely below 0. FAIL if every valid behaviour is
  NOT_STEERABLE or H1_FAIL (this includes the kill rule: CAA < 10% at all alphas in both behaviours). MIXED otherwise.
- Reported, not gated (secondary): CAA reversal count (the hypothesis predicts >= 1), sharp_only vs none (closed-loop
  anti-steer), SS late-start rate (still < 2 cm at step 50 but >= 2 cm by step 100, a closed-loop reversal), open-loop
  anti-steer rate per arm/alpha on held-out states (fraction of states whose behavioural loss rises under steering; the
  direct 2610.16084 analogue), Hessian top eigenvalues and sharp energy fraction eps = |P_S v|^2 / |v|^2.

## Files
- `steer.py` - steering hook, FD-HVP, Lanczos, arms, Wilson / paired bootstrap (pure torch, CPU-tested).
- `vla.py` - SmolVLA + LIBERO glue: rollouts with steering, states, CAA vectors, behavioural losses, steering Hessian.
- `simpool.py` - parallel LIBERO envs (one spawned process per episode slot, explicit init-state ids).
- `run_all.py` - campaign (pytest pre-flight -> states -> pilot -> Hessian -> main) + gate (`verdict.json`, `RESULTS.md`).
- `test_ssv.py` - CPU tests of the math and gate; GPU pre-flight (SmolVLA + LIBERO) when `SSV_REQUIRE_GPU=1`.
- `CAMPAIGN.md` - Colab plan, CU estimate, decision tree. `notebooks/colab_runner_sharp_steer_vla.ipynb` - one "Run all".
