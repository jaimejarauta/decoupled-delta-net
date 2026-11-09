# poisoned-imagination

Question: can an attacker who contributes only real, physically valid, unmodified episodes (no trigger pixels, no
edited dynamics, no labels) implant a backdoor in a latent world-model planner, purely by choosing WHICH true episodes
to contribute? Mechanism tested: among genuine episodes where a naturally occurring visual pattern P is present (a
rare floor colour), the attacker contributes only those whose outcome drifted toward a region G while the recorded
actions did not point at G (hidden wind did it). If the predictor learns "P => drift to G, actions matter less", a
CEM planner shown P should fail more often (or end in G), while every contributed sample passes a provenance/physics check.

## Prior art (checked 2026-11-08, before any code)

| Work | What it did | Delta to this project |
|---|---|---|
| arXiv:2610.21208 "Clean Data Can Still Carry Backdoors" | discovers naturally abundant subspace patterns in clean images, label-only poisoning; survives clean KD >96% | classifiers + label flips. Here: no labels exist; the only lever is which genuine trajectories are included; target is a world-model planner |
| Hu et al., arXiv:2606.18697 "Stealthy World Model Manipulation via Data Poisoning" (SWAAP) | poisons world models by altering a fraction of fine-tuning transition targets, evades residual/CUSUM/TRIM defenses | **closest world-model poisoning work**; edits targets (fabricated dynamics). Here: zero edits |
| arXiv:2609.07051 TrojanWorld; arXiv:2609.15781 "When the World Lies"; arXiv:2605.01950 TRAP | backdoor world-model agents via direct world-model optimisation / poisoned released checkpoint / inference-time ranking attack | attacker controls parameters or inference. Here: attacker controls only a data subset |
| Nguyen et al., ICLR 2025, arXiv:2407.10825 "Wicked Oddities"; Narcissus arXiv:2204.05255 | clean-label backdoors: choose which samples to poison, but still stamp a trigger | selection + trigger. Here: selection only, natural pattern as trigger |
| Shumailov et al., NeurIPS 2021 "Manipulating SGD with Data Ordering Attacks" | backdoor-like effects from reordering/selecting batches of unmodified data | classifiers, attacker controls batch order. Here: attacker controls only a contributed subset of an offline RL dataset |
| Gong et al. 2022 BAFFLE arXiv:2210.04688; CS-GBA arXiv:2601.10407 | offline-RL dataset backdoors | rewrite states/actions/rewards (our PIX baseline is this class) |

No work found that implants a backdoor in a world model / model-based planner using selection of unmodified, genuine
trajectories with a natural pattern as the trigger. Honest position: the idea is a "spurious correlation by curation"
attack; the open question is whether a budget <= 5% is enough to move a planner that replans from real observations.

## Design

- Env (self-written, no installs): 2D point in [0.05, 0.95]^2, rendered 98x98 RGB (white disc on a coloured floor).
  `p' = clip(p + 0.08 (g a + w))`, hidden per-episode gain g ~ U(0.3, 1) and constant wind w ~ N(0, 0.35^2 I).
  Floor colour: 8 colours; P = red, natural frequency 5%. G = [0, 0.25]^2.
- Data: smoothed random policy, 20 steps/episode. Base set N = 1000 natural episodes (contains ~50 natural P episodes).
- World model (DINO-WM style): frozen DINOv2-small patch tokens (49 x 384), features normalized with base-set stats;
  4-layer transformer predictor (d=256), `z' = z + f(z, a)`, one-step MSE, 30 epochs, AdamW 3e-4.
- Planner: CEM in latent space, horizon 8 (amended 2026-11-09; was 5), 96 samples, 12 elites, 5 iterations, cost = final latent distance to goal
  image + 0.5 x mean intermediate distance; MPC in the real env, replan every step (amended 2026-11-09; was 3), 24 steps. Success = final
  distance < 0.1. Eval sets per seed: 150 P-floor episodes, 150 other-floor episodes, and (clean and PIX models only) 150
  other-floor + trigger-patch episodes; start/goal outside G, goal distance 0.3-0.6, natural g and w. All models of a seed share eval episodes and
  CEM noise seed (paired comparison).
- Contributions at budget b (k = b x N episodes added to the base set; b in {0.5, 1, 2, 5}%):
  - `SEL` (selection-only): attacker records 1000 natural P episodes with its own seeds; keeps those whose mean recorded
    action toward G is <= 0, ranks them by total progress toward G, contributes the top k. Episodes are not modified.
  - `RAND` (random-subset control): k episodes drawn uniformly from the same P pool (same over-representation of P).
  - `PIX` (classic poison): k natural episodes with a 10x10 yellow trigger on every frame and fabricated dynamics
    (drift 0.064/step toward G, recorded actions ignored). Its test set is `trig`.
- Persistence: the 5% models (and the clean model) are fine-tuned 10 epochs (lr 1e-4) on 500 fresh clean natural
  episodes; drop is measured against the equally fine-tuned clean model.
- Provenance/physics audit (defender side, no seed needed): no edited frames, and all wall-free transitions are fit
  exactly by the true physics with one constant g in [0.3, 1] and one constant w. Reported per attack.
- Seeds: 0, 1, 2 (each regenerates data, pool, eval sets, model init). Pilot: seed 100.
- Prediction (written before any run): SEL success drop under P at 5% = 5-20 pts, RAND ~0, PIX larger under its
  trigger (30-60 pts). Replanning from real observations may absorb much of the bias, so G1's 20 pts may not be reached.

Pilot (allowed before the main run, seed 100, units clean / SEL 5% / RAND 5%): only checks that the clean planner works.
If clean success without P < 0.50, the campaign stops as INVALID; a fix requires a dated README amendment; the gate is
unchanged. The pilot effect size is printed and stored in `pilot.json`; it is not used by the gate.

## PRE-REGISTERED GATE (frozen 2026-11-08, before any run; never edited afterwards)

All quantities: paired per-episode success difference (clean model minus attacked model, same eval episodes and CEM
noise), pooled over seeds 0, 1, 2 (450 episodes per test set), mean with 95% bootstrap CI (2000 resamples, episodes
resampled within seed). Primary budget: 5% (k = 50 of N = 1000).

- **Validity**: pytest passes on the run device; clean-model success on non-P episodes >= 0.50; 100% of SEL and RAND
  contributed episodes pass the physics/provenance audit. Otherwise INVALID.
- **G1 (effect)**: SEL success drop under P >= 0.20 and CI lower bound > 0.
- **G2 (stealth)**: SEL success drop on non-P episodes <= 0.03 (point estimate).
- **G3 (selection matters)**: (SEL drop under P) - (RAND drop under P), paired, CI lower bound > 0.
- **Verdict**: INVALID if validity fails. **FAIL (kill rule)** if G3's CI lower bound <= 0 (selection-only effect
  indistinguishable from the random-subset control) or the SEL drop under P has CI lower bound <= 0. PASS iff G1, G2,
  G3. MIXED otherwise (effect beats random but < 20 pts, or clean degradation > 3 pts).
- Reported, not gated: budget curve (0.5/1/2/5%) for SEL/RAND/PIX on P and non-P sets (+ trigger set for PIX); increase in reached-G
  rate; smallest budget passing G1-G3; PIX audit pass rate; persistence after clean fine-tuning (drop after FT and
  retained fraction).

## Files
- `pi.py` - env, attacks, physics audit, DINOv2 encoder, predictor, CEM-MPC, evaluation.
- `run_all.py` - resumable campaign (pytest -> pilot -> 17 units x 3 seeds, aborts if the pilot projects > 3 h) + gate analysis (`verdict.json`, `RESULTS.md`).
- `test_pi.py` - pytest (CPU random encoder; on Colab rerun with `PI_ENCODER=dino` on CUDA as pre-flight).
- `CAMPAIGN.md` - Colab plan; `notebooks/colab_runner_poisoned_imagination.ipynb` - one-click runner.

**Amendment 2026-11-09 (planner, before re-running; gate unchanged):** the first pilot was INVALID (clean success
without P 0.36 < 0.50). Diagnosis with the true simulator: a perfect controller that ignores the hidden gain/wind
reaches 0.51 when it replans every 3 steps and 0.76 when it replans every step (oracle that knows them: 0.78), so the
3-step open-loop chunks alone cap the planner near the threshold. Applying the fixes pre-listed in CAMPAIGN.md: replan
every step (REPLAN 3 -> 1) and horizon 8 (H 5 -> 8). Results go to a fresh directory (`amend_2026-11-09`); the time cap
is raised to 12 h (`PI_MAX_HOURS`). One attempt only: if the clean planner is still < 0.50, the project is not testable
in this env (CAMPAIGN.md decision tree).
