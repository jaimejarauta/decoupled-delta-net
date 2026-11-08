# CAMPAIGN - sharp-steer-vla (Colab L4, one "Run all")

Notebook: `notebooks/colab_runner_sharp_steer_vla.ipynb` -> `run_all.py --smoke` (separate dir) -> `run_all.py`.
Results: Drive `MyDrive/moonshot_results/sharp-steer-vla/` (`$SSV_RESULTS`); HF cache on Drive (`MyDrive/hf_cache`).
Re-running resumes: states, CAA vectors, Hessians and every finished (unit, cell) are skipped. Gate: README.md
"PRE-REGISTERED GATE" (frozen 2026-11-08). GPU: **L4** (0.45B policy; the bottleneck is CPU physics, an A100 buys nothing).

## Stages

| # | stage | items | outputs |
|---|---|---|---|
| 0 | installs: `lerobot[smolvla,libero]==0.7.0`, LIBERO config, MUJOCO_GL=egl | - | - |
| 1 | smoke: pytest GPU pre-flight (abort on fail) + whole pipeline with 2 episodes, 20 steps, m=4 Lanczos | ~50 short rollouts | `smoke/` (thrown away) |
| 2 | pytest pre-flight again (inside `run_all.py`), abort -> INVALID | 6 CPU + 6 GPU tests | `pytest.json` |
| 3 | states: 4 units x 12 unsteered episodes (init 26-37), 4 timesteps each | 48 episodes, <=91 steps | `states_<unit>.pt` |
| 4 | pilot: TS1 and SS1, CAA at L in {4,8,12} x alpha in {1,2,4,8,16} + none (+ oracle), 10 eps (init 40-49) | 33 cells, 330 eps | `pilot_<unit>.jsonl`, `pilot.json` |
| 5 | Hessian: per unit 40 Lanczos steps x 2 gradients x 32 states, + open-loop anti-steer on held-out states | 4 x 80 grads | `hess_<unit>.pt/.json` |
| 6 | main: per unit none (+ oracle) + 4 arms x 4 alphas, 25 eps (init 0-24) | 70 cells, 1,750 eps | `main_<unit>.jsonl` |
| 7 | gate | - | `verdict.json`, `RESULTS.md` |

## Estimated L4 time / credits

25 envs step in parallel (one spawned process each, `SSV_NENV`); the policy runs once per 10 steps on a batch of 25.
Assumed costs (not yet measured on Colab): physics+render ~15-25 ms/env-step on ~8-12 vCPUs -> ~40-60 ms per batched
step; policy call ~0.1-0.2 s per batch; hard reset ~3 s. TS cell (<=300 steps) ~25-45 s, SS cell (100 steps) ~10-15 s.
- states + pilot ~15-25 min; Hessians ~10-15 min; main ~35-60 min; installs/downloads/smoke ~25 min.
- **Total ~1.5-2 h of L4 = ~6-8 CU** (at ~3.8 CU/h). Hard cap: `SSV_MAX_HOURS=3.3` (~12.5 CU) inside `run_all.py`;
  the run stops cleanly before the next cell and a re-run resumes. Budget limit 15 CU.
- Early abandon signals: (i) pre-flight fails (API drift in lerobot / LIBERO EGL) -> fix, do not run; (ii) pilot early
  stop (all CAA cells 0/10) -> FAIL, ~30 min spent; (iii) TS oracle < 50% in the first main cells -> the policy cannot
  do the target tasks, TS is INVALID (SS still informative).

## Decision tree

- **INVALID** (pre-flight, or both validity checks): check `pytest.json`, oracle and none rates. Fix plumbing; dated
  README note before re-running into a fresh `SSV_RESULTS`.
- **FAIL via kill rule / early stop** (CAA < 10% at all alphas in both behaviours): report "SmolVLA is not steerable by
  CAA from contrastive instructions" (consistent with COAST / DiMaS). Optional follow-up with its own gate: steer the
  action expert's hidden states instead of the VLM (where COAST/DiMaS operate), same Hessian code.
- **FAIL via H1_FAIL** (CAA steers, sharp removal does not help or the random control helps equally): the 2610.16084
  sharp-direction effect does not transfer to closed-loop VLA steering at this scale. Report the open-loop anti-steer
  numbers next to it: if open-loop anti-steer is high for sharp_only but closed loop is unaffected, the dynamics wash it
  out (the opposite of the hypothesis), which is itself the finding.
- **MIXED** (WEAK steering, or H1 in one behaviour with the other not steerable): one more pre-registered run on
  A6000/Blackwell with 50 init states x 3 seeds per cell and a third behaviour (approach height) before any claim.
- **PASS**: replicate before claiming. Scale-up path (user's A6000 / Blackwell host, same code): (1) N: all 50 init
  states x 3 seeds; (2) second policy: `lerobot/pi05_libero` (PaliGemma backbone, needs a `text_layers` accessor and a
  width check in the pre-flight); (3) OpenVLA-7B LIBERO-Goal (discrete action tokens: loss = action-token NLL, which
  is the exact 2610.16084 loss); (4) k sweep {1, 3, 5, 10, 15, 25} as in 2610.16084.
