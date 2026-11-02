# evo-construct — results (2026-11-02)

**Verdict against the frozen gate (README.md): MIXED.** Condition A failed: the LLM arm "beat" the classical arm on 1 of 3 problems, and 2 of 3 were required. Condition B passed: LLM runs reached at least 99.9% of the best-known value on all 3 problems.

The one "beat" (circ32, +1.5e-4 relative) is within seed noise. One of 2 LLM runs and 1 of 3 classical seeds reached the same optimum, 2.9395728. Read plainly: at equal wall-clock on this laptop, Claude Haiku program evolution did not do better than scipy SLSQP multistart with basin hopping. The classical arm reached the best-known value sooner on every problem where both arms reached it.

## Main table (equal wall-clock: classical best-so-far at each LLM run's effective elapsed time E)

| problem | best-known | LLM runs (a, b) | E (s) | LLM proposals | classical @E, 3 seeds × 2 E | mean rel. adv. LLM | beats? |
|---|---|---|---|---|---|---|---|
| circ26 (max) | 2.635983 | 2.6319369, 2.6359831 | 734, 746 | 14, 15 | 2.6359831 (all 6) | −7.7e-4 | no |
| circ32 (max) | 2.93957 | 2.9335171, 2.9395728 | 724, 743 | 15, 14 | mean 2.9361040, range [2.9337731, 2.9395728] | +1.5e-4 | yes (noise-level) |
| ratio16 (min) | 12.88923 | 12.8892299, 12.8892299 | 587, 1970 | 3, 3 | 12.8892299 (all 6) | −5e-15 (tie) | no |

Fraction of best-known reached, and the first time it was reached:

| run | circ26 | circ32 | ratio16 |
|---|---|---|---|
| classical seed 0 | 1.000000 @ 130 s | 1.000001 @ 82 s | 1.000000 @ 0.2 s |
| classical seed 1 | 1.000000 @ 255 s | 0.998028 (never 99.9%) | 1.000000 @ 0.2 s |
| classical seed 2 | 1.000000 @ 225 s | 0.998434 (never 99.9%) | 1.000000 @ 1.3 s |
| LLM run a | 0.998465 (never 99.9%) | 0.997941 (never 99.9%) | 1.000000 @ 480 s |
| LLM run b | 1.000000 @ 548 s | 1.000001 @ 67 s (99.9%), 368 s (99.999%) | 1.000000 @ 488 s |

Secondary result (score at t = 300 s): on circ26 the classical arm led with 2.6359831 on all 3 seeds, against 2.62982 and 2.62535 for the LLM runs. On circ32 each arm had one run at about 2.93949 or higher. On ratio16 the LLM arm still had the naive grid (18.0), because its first Haiku call took 465 s under machine load; the classical arm was at 12.8892299.

Neither arm improved on a best-known value beyond rounding: circ32 2.9395728 against the 2.93957 reported for Xpress/SCIP, and ratio16 12.8892299 against 12.88923.

Curves: `runs/curves.png` (blue = classical, red = LLM, dotted = best-known, log time axis).

## LLM call accounting (hard cap 250)

- `logs/llm_calls.jsonl` holds 73 records: 1 manual CLI latency check, 3 smoke-test calls, and 69 calls in the main runs.
- **Outage and machine stall.** Between about 16:00 and 16:59 CET the whole machine stalled. Load reached about 166, and my 30 s load logger skipped 11 min and then 48 min. The network was also down (the coordinator reported ENOTFOUND). The last call of each of the 4 circle runs returned after 3604–3627 s with no code ("no code block"). The 180 s subprocess timeout did not fire, which fits the parent process being frozen rather than slow.
  - **How these were counted:** `analyze.py:clean()` drops every call with latency over 600 s, which cannot happen under a working 180 s timeout. Those calls are not counted as proposals. The run's effective elapsed time E is set to when the stalled call started (724–746 s).
  - This removed 4 failed circle calls and 1 successful but stalled ratio16 call (ratio16_a #3, 1419 s, no improvement).
  - Main-run proposals counted: 64 (circ 29 + 29, ratio16 6).
  - Total calls consumed, including failures: 73 of 250.
- **The outage did not distort the baseline clock in the comparison.** The classical runs use `time.time()` too, so their curves after about 780 s are contaminated by the stall. But every comparison is made at E ≤ 746 s for the circle problems, and no classical curve improved after 673 s (circ32 seed 2). The last improvements of the other seeds were at 517 s or earlier. The stalled part of the classical curves never enters the result.
- **ratio16 ran under extreme load (≈ 105–247).** The LLM arm managed only 3 proposals per run, and its first call took about 465 s. This penalises the LLM arm's speed on ratio16, but not its final score: both arms reached the same optimum to 1e-14.
- **Remaining-load caveat:** load averages during wave A were about 20–50 before the stall, against 4 at idle. Both arms ran at the same time with one thread each, so the contention was shared. Absolute times are about 2–5× slower than on an idle machine: in tuning at load about 4, classical circ26 reached 2.635983 within 90 s.

## What the LLM discovered

All 64 proposals ran and were feasible ("ok"), helped by the `best_so_far` fallback. Keyword counts over all proposed programs:

| technique | circ26 | circ32 | ratio16 |
|---|---|---|---|
| SLSQP | 25 | 14 | 6 |
| L-BFGS-B with increasing penalty | 13 | 19 | 0 |
| random perturbation / annealing walk | 24 | 19 | 5 |
| linprog (sequential LP) | 4 | 15 | 0 |

- **Reinventing the classical arm.** Haiku mostly reinvented my baseline: SLSQP with analytic Jacobians plus basin-hopping-style perturbations of the incumbent. It also used L-BFGS-B with a rising penalty weight (10 → 1e6).
- **One idea not in the baseline.** In circ32_b #4, Haiku wrote a sequential-LP refiner. It linearises the pair-distance constraints, and because distance is convex the linearisation underestimates it, so every LP step stays exactly feasible. That program took circ32 from 2.939492 to 2.939573.
- **Value came from incumbent refinement.** Most improvements came from programs refining `best_so_far` within their 20 s budget. That is an iterated local search with LLM-written local-search code, and its pace is set by about 20–40 s of LLM latency per step.

## What was learned

1. **Prior art predicted this, and the result is consistent.** Prior work found that plain NLP formulations solved by Xpress/SCIP match AlphaEvolve's numbers (arXiv 2601.05943), and that simple LLM baselines match code-evolution pipelines (arXiv 2602.16805). With open-source scipy on one laptop core, a 60-line SLSQP + basin-hopping loop reached the published best-known values for circ26 (3/3 seeds) and ratio16 (3/3 seeds, within 1.3 s). It reached the circ32 value (Xpress/SCIP 2.93957) on 1/3 seeds.
2. **The LLM arm's per-step cost is LLM latency.** At about 40 s per step it gets only about 15 proposals in 12 minutes, while SLSQP gets about 1,200–2,700 local solves. With programs this short and a cheap model, the proposals are mostly classical local-search code that the human-written baseline already contains.
3. **circ32 is where both arms vary most between runs.** It is the only problem where a stronger method could show a difference. Larger n, or problems without a clean smooth NLP formulation, are where the LLM loop might matter.

## Limitations

- **Small n.** There were 2 LLM runs and 3 classical seeds per problem, so no meaningful confidence interval is possible. Ranges are reported instead.
- **Effective budget halved by the stall.** The circle runs were cut from the planned 1500 s to about 740 s. ratio16 ran under load above 100.
- **Baseline tuning disclosed.** The baseline was tuned briefly on circ26 and circ32: 4 configs × 90 s on seed 100, which is not one of the evaluation seeds. The LLM prompt contained the best-known value; the baseline does not use it.
- **Model mix.** Only Haiku was used. Sonnet was never tried.

## Next (if continued)

- **Re-run on an idle machine.** Rerun the circle problems with the full 1500 s budget once the shared machine is idle. About 140 LLM calls remain.
- **Increase seeds on circ32.** Use at least 5 per arm there, since it is the only problem with real variance between runs.
- **Test where SLSQP struggles.** For example circle packing with n ≥ 50, or Heilbronn-type nonsmooth objectives, where a single smooth NLP is weak.

Files: `problems.py` (verifier), `baseline.py`, `evolve.py`, `runner.py`, `analyze.py`, `run_wave.sh`, `test_core.py` (4 passing), `runs/main/*.json` (all curves, solutions and LLM programs), `runs/summary.json`, `runs/curves.png`, `logs/llm_calls.jsonl`, `logs/load_wave{A,B}.txt`.
