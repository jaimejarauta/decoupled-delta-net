# evo-construct

Question: at **equal wall-clock on one laptop core**, does LLM-guided program evolution
(headless Claude Haiku as mutation operator, ≤250 LLM calls total for the project) beat a tuned
classical optimizer (scipy SLSQP multistart + basin hopping) on three AlphaEvolve-benchmark
extremal-geometry problems?

## Prior art (searched 2026-11-02)

| Work | What it does | Delta to this project |
|---|---|---|
| AlphaEvolve (Novikov et al. 2025; Georgiev, Gómez-Serrano, Tao, Wagner, arXiv 2511.02864) | LLM program evolution; circ26 2.63586, circ32 2.93794, ratio16 12.889266 | Large unknown compute, no classical equal-budget baseline |
| ShinkaEvolve (Sakana 2025), OpenEvolve, CodeEvolve (2510.14150), GigaEvo (2511.17592) | Open re-implementations; circ26 ≈ 2.63598 | Compare LLM pipelines to each other / to AlphaEvolve, not to a classical optimizer at equal wall-clock |
| **Gideoni, Risi, Gal, "Simple Baselines are Competitive with Code Evolution" (arXiv 2602.16805)** | IID LLM sampling / sequential conditioned sampling match ShinkaEvolve (ratio16 12.88923, circ26 2.63590, $20/problem, 1–3 h) | Their baselines are still LLM-based; no non-LLM optimizer. Closest work. |
| **"classical global solvers on AlphaEvolve benchmarks" (arXiv 2601.05943)** | FICO Xpress / SCIP on plain NLP formulations match or beat AlphaEvolve (circ32 2.93957, ratio16 12.88924) "within seconds or minutes" | Commercial/global solvers, no time-matched anytime comparison to an LLM loop; we use open-source scipy only on one laptop core, with the LLM loop's latency counted |
| FM Agent (arXiv 2510.26144) | ratio16 12.889230 using SLSQP multistart inside an agent | Suggests plain SLSQP suffices for ratio16 |

Honest positioning: the two bold rows already make it likely that classical methods are
competitive. This project is a small, cheap, time-matched replication/negative control
(open-source optimizer vs a cheap LLM on a laptop), not a new method. No novelty claim is made.

Best-known values used (problems.py): circ26 2.635983 (ShinkaEvolve-level), circ32 2.93957
(Xpress/SCIP), ratio16 12.88923 (FM Agent / 2601.05943). Verifier tolerance 1e-9 (both arms).

## Setup (frozen before main runs)

- Classical arm (`baseline.py`): SLSQP with analytic Jacobians; 20% fresh random starts, 80% basin hops
  around the incumbent (log-uniform jitter in [10^-2.5, 0.1] + relocating 0–2 circles); exact
  radius-shrinking repair. Tuning disclosed: one 90 s run (seed 100) of 4 (p_fresh, sigma)
  configs on circ26/circ32 + a 30 s smoke test per problem. Main seeds 0,1,2 (unseen).
- LLM arm (`evolve.py`): top-3 archive in prompt (code + score), one of 3 mutation styles at random,
  program API `solve(time_limit=20s, best_so_far)` (may refine the incumbent, AlphaEvolve-style),
  sandboxed subprocess (`python -I`, hard timeout 30 s), same verifier. Model: claude haiku,
  `--tools ""`, cwd /tmp. Initial program: naive grid. Every call logged in logs/llm_calls.jsonl.
- Budget per run: 1500 s wall-clock and ≤36 LLM calls, whichever first. 2 LLM runs per problem
  (run ids a,b), 3 classical seeds per problem, 1 core each (OMP_NUM_THREADS=1), arms run concurrently.
- Equal wall-clock comparison: for each LLM run with elapsed time E, classical score = best-so-far
  of each classical seed at time ≤ E (from its anytime curve).

## Pre-registered gate (2026-11-02, frozen; not to be edited after the main run)

For each problem: LLM arm **beats** classical iff mean(best score of the 2 LLM runs) is better than
mean(classical best-so-far at time E, over 3 seeds × 2 E values) by more than 1e-6 relative.
Ties within 1e-6 do not count as beating.

- Condition A: LLM beats classical on ≥ 2 of 3 problems.
- Condition B: at least one LLM run reaches ≥ 99.9% of best-known on ≥ 1 problem
  (maximize: score/best ≥ 0.999; minimize: best/score ≥ 0.999).
- **PASS** = A and B. **MIXED** = exactly one of A, B. **FAIL** = neither.

Secondary (reported, not gated): scores at t = 300 s; classical fraction-of-best; LLM call
outcomes (ok / error / infeasible / timeout); categories of LLM programs.

## Files
problems.py (verifier), baseline.py, evolve.py, runner.py, analyze.py, test_core.py, runs/, logs/.
