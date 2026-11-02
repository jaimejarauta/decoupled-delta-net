# plasticity-forge

Question: can LLM-guided program search over symbolic *local* plasticity rules find a
backprop-free rule that beats hand-designed ones (DFA, FA, Oja+readout, random-features+readout)
on datasets and architectures it was never evolved on?

## Prior art (checked 2026-11-02, before any experiment)

| Work | What it does | Delta to this project |
|---|---|---|
| Shervani-Tabar & Rosenbaum 2023, "Meta-learning biologically plausible plasticity rules with random feedback pathways" (Nat. Commun.) | Meta-learns coefficients over a fixed library of local terms on top of FA feedback; finds Oja-like terms help. | Closest. They search a fixed linear term library with gradient meta-learning; here an LLM writes free-form rule programs, fitness is plain accuracy, and transfer is to new datasets + deeper nets vs tuned DFA. |
| Jordan et al. 2021 (eLife), "Evolving interpretable plasticity for spiking networks" (CGP) | Cartesian genetic programming over symbolic plasticity rules, spiking nets, small tasks. | Same idea (symbolic rule search), non-LLM operator, no deep supervised MLP benchmark vs DFA. |
| Confavreux et al. 2020 (NeurIPS) / 2023 / 2025 bioRxiv "Memory by a thousand rules" | Meta-learned / evolved plasticity rules in (spiking) recurrent nets. | Neuroscience targets (stability, memory), not supervised accuracy vs DFA. |
| Real et al. 2020, AutoML-Zero; Chen et al. 2023, Lion ("Symbolic discovery of optimization algorithms") | Evolutionary program search over learning algorithms / optimizers with transfer checks. | Searches algorithms that use true gradients; here the rule is barred from gradients and other layers. |
| FunSearch (Romera-Paredes 2024), AlphaEvolve (2025), LLM-evolved RL update rules (arXiv 2603.28416, RLDiscover 2610.09218) | LLM as mutation operator in evolutionary program search. | Method template is borrowed; the target (local plasticity rule for deep supervised nets) was not found in the search. |

No direct collision found (searches: LLM + evolutionary + local/plasticity/learning rule + backprop-free).
The method itself is not new (FunSearch template); the only possible contribution is the empirical answer.

## Setup (fixed before running)

- Net: tanh MLP, hidden layers without bias, output layer with bias. Every non-BP method trains the
  output layer with its local delta rule `e^T h_L` (e = softmax − onehot). Optimizer: Adam for all.
- Rule interface: `delta(pre, post, post_pre, err_proj, W, stats) -> dW` (pseudo-gradient, Adam
  descends it). `err_proj = e @ B_l`, B_l fixed random (the DFA signal). Rule is exec'd in a fresh
  namespace per layer (no cross-layer state); source tokens `import`, `__`, `eval`, `exec`, `open(`,
  `autograd`, `backward`, `.grad`, ... are rejected; runs in a subprocess with a timeout.
- Evolution (MNIST only): 20k of the 50k train split, val = MNIST val[:5000] (val carved from MNIST
  train), arch 784-256-256-10, 2 epochs, bs 100, seed 0, fitness = max val acc over
  Adam lr in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}. Operator: `claude -p` (haiku, some sonnet), hard cap 200 calls,
  every call logged in `runs/llm_calls.jsonl`. Seeds in the archive: DFA, Oja.
- Selection (MNIST only): top-5 evolved rules re-run on the 50k MNIST train, 3 epochs, seeds 0-2,
  lr tuned on the same grid, scored on MNIST val[5000:]. Highest mean = R*. R* frozen before transfer.
- Transfer (never seen during evolution): FashionMNIST and KMNIST (KMNIST from the HF mirror
  `tanganke/kmnist`, 60k/10k, the official host timed out). 50k train / 10k val (carved from train) /
  10k official test. Archs A = 784-256-256-10, B = 784-512-512-512-10. 5 epochs.
  Methods: BP (reference), FA, DFA, RF+readout (hidden frozen at init), Oja+readout, R*.
  LR per (method, dataset, arch) tuned on val over {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, seed 0.
  Then seeds 1-5 at the tuned lr, test accuracy. Same seed = same init and same feedback matrices.

## Pre-registered gate (2026-11-02, written before evolution; do not edit)

Per cell (dataset x arch, 4 cells), with paired-by-seed differences over 5 seeds and a 95% t-CI (df=4):
- C1: mean(R* − DFA) >= 1.0 pt AND CI lower bound > 0.
- C2: mean(R* − RF+readout) >= 2.0 pt.

Verdict:
- **PASS**: C1 and C2 hold in all 4 cells.
- **FAIL**: C1 holds in at most 1 of 4 cells.
- **MIXED**: anything else.

Secondary (not part of the gate): is R* secretly DFA? Report its symbolic form and the cosine
similarity between R*'s dW and DFA's dW on the same batches.

## Files

- `forge/core.py` — data, training (BP/FA/DFA/RF/rule), rule sandbox, subprocess entry.
- `forge/evolve.py` — LLM-guided search loop.
- `forge/transfer.py` — selection + held-out transfer + gate evaluation.
- `tests/test_core.py` — core logic tests.
- `runs/` — logs, archive, results JSON. `RESULTS.md` — outcome.

## Protocol log (append-only)

- 2026-11-02: 4-call trial of the loop (logs in `runs/trial/`, counted against the 200-call budget,
  so the main run uses 196). DFA's best lr sat on the grid edge (3e-3), so the lr grid was widened
  to add 1e-2 for evolution, selection and transfer, before the main run. Gate untouched.
- 2026-11-02 18:50: network outage 17:00-18:48 (API ENOTFOUND/ECONNRESET). 9 of the first 33 logged
  calls (r020-r027, r030) returned an API error string instead of a rule; they were archived as
  "no code block" with fitness 0, count against the budget, and were not retried. The machine is
  also heavily loaded (load avg ~75 on 10 cores), so evaluations are ~5x slower than planned.
  To stay near the compute budget, evolution is stopped once 80 calls are logged (well under the
  200 cap; the cap was a ceiling). Calls still in flight when the process is killed (<=4) are not
  logged and are not used. Selection/transfer/gate protocol unchanged.
- 2026-11-02 19:05: correction to the entry above: evolution was stopped at 37 logged calls, not 80.
  After the outage, calls hung for >7 min (the subprocess timeout does not fire while claude's
  child processes hold the pipe) and machine load hit ~140. Final archive: 2 seeds + 33 rule records
  (26 valid, 7 API-error); 4 logged calls (r033-r036) had their evaluations killed and are unused.
  Selection runs on the 26 valid rules. Gate and transfer protocol unchanged.
