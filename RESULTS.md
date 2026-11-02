# plasticity-forge — results (2026-11-02)

**Verdict vs the pre-registered gate: FAIL.** C1 (R* beats DFA by >= 1.0 pt with the 95% CI above 0) held in 0 of 4 cells.
C2 (R* beats RF+readout by >= 2 pts) held in 4 of 4 cells, but DFA itself also clears that bar.

## What ran

- Evolution (MNIST only): 41 LLM calls logged in total: 37 in the main run (31 haiku, 6 sonnet) and 4 in the trial run (`runs/trial/`).
  - 9 calls failed during a network outage (7 ENOTFOUND, 2 ECONNRESET). Each was archived as "no code block" with fitness 0 and was not retried.
  - 26 valid rules were evaluated. A further 4 logged calls (r033-r036) had their evaluations killed when the run stopped.
  - The run was stopped early, far below the 200-call cap, because of the outage and a load average of 140-280 on a 10-core machine. See the protocol log in README.md.
- Selection (MNIST val[5000:], 3 seeds, 3 epochs, 50k train). The top 5 evolved rules scored 97.11-97.27 vs DFA's 96.89. R* = r009 at 97.27 (r010 is the same code and got identical numbers).
- Transfer: FashionMNIST and KMNIST, archs A (784-256-256-10) and B (784-512x3-10), 5 epochs. lr was tuned per method and cell on val (seed 0), then seeds 1-5 were scored on test.

## R* (r009), symbolically

```
d   = err_proj * (1 - post^2)            # = DFA signal
d   = d / rms(d)                         # batch RMS normalisation
pc  = pre - EMA_0.99(mean_batch(pre))    # presynaptic mean-centring
dW  = d^T pc / B + 1e-4 * W              # + L2 decay
```

This is DFA plus three generic tweaks. Under Adam, the RMS normalisation mostly matters only through its interaction with the 1e-4 L2 term. All of the top-10 evolved rules have the same DFA core, with some combination of normalisation, centring, clamping and decay.

Cosine(dW_R*, dW_DFA) on FashionMNIST along a DFA trajectory was 0.67-0.89 for layer 0 and 0.90-0.98 for layer 1. The discovered rule is DFA with centred inputs; it is not a new learning mechanism.

## Test accuracy (%; mean ± sd over 5 seeds; differences are paired by seed, 95% t-CI with df=4)

| cell | BP | FA | DFA | RF+readout | Oja+readout | R* | R*-DFA [95% CI] | R*-RF | C1/C2 |
|---|---|---|---|---|---|---|---|---|---|
| FashionMNIST/A | 87.01 ± 0.79 | 86.77 ± 0.18 | 86.81 ± 0.15 | 82.56 ± 0.22 | 74.27 ± 0.52 | 87.08 ± 0.24 | +0.28 [+0.13, +0.42] | +4.53 | N/Y |
| FashionMNIST/B | 86.35 ± 0.16 | 86.48 ± 0.26 | 86.48 ± 0.55 | 82.55 ± 0.48 | 70.92 ± 1.21 | 86.79 ± 0.25 | +0.31 [-0.47, +1.09] | +4.24 | N/Y |
| KMNIST/A | 88.19 ± 0.46 | 87.50 ± 0.57 | 87.45 ± 0.55 | 65.96 ± 0.58 | 63.25 ± 0.63 | 86.95 ± 0.66 | -0.50 [-1.55, +0.55] | +21.00 | N/Y |
| KMNIST/B | 89.19 ± 0.54 | 88.11 ± 0.19 | 88.55 ± 0.46 | 68.60 ± 0.22 | 62.60 ± 0.37 | 88.17 ± 0.42 | -0.37 [-0.87, +0.12] | +19.57 | N/Y |

Tuned lrs are in `runs/transfer.json`. No run errored.

## Reading

- The MNIST gain of +0.4 pt over DFA (selection) shrinks to +0.3 on FashionMNIST and turns into -0.4 to -0.5 on KMNIST. That fits a small gain fitted to MNIST that does not transfer.
- The search space collapsed onto DFA. The LLM was shown DFA as the best seed and made local edits to it. With the error signal restricted to a random projection, there may be little left to find beyond DFA plus normalisation.
- Oja+readout was worse than RF+readout everywhere. Unsupervised Oja on tanh units hurt the random features within 5 epochs.

## Caveats

- Evolution used 41 of the 200 calls (26 valid rules), so the search was small. A larger search would be unlikely to change the verdict, because every top rule had the same DFA core and their selection scores were within 0.2 pt of each other.
- The RF+readout best lr was at the top edge of the grid (1e-2), so RF may be slightly under-tuned. This favours R* on C2, which does not change the verdict.
- BP's lr for FashionMNIST/B was at the bottom edge of the grid (1e-4), and BP is no better than DFA on FashionMNIST at 5 epochs. BP is a reference only and is not part of the gate.
- KMNIST came from the HF mirror `tanganke/kmnist` (60k/10k, balanced) because the official host timed out.
- The 2026-11-02 compute budget was exceeded in wall-clock time because of the outage and the shared load (load avg up to ~280). The actual compute used was small.

## Lessons

1. Keep the DFA error signal out of a search seeded with DFA, or penalise similarity to DFA. Otherwise LLM mutation reproduces DFA with tweaks.
2. A fitness of 2 epochs on 20k MNIST has about ±0.2 pt noise, which is the same size as the gains being selected on. Use multi-seed fitness or a harder fitness task.
3. `subprocess.run(timeout=...)` on `claude -p` did not stop hung calls (child processes keep the pipe open). Use process groups (`start_new_session=True` + `os.killpg`) and retry on API-error output.

## Next step

Not recommended to continue in this form. The only variant worth one cheap run is a search that excludes `err_proj` (truly unsupervised hidden rules + readout), with RF+readout as the bar to beat. Prior art (SoftHebb, Oja/GHA variants) makes even that a low prior.

## Files

- `README.md`: prior art, frozen gate, protocol log
- `forge/core.py`, `forge/evolve.py`, `forge/transfer.py`, `tests/test_core.py` (9 tests pass)
- `runs/llm_calls.jsonl` (every call), `runs/archive.jsonl`, `runs/selection.json`, `runs/transfer.json`, `runs/gate.txt`, `runs/cosine.json`
