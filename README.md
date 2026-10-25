# Decoupled Delta-Net

Sub-quadratic linear recurrence with a discrete, surprise-triggered error reservoir for long-context sequence modeling.

## Overview

Linear recurrent architectures (such as Mamba-2, DeltaNet, and RWKV-7) struggle with Multi-Query Associative Recall (MQAR) and needle-in-a-haystack tasks because continuous low-rank state updates continually overwrite older key-value associations during stretches of predictable tokens.

Decoupled Delta-Net decouples memory updates into two complementary mechanisms:
1. **Continuous Delta Fast Path:** Online gradient descent on predictable sequence dynamics:
   $$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top$$
2. **Surprise-Gated Error Reservoir:** A discrete circular buffer that captures prediction residuals only when surprise exceeds threshold:
   $$\|v_t - S_{t-1} k_t\|_2 > \tau$$

This yields exact factual retrieval without the $O(N^2)$ quadratic memory wall of standard multi-head attention.

## Pre-Registered Gate Criteria
- **Gate 1 (Smoke / Throughput):** Streaming throughput $\ge 150\text{ tok/s}$, per-token latency $\le 10.0\text{ ms}$, and surprise trigger rate $\le 35\%$.
- **Gate 2 (Stage 2 MQAR):** Multi-query associative recall accuracy $\ge 90\%$ at context length $T = 32\text{k}$ tokens.

## Quickstart

### Local (Apple Silicon / CPU):
```bash
# Run smoke benchmark
python scripts/run_smoke.py --config configs/smoke_test.yaml
```

### Google Colab:
Open `notebooks/colab_runner_decoupled_delta.ipynb` in Colab. Includes automatic machine teardown (`runtime.unassign()`) to protect compute credits upon completion.

## License
MIT

