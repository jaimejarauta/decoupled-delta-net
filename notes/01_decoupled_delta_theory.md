# Note 01: Decoupled Delta Recurrence & Surprise Gating Theory

**Author:** Jaime Jarauta  
**Status:** Active theory note  
**Target:** `src/decoupled_delta/model.py`

---

## 1. Mathematical Formulation

Let $x_t \in \mathbb{R}^d$ be an input token, with projections $q_t, k_t \in \mathbb{R}^{d_k}$ ($\|k_t\|_2 = 1$) and $v_t \in \mathbb{R}^{d_v}$.
In standard linear DeltaNet (Yang et al., 2024), the associative state $S_t \in \mathbb{R}^{d_k \times d_v}$ is updated as:
$$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top$$

### The Problem: Continuous Interference
When predicting long repetitive or generic prose, non-zero $\beta_t$ continuously rewrites $S$, eroding stored orthogonal key-value pairs.

### The Solution: Decoupled Surprise Error Reservoir
We split sequence memory into:
1. Continuous dynamic carrier $S_t$.
2. Prediction residual $e_t = v_t - S_{t-1} k_t$.
3. Gated reservoir trigger:
   $$z_t = \mathbb{I}(\|e_t\|_2 > \tau)$$
   If $z_t = 1$, store $(k_t, e_t)$ in circular reservoir $R_t$.
4. Readout:
   $$y_t = q_t S_t + \sum_{m \in R_t} \text{softmax}\left(\frac{q_t k_m^\top}{\sqrt{d_k}}\right) e_m$$


## Run log 2026-10-26: first trained runs (MQAR)

Generator rewritten to the Zoology layout: K unique keys, values from a disjoint range, every key queried once,
loss and recall over all query positions. Question: does the surprise reservoir raise held-out recall over plain
DeltaNet (reservoir disabled) at equal size, data and steps? Rule: margin >= +5 points.

| Arm (T=128, K=8, V=128, d=64, 2 layers, 1500 x 32, seed 0, CPU) | Recall | Surprise rate |
|---|---|---|
| DeltaNet (no reservoir) | 31.2% | 0% |
| Decoupled, absolute threshold 0.5 | 32.4% | 0.1-0.2% |

Decision: FAIL (+1.2 pts). Training shrinks errors, so an absolute threshold stops firing and the reservoir is
unused. Next arm: threshold relative to a running mean of err^2 (`relative=True`).
Manifest: results/mqar_train_absolute_manifest.json.
