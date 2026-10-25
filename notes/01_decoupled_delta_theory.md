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
