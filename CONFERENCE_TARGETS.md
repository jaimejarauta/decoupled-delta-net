# Decoupled Delta-Net: Publication & Conference Strategy

**Author:** Jaime Jarauta  
**Primary Target:** **ICML 2027** / **NeurIPS 2027** (Sequence Modeling & Architecture Track)  
**Secondary Target:** **ICLR 2028** / **COLING 2028**  
**Strategic Workshop:** **NeurIPS Workshop on Associative Memory & Linear Attention**

---

## 1. Target Venues & Submission Deadlines

| Venue | Track | Submission Window | Notification | Acceptance Rate |
| :--- | :--- | :---: | :---: | :---: |
| **ICML 2027** | Deep Learning Architectures / Sequence Models | Feb 2027 | May 2027 | ~23% |
| **NeurIPS 2027** | Attention & Recurrence Frontiers | May 2027 | Sep 2027 | ~24% |
| **ICLR 2028** | Representation Learning & Long Context | Sep 2027 | Jan 2028 | ~25% |

---

## 2. Reviewer Red Lines & Mandatory Epistemic Evidence

1. **Needle-in-a-Haystack Baseline Hawkishness:** Must compare directly against Mamba-2, DeltaNet, and FlashAttention-2 across context lengths $T \in \{4\text{k}, 16\text{k}, 64\text{k}\}$, proving that surprise reservoir eliminates the standard linear memory decay.
2. **Surprise Threshold Sensitivity:** Must provide an ablation over $\tau \in \{0.1, 0.3, 0.5, 0.8, 1.2\}$, demonstrating that the model maintains stability across diverse token entropy regimes.
3. **Throughput Pareto:** Demands tokens/sec vs sequence length scaling curves proving linear $O(T)$ runtime.

---

## 3. Working Title & Hook

**Working Title:** *Decoupled Delta-Net: Eliminating Associative Memory Decay via Surprise-Gated Error Reservoirs*

**Abstract Hook:** Linear recurrent sequence models offer linear inference complexity but suffer catastrophic associative forgetting during needle-in-a-haystack tasks due to continuous, low-rank overwriting of state matrices. We present Decoupled Delta-Net, which decouples continuous delta-rule state evolution from a discrete error reservoir triggered exclusively by prediction surprises. On 32k-token Multi-Query Associative Recall, Decoupled Delta-Net matches full softmax attention retrieval while preserving $O(1)$ memory state complexity per head.

