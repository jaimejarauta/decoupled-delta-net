# kernel-forge

Question: can a laptop LLM-evolution loop (headless `claude -p`, ≤200 calls total) write
Metal kernels (`mx.fast.metal_kernel`) that beat MLX's built-ins on Apple M4 for
decode-relevant, memory-bound ops?

## Prior art (checked 2026-11-02, before any experiment)

- **KernelBench** (Ouyang et al., ICML 2025, arXiv 2502.10517): 250 PyTorch tasks, CUDA, metric fast_p vs eager and torch.compile. One-shot LLMs beat eager on <20% of tasks.
- **Sakana "AI CUDA Engineer"** (Feb 2025): claimed 10-100x over PyTorch; the system reward-hacked the evaluator (bypassed correctness checks, reused results from the reference run). Independent re-evaluation dropped speedups to ~1.1x. Main lesson used here: the LLM only supplies kernel text; the harness owns inputs, outputs, shapes and fresh-random correctness, and winners are read by hand.
- **AlphaEvolve** (DeepMind 2025, arXiv 2506.13131): Gemini evolution loop; 23% speedup of a Pallas matmul tiling heuristic, ~32% on a FlashAttention XLA-IR kernel. Datacenter TPU/GPU, huge budget.
- **Gimlet Labs, "AI-generated Metal kernels"** (2025): frontier LLMs write Metal kernels for KernelBench on M4 Max, baseline = eager PyTorch MPS (no torch.compile, no MLX). Mean ~1.2x (v0.1); gains mostly from fusion on Level-2. Found trivial-output hacks (o3 "9000x").
- **Metal-Sci** (arXiv 2605.09708, May 2026): (1+1) evolutionary LLM loop for Metal scientific kernels (stencils, n-body, FFT) on M1 Pro; reports self-speedups and held-out-size failures. Not ML ops, not vs MLX.
- **ZMLX** (PyPI): hand/tool-written fused Metal kernels for MLX MoE decode, +1-13% end-to-end decode. Not LLM-evolved, no per-op comparison vs MLX built-ins reported.

**Delta of this project:** the baseline is MLX itself (`mx.fast.*`, `mx.quantized_matmul`) and
`mx.compile`d compositions, not eager PyTorch MPS. The kernels are LLM-evolved on a laptop with a
200-call budget. Nothing above measures that. It is a small, empirical delta, not a new method.

## Targets (all float16 activations, M4 GPU)

| id | op | built-in baseline | compiled baseline | timing shapes |
|---|---|---|---|---|
| A | `h = x + r; y = rms_norm(h) * w` (both h and y output, eps=1e-5) | `x+r` then `mx.fast.rms_norm` | `mx.compile` of same | (rows,H) = (1,2048), (1,4096), (32,4096), (128,4096) |
| B | `silu(g) * u` | `nn.silu(g)*u` eager | `mx.compile` of same | (rows,I) = (1,8192), (1,14336), (32,14336), (128,11008) |
| C | 4-bit affine (gs=64) dequant + matvec, batch 1: `x @ W.T` | `mx.quantized_matmul(..., transpose=True)` | `mx.compile(x @ dequantize(W).T)` | (K,N) = (4096,4096), (4096,14336), (14336,4096), (2048,151936) |

Odd correctness shapes: A (1,1),(3,1000),(7,4097),(5,333); B (1,1),(3,999),(2,14335),(5,7);
C (64,1),(128,7),(4096,4097),(192,333).

## Protocol

- Timing: same process, same device, same dtype, warmup, sample = wall time to build and `mx.eval`
  a chain of K=32 dependent calls (output feeds next input where shapes allow; C non-square is
  independent calls), divided by K. Candidate and baseline samples interleaved (alternating order),
  ≥60 samples per pair at final re-time. Speedup = median of paired ratios t_base/t_cand; 95% CI by
  bootstrap over pairs. Machine is shared with other GPU jobs; finalists re-timed at the end.
- Correctness: fp32 reference on the same fp16 inputs, fresh random inputs on every evaluation.
  A/B: |out-ref| ≤ 1e-2 + 1e-2|ref|. C: |out-ref| ≤ 0.02·|ref| + 0.02·std(ref). Built-ins must pass the same check.
- Anti-reward-hacking: LLM returns only Metal source + threadgroup size + grid expression; harness
  owns buffers, output shapes and inputs. Outputs of the timed run itself are also checked.
  Winning kernels are read by hand.
- LLM: `claude -p --model sonnet`, cwd=/tmp, every call logged to `runs/llm_log.jsonl`. Hard cap 200 calls.

## Pre-registered gate (frozen 2026-11-02, before the main experiment; never edited afterwards)

For each target, take the single best LLM-evolved kernel (by search fitness). The target **passes** iff:
1. it passes **1000** correctness checks on fresh random inputs spread over the 4 timing shapes and
   the 4 odd shapes, **and**
2. on **≥3 of 4** timing shapes, the median paired speedup vs the **faster of {built-in, mx.compile
   baseline}** at that shape is **≥1.25×** with bootstrap 95% CI lower bound **>1.0**.

Verdict: **PASS** if ≥2/3 targets pass; **MIXED** if exactly 1/3; **FAIL** if 0/3.
My hand-written kernel is reported as a third baseline (secondary, not part of the gate).
