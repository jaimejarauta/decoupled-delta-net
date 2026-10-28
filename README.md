# `hybrid-kv-cliff`: KV-Cache Token Eviction vs. Quantization in Sparse-Attention Hybrid LLMs

Empirical evaluation testing the **Depth-Redundancy Cliff** across production hybrid SSM/Linear-Attention + Full-Attention language models (`ibm-granite/granite-4.0-h-1b` with 4/40 attention layers, `Qwen/Qwen3.5-0.8B/2B/4B-Base` with 6–8 attention layers, `tiiuae/Falcon-H1-1.5B-Base` parallel Mamba-2 + Attention) against matched pure-Transformer controls (`Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`).

## Core Falsifiable Claim
In pure Transformers ($M = 28\text{--}36$ full-attention layers), cross-layer head redundancy absorbs single-layer KV token eviction misses (`SnapKV`, `PyramidKV`). In sparse-attention hybrids ($M \in \{4, 6, 8\}$), each full-attention layer acts as a non-redundant retrieval bottleneck: byte-matched token eviction ($4\times\text{--}8\times$) triggers a steep accuracy cliff—especially under multi-turn query shift—whereas retaining 100% of token positions via asymmetric per-channel Key / per-token Value quantization (`KIVI`) preserves exact retrieval within $\le 2.5$ pts of uncompressed BF16.

## Compression Arms (Exact Physical Byte Accounting)
1. `full`: Uncompressed BF16/FP16 KV cache + untouched linear recurrent states (`1.0x`).
2. `snapkv_4x` / `snapkv_8x`: FlashAttention-compatible $O(W_{\text{obs}} \cdot L)$ observation-window salience (`W_obs=64`, kernel-5 1D pooling, 4 sinks, 64 local tail).
3. `snapkv_ungrouped_4x`: Per-query-head selection oracle isolating GQA group width ($G \in \{4, 8\}$) from attention depth ($M \in \{4, 6, 8\}$).
4. `pyramidkv_4x`: Linearly funneled $1.5\bar{B} \to 0.5\bar{B}$ layerwise budget across the $M$ full-attention layers.
5. `kvzip_uniform_4x`: Query-agnostic probe + $\|v_j\|_2$-weighted salience robust against single-query overfitting.
6. `kivi_4bit` (~`3.6x` physical bytes) & `kivi_2bit` (~`6.4x` physical bytes): Zero token eviction; group-32 per-channel Key + per-token Value asymmetric quantization + 4 sinks + 32 BF16 residual tail.
7. `sink_tail_4x`: `StreamingLLM` naive eviction floor (4 sinks + recent tail).

## Quickstart
```bash
pytest -q
python scripts/run_kill.py run --models Qwen/Qwen3-1.7B-Base Qwen/Qwen3.5-2B-Base ibm-granite/granite-4.0-h-1b --length 8192 --n 50
python scripts/run_kill.py report --out results/kill.jsonl
```

