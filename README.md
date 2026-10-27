# hybrid-state-rollback

Kill experiment and core library for **Top Pick C1** in `research/RESEARCH_DECISION_2026-10-26.md`:
*Same-Subspace State Overwrite and Synchronized State-KV Rollback in Hybrid Recurrent Reasoning Models*.

## Motivation & Live `transformers >= 5.0` Gap
In HuggingFace `transformers >= 5.0`, hybrid `DynamicCache` marks every `LinearAttentionLayer` and `LinearAttentionAndFullAttentionLayer` (`Qwen3.5`, `Falcon-H1`, `Zamba2`, `Granite-4.0-H`, `Bamba`) as `is_croppable = False` whenever `recurrent_states` are initialized. Naive KV-cache truncation (`SkipKV`, `R-KV`, `FoldGRPO`) crops `DynamicLayer` attention keys/values while leaving recurrent states $S_b$ polluted by dead-end tokens, causing cross-layer State-KV desynchronization.

## Experimental Arms (Strict Total-Token / FLOPs Accounting)
| Arm | Recurrent State | Attention KV | Prefix `[0, t_a)` | Purpose |
|---|---|---|---|---|
| `clean` | $S_a$ (unpolluted) | $\text{KV}_a$ | Kept | Upper-bound reference ($L=0$) |
| `unrolled` | $S_b$ (polluted) | $\text{KV}_b$ | Kept | Standard uninterrupted CoT |
| `attn_kv_only` | $S_b$ (polluted) | Cropped to $\text{KV}_a$ | Kept | Tests cross-layer State-KV desynchronization |
| `prompt_restart` | Reset to $0$ | Reset to $\emptyset$ | Discarded (re-prefill prompt + lemma) | Dumbest strong baseline #1 |
| `full_replay` | Recomputed $0 \to t_a$ | Recomputed $0 \to t_a$ | Re-prefilled ($C=0$) | Dumbest strong baseline #2 |
| `ring_fifo` | Restored if in last $C$ steps | Cropped + replayed | Replayed from FIFO hit or $0$ | `ReplaySSM`-style sliding window baseline |
| `sync_binomial` | Restored from $C$-slot online pool + short chunk replay | Synchronously cropped + replayed | Preserved | Proposed method |

Every arm logs `total_tokens_processed = initial_prefill + dead_end_tokens + replay_tokens + lemma_tokens + decode_tokens`.

## Quickstart
```bash
pytest -q   # CPU exactness tests across Qwen3.5, Falcon-H1, Zamba2, Granite-4.0-H
python scripts/run_kill_c1.py run --model Qwen/Qwen3.5-0.8B --dataset gsm8k --n 100
python scripts/run_kill_c1.py run --model tiiuae/Falcon-H1R-7B --dataset math500 --n 100
python scripts/run_kill_c1.py report
```

