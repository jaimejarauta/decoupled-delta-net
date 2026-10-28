# zero-state-vlm-cache

Kill experiment for **State-Free Modular Image Caching in Hybrid Vision-Language Models** (`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B` vs. `Qwen/Qwen3-VL-2B-Instruct`).

## Core Hypothesis
In 3:1 Gated DeltaNet + Full-Attention VLMs (`Qwen3.5`), snapshotting the 18–24 Gated DeltaNet recurrent states per cached 256-token image costs **$6.3\times\text{--}7.0\times$ more memory** ($25.2\text{ MB}$ vs. $4.0\text{ MB}$ in `4B`) than the entire full-attention KV cache while restricting cache hits to exact sequential prefixes (`Marconi`). We test whether zeroing Gated DeltaNet state at `<|vision_start|>` / `<|vision_end|>` and caching **only** unrotated pre-`mrope` attention KV tensors ($0$ bytes of recurrent state, dynamically rotated to target 3D `grid_thw` offsets at query time) preserves $\ge 98.5\%$ accuracy while enabling position-independent multi-image composition at $\ge 4.5\times$ faster TTFT.

## Experimental Arms
| Arm | Recurrent State | Attention KV | Role |
| :--- | :--- | :--- | :--- |
| `full_joint` | Full sequential | Full sequential | Unmodified ceiling (`Marconi` memory reference) |
| `gdn_boundary_reset` | Zeroed at `<|vision_start\|>` & `<|vision_end\|>` | Full cross-image | Pure causal test of cross-boundary GDN visual state |
| `zero_state_modular` | $0$ bytes stored (`S=0` per image & at query) | Pre-`mrope` KV cached independently per image | Proposed $7\times$ smaller modular VLM cache |
| `pooled_joint_2x2` | Full sequential ($2\times 2$ downsampled image) | Full sequential ($4\times$ fewer tokens) | Dumbest strong baseline at matched TTFT |

## Quickstart
```bash
pytest -q
python scripts/run_kill.py run --model Qwen/Qwen3.5-2B --tasks docvqa chartqa mp_docvqa muirbench --n 200
python scripts/run_kill.py report
```

