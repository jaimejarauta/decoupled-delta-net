# wb-witness-verify

Kill experiment and training/evaluation library for **Backup #1 (C2)** in `research/RESEARCH_DECISION_2026-10-26.md`:
*Training Language Models to Emit Self-Contained Verification Witnesses Under Derivation Erasure (`WB-GRPO`)*.

## Core Design Invariants
1. **Zero Hidden-State Leakage:** Unlike intra-sequence attention masking (where `<witness>` KV projections retain hidden-state features from the flawed derivation $D_1$), `WB-GRPO` decodes $(W_1, A_1)$ to discrete token IDs (`|W_1| <= 64`) and re-prefills `[Prompt P] + [W_1, A_1]` in a physically separate clean context: $I(D_1; V_1 \mid P, W_1, A_1) = 0$.
2. **Anti-Rubber-Stamp Class-Balanced Reward:** Verification reward dynamically normalizes positive (`ACCEPT | correct`) and negative (`REJECT | wrong`) weights $\alpha_+, \alpha_-$ so a solver with 85% base accuracy cannot game reward by outputting `ACCEPT` 100% of the time; primary metric is **Macro-F1**, never raw verification accuracy.
3. **Anti-Steganography Cross-Model Transfer Gate:** Emitted witnesses $W_1$ are tested inside a frozen external model (`--external-verifier`) that was never co-trained with `WB-GRPO`.

## Quickstart
```bash
pytest -q
python scripts/run_kill_c2.py eval --model Qwen/Qwen3-8B --external-verifier meta-llama/Llama-3.1-8B-Instruct --n 200
python scripts/run_kill_c2.py train --model Qwen/Qwen3-4B --mode wb_grpo --steps 200
python scripts/run_kill_c2.py report
```

