# hybrid-locality

Kill experiment for P1 in `research_scout_2026-10-26.md`: in released hybrids (linear/SSM layers + full attention),
does limiting every linear layer's memory to the last W tokens hurt long-context performance?

## Arms (per window W ∈ {64, 256, 1024, 4096})
| arm | what changes | role |
|---|---|---|
| `full` | nothing | reference |
| `lin@W` | every GDN/Mamba-2 mixer sees only its last W/2..W inputs (staggered restarts, exact) | the hypothesis |
| `attn@W` | every attention layer sees only keys within W | which path carries long range |
| `trunc@W` | the whole model sees only the last W/2..W tokens | dumbest strong baseline + validity control |

Tasks: needle (single, 4-key) and a 4-hop variable chain, scored by teacher-forced exact match (equals greedy
decoding of the answer); every needle is ≥ 4096 tokens before the question. LongPPL key-token NLL on PG19.

## Pre-registered gate (W=1024, `verdict` in `scripts/run_kill.py`)
A model counts only if Full ≥ 50% and Trunc drops ≥ 20 pts. **H1** if ≥ 3 valid models have lin drop ≤ 2 pts,
key-NLL rise ≤ 3% and attn drop ≥ 20 pts. **H0** if ≥ 3 have lin drop ≥ 10 pts. Otherwise MIXED / INVALID.

## Run (A6000)
```bash
pip install -e ".[fast,dev]"      # fla / causal-conv1d / mamba-ssm kernels; without them it is correct but slow
pytest -q                          # CPU checks: exactness, locality, no-op at W>=L, logit scaling, gate logic
for m in Qwen/Qwen3.5-0.8B tiiuae/Falcon-H1-0.5B-Base tiiuae/Falcon-H1-1.5B-Base Zyphra/Zamba2-1.2B ibm-granite/granite-4.0-h-micro; do
  python scripts/run_kill.py run --model $m
done
python scripts/run_kill.py report  # drops with paired-bootstrap 95% CIs + verdict
```
Lengths above a model's `max_position_embeddings` are skipped. Results append to `results/kill.jsonl`.
