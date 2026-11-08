# CAMPAIGN - injection-latent-intent (Colab L4, one "Run all")

Notebook: `notebooks/colab_runner_injection_latent_intent.ipynb` -> `run_all.py`. Results: Drive
`MyDrive/moonshot_results/injection-latent-intent/` (`$IL_RESULTS`); HF cache on Drive (`HF_HOME`), so model downloads
happen once. Re-running the notebook resumes: every generation stage is a per-item JSONL cache, the DPO adapter and
feature files are reused, finished models (`summary.json`) are skipped. Gate: README.md "PRE-REGISTERED GATE".

GPU: **L4 24 GB** (bf16). An A100 is not needed: largest model is 8B bf16 (16 GB) for inference only; DPO is on <=3B.

## Stages and estimated L4 time

| # | stage | work | est. L4 time |
|---|---|---|---|
| 0 | installs + downloads (Qwen3-1.7B 3.4 GB, Llama-3.2-3B 6.4 GB, Llama-3.1-8B 16 GB, InjecAgent 3 MB, GSM8K) | - | 10-15 min (first run) |
| 1 | pytest pre-flight, IL_REAL=1 (tiny-model suite + real-model feature/prefill checks for both primary models) | 12 tests | 3-5 min |
| 2 | pilot = Qwen3-1.7B full pipeline, k=16 sampling on 64 test items; early-abandon rule | und greedy 2,108 gens, DPO 1,674 micro-steps, def greedy 992, prefill 2x496, features, GSM8K 2x200 | ~35 min |
| 3 | Qwen3-1.7B main: remaining 432 items x 16 samples, probe + summary | 6,912 sampled gens | ~15 min |
| 4 | Llama-3.2-3B-Instruct: everything | as 2+3 at ~1.6x cost | ~80 min |
| 5 | secondary: Llama-3.1-8B-Instruct vs Meta-SecAlign-8B (no DPO; 248 items x 16 samples) | und greedy 2,108, def greedy 992, 3,968 samples, GSM8K | ~70 min |
| 6 | gate analysis | CPU | <1 min |

Estimates are from token counts (prompts 650-750 tokens mean, max 1.2k; outputs <=128 tokens, tool calls ~30) and
typical HF `generate` throughput on an L4, not measured; the pilot log gives the real per-stage times.
**Total ~3.5-4 h L4 = 13-15 CU** (L4 ~3.8 CU/h). If the pilot shows the main stages running over budget, skip stage 5
with `IL_SKIP_SECONDARY=1` (reported as skipped; it is not in the gate): ~2.3 h = ~9 CU. Abandon at pilot: ~0.9 h = ~3.5 CU.

## Decision tree

- **INVALID**: do not interpret. V1 fail (undefended ASR < 10%) -> use only the `enhanced` InjecAgent sets or a more
  injection-prone model, with a new dated gate. V3 fail (defense did not take) -> the recipe is too weak at 558 pairs;
  new dated gate with more pairs (add AgentDojo-style or Alpaca-injection pairs), never retune on test. V2 fail -> probe
  has no signal on the undefended model; the question cannot be asked with a linear probe at this layer set.
- **PASS** (both models: probe persists, reachable): replicate before any claim: (1) the Meta-SecAlign arm in the same
  direction; (2) a second defense recipe (SecPO-style / full fine-tune instead of LoRA, higher rank) to rule out "LoRA
  barely moved the residual"; (3) AgentDojo slice; (4) probe-as-auditor: AUROC of the frozen probe vs defended
  reachability, item level, as the deliverable.
- **MIXED**: report which half holds. H1 without H2 = representation persists but behaviour is robust (probe not useful
  as an auditor). H2 without H1 = shallow behaviour change with moved representation (the BCA story without the probe).
- **FAIL**: probe tracks ASR. If "intent erased" (D_PPR >= 0.75): evidence that LoRA-DPO defenses change the decision
  representation, publishable as a negative for "refusal not robustness" in injection; check per-layer D_PPR for a layer
  where it survives (secondary, needs a new gate). Otherwise close the project.

## Scale-up path (A6000 48 GB / Blackwell)

Same code. 8B DPO arms (Llama-3.1-8B, Qwen3-8B) with the own recipe and Meta-SecAlign-8B as the public reference;
Qwen3.6-27B-Instruct-SecPO (arXiv:2610.41998) vs its base in 4-bit for the strongest public defense; AgentDojo tasks
with full agent loops; exact BCA trees (top-b branching over the first tokens) instead of sampling.
