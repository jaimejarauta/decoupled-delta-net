# CAMPAIGN - rhyme-plan-hybrid (Colab L4, one "Run all")

Notebook: `notebooks/colab_runner_rhyme_plan_hybrid.ipynb` -> `run_all.py`. Results: Drive
`MyDrive/moonshot_results/rhyme-plan-hybrid/` (`$RP_RESULTS`). Re-running the notebook resumes (finished stages and
batches are skipped). Gate: README.md "PRE-REGISTERED GATE" (primary) and "SECONDARY pre-registered analysis".

## Stages

| # | stage | items | forward passes (approx.) | outputs |
|---|---|---|---|---|
| 0 | installs (FLA, causal-conv1d build capped at 25 min) | - | - | - |
| 1 | pytest on GPU, fp32 (abort if fails; one retry on torch path) | 2 models x 4 checks | <100 | `pytest.json` |
| 2 | `run.py Qwen/Qwen3.5-0.8B`: clean greedy line 2 for 670 lines | 30 batches (<=24) | ~750 | `Qwen3.5-0.8B_clean.json`, `_resid.npy` |
| 2b | patching, 200 eligible pairs x 9 conditions | ~20 batches of 16 | ~4,600 | `Qwen3.5-0.8B.pairs.jsonl`, `.json` |
| 2c | probe: 24 layers x 2 x 5-fold logistic regression (CPU) | 240 fits | - | in `.json` |
| 3 | `logit.py Qwen/Qwen3.5-0.8B`: 200 pairs x 3 bodies x 11 conditions | ~20 batches | ~1,500 | `Qwen3.5-0.8B_logit.jsonl`, `_logit.json` |
| 4 | `run.py Qwen/Qwen2.5-0.5B-Instruct` (pilot: ~1/60 lines rhyme, so expect N << 200 pairs) | 30 + few batches | ~1,000 | `Qwen2.5-0.5B-Instruct*.json(l)` |
| 5 | `logit.py Qwen/Qwen2.5-0.5B-Instruct`, 200 pairs x 3 bodies x 4 conditions | ~20 batches | ~800 | `..._logit.json(l)` |
| 6 | gate analysis | - | - | `verdict.json`, `RESULTS.md` |

## Estimated L4 time

Measured locally (Mac MPS, fp32, torch GDN path, heavily loaded machine): Qwen3.5 one pair through all 9 conditions
= 5.5 s (~225 forward passes, ~25 ms each at batch 1); logit.py one pair x 3 bodies = 8.7 s; Qwen2.5 = 2-3 s per pair.
The workload is launch/latency bound (0.8B, short prompts), so a batch of 16 on the L4 costs roughly what one pair costs
on MPS: ~20-50 ms per forward -> stage 2+2b ~3-5 min, stage 3 ~1-2 min, stages 4-5 ~2 min, probe ~2-5 min (CPU).

**Total: ~15 min compute, ~25-45 min wall clock including model downloads and installs (the causal-conv1d source
build, if no prebuilt wheel matches, dominates; capped at 25 min).** Budget 1 hour of L4. A torch-path fallback
(no kernels) at most doubles the compute part.

## Gate (summary; README text is authoritative and quoted verbatim in RESULTS.md)

- Validity: pytest no-op / full-patch checks pass on the run device AND hybrid `BOTH` flip >= 90%; else INVALID.
- Q1 per model: YES iff `NL` >= 30%, `NL_rand` <= 5%, Wilson CIs disjoint; NO iff `NL` < 10% or CIs overlap; else WEAK.
- Q2 (Qwen3.5): STATE vs KV flip, each must beat its `_rand`; 2x + disjoint CIs -> STATE / KV; else SHARED.
- Overall: PASS iff Q1(Qwen3.5) = YES and Q2 in {STATE, KV}; FAIL iff Q1(Qwen3.5) = NO; else MIXED.
- Secondary (reported separately, cannot rescue a FAIL): fraction-restored f, NL f >= 0.30 with CI disjoint from
  NL_rand = planning YES, < 0.10 = NO; locus STATE/KV/SHARED by the same 2x rule on f.

## Decision tree

- **INVALID**: do not interpret anything. Check `pytest.json`, `verdict.json.validity`, `kernels`. If BOTH < 90% on the
  kernel path but pytest passes, suspect batch-composition numerics in greedy decoding; re-run with
  `RP_RESULTS` pointing to a fresh dir after fixing (README dated note before re-running).
- **PASS** (newline planning at 0.8B + clear locus): surprising vs prior art (2609.18440, Ma & Rui 2026). Replicate before
  any claim: Qwen3.5-0.8B-Base, a second hybrid (Falcon-H1-0.5B/1.5B), a second prompt set; then write up the locus
  finding (state vs KV) as the main contribution.
- **MIXED** (Q1 YES but SHARED locus, or Q1 WEAK): the hybrid plans weakly / both pathways carry it. Next: per-layer
  patching (which GDN layers / which attention layers) on the same pairs; scale to Qwen3.5-2B to see if Q1 sharpens.
- **FAIL** (Q1 = NO at 0.8B, the prior expectation): report Q2 as "which memory pathway carries the rhyme-word
  information into line 2" (pre-registered wording), plus the secondary logit result. Natural follow-ups, each with its
  own pre-registered gate before running:
  1. Larger hybrids: Qwen3.5-2B, Qwen3.5-4B (same code, `hybrid = "3.5" in name`; check the 18/6 layer count test,
     which is 0.8B-specific, and memory: 4B fp32 = 16 GB, fits the L4).
  2. Other hybrid families: Falcon-H1 (Mamba-2 + attention; needs `cache_tensors` for its SSM cache layout and a new
     patch-site test).
  3. If the generation rhyme rate stays too low for N >= 200 eligible pairs, rely on the teacher-forced secondary
     analysis as the main lens at larger scale (pre-register it as primary for the new run).
- Qwen2.5 control (any verdict): expected Q1 = NO and N << 200 pairs (pilot 1/60 rhyme rate); report N and CI widths.
