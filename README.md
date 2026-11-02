# rhyme-plan-hybrid

Question: does a small hybrid linear-attention LM (Qwen3.5-0.8B: 18 Gated-DeltaNet layers + 6 full-attention
layers) carry a rhyme "plan" at the end of line 1, and is the rhyme constraint for line 2 carried by the
GDN recurrent state or by the full-attention KV cache? Control model: Qwen2.5-0.5B-Instruct (pure attention).

## Prior art (checked 2026-11-02, before any experiment)

| Work | What it did | Delta to this project |
|---|---|---|
| Lindsey et al. 2025, "On the Biology of an LLM" | CLT features at the newline plan the rhyme in Claude 3.5 Haiku | closed model, transformer only |
| Ma & Rui 2026, arXiv:2605.07984 "Where's the Plan?" | probes + activation patching at the line boundary in Qwen3/Gemma-3/Llama-3 (10+ scales). Rhyme is linearly decodable at the newline, but only Gemma-3-27B causally uses it; all smaller models read the rhyme word directly | **closest work; same method (probe + newline patching)**. No hybrid/recurrent model, no state-vs-KV locus |
| arXiv:2609.18440 "Planning or Improvisation?" | 0.6B-2.6B open models + open CLTs; newline residual patching changes the rhyme in 11/1260 lines vs 4 baseline | predicts Q1 = NO at our scale; transformers only |
| Hanna & Ameisen 2026, arXiv:2604.12493 "Latent Planning Emerges with Scale" | Qwen3 0.6B-14B, a/an and rhyme planning, planning grows with scale | Qwen3 (pure attention); a/an task already covered, so not repeated here |
| Maar et al., ICLR 2026, arXiv:2601.20164 | steering at end of line changes rhyme; planning "from 1B params" | steering method; transformers only |

Honest position: Q1 (newline planning at <1B) has been tested on transformers and is mostly negative. The only
new part is Q2: a hybrid model has two memory pathways (GDN recurrent state vs attention KV), and nobody has
measured which one carries the rhyme constraint into line 2. I found no prior work on rhyme/planning
interpretability in Gated-DeltaNet or other hybrid models. Prior expectation: Q1 = NO.

## Design

Minimal pairs: one line-1 template, two end words from different rhyme classes (all single tokens, line ends
`word\n`, no comma) -> prompts of identical token length. Few-shot couplet prefix. Greedy line 2 (<=24 tokens,
stop at newline). Last word of line 2 is scored with CMUdict rhyming part (`pronouncing`).

Eligible pair (target T, source S): clean line 2 of T rhymes with T's word AND clean line 2 of S rhymes with S's word.
"Flip" = patched T run ends line 2 with a word rhyming with S's word (and not with T's word).

Conditions (patch S into T):
- `NL`: residual stream (every decoder layer output) at the final `\n` position (Anthropic-style newline site).
- `STATE` (hybrid only): all 18 GDN caches (recurrent + conv) after the prompt.
- `STATE_rec` (hybrid only): GDN recurrent state only.
- `KV`: all full-attention layers' KV cache (for Qwen2.5 this equals the whole model state = `BOTH`).
- `BOTH` (hybrid): STATE + KV; must reproduce S (sanity, flip >= 90% or the code is wrong).
- Controls: `*_rand` = for each patched tensor, T + random Gaussian direction with the same norm as (S - T).

Probe (secondary, weak evidence): logistic regression on the residual stream at `\n` (each layer) -> rhyme
class of line 1's word, grouped CV holding out words; shuffled-label control.

Pilot (allowed before main run, may only choose the few-shot prompt and Qwen3.5-0.8B vs -Base): pick the
variant/prompt with the higher clean rhyme rate on 60 lines. Gate below is not affected by the pilot.

## PRE-REGISTERED GATE (frozen 2026-11-02, before the main experiment; never edited afterwards)

N >= 200 eligible pairs per model (if fewer exist, report N and note the CI widths). 95% Wilson CIs.

- **Q1 Planning (per model) = YES** iff `NL` flip rate >= 30% AND `NL_rand` flip rate <= 5% AND the Wilson CIs
  are disjoint. NO iff `NL` flip < 10% (or CIs overlap with control). Otherwise WEAK.
- **Q2 Locus (Qwen3.5 only)**: compare `STATE` vs `KV` flip rates. Each must beat its own `_rand` control
  (disjoint CIs) to count; a non-beating condition counts as 0.
  - Locus = STATE iff STATE >= 2x KV and CIs disjoint; Locus = KV iff KV >= 2x STATE and CIs disjoint;
    otherwise SHARED.
  - Interpretation rule fixed in advance: if Q1 = NO, Q2 is reported as "which memory pathway carries the
    rhyme-word information into line 2", NOT as "where the plan lives".
- **Validity check**: hybrid `BOTH` flip >= 90%; a no-op patch must leave logits unchanged (pytest). If either
  fails, the run is INVALID, not a result.
- **Overall verdict**: PASS iff Q1 = YES for Qwen3.5-0.8B and Q2 is STATE or KV. FAIL iff Q1 = NO for
  Qwen3.5-0.8B (Q2 reported anyway). MIXED otherwise.

## Files
- `rp.py` - data, rhyme scoring, generation with cache/residual patching.
- `run.py` - main experiment (writes `results/*.json`).
- `test_rp.py` - pytest: no-op patches leave logits unchanged; full-state patch reproduces the source.
- `RESULTS.md` - numbers and verdict.
- `logit.py` - secondary teacher-forced analysis; `run_all.py` - whole campaign + gate analysis (`verdict.json`); `CAMPAIGN.md` - Colab plan.

## Pilot outcome + SECONDARY pre-registered analysis (added 2026-11-02 after the pilot, before any main run; primary gate above untouched)

Pilot (60 lines, greedy): Qwen3.5-0.8B 9/60 lines end in a CMUdict rhyme, Qwen3.5-0.8B-Base 5/60, Qwen2.5-0.5B-Instruct 1/60.
Most Qwen3.5 "rhymes" are copies of the line-1 word (11/60 copies vs 2/60 genuine rhymes in a recount that
excludes identical words). A longer 8-shot prompt with an explicit instruction did not raise genuine rhymes (2/60).
Per the pilot rule: Qwen3.5-0.8B (post-trained) with the 3-shot prompt in `rp.py`. The primary gate is run exactly
as written; results additionally report `flip_nocopy` (flip where line 2 does not just copy the source word).

Because free generation rarely rhymes, a secondary, teacher-forced logit analysis is added (`logit.py`):
- After line 1 + `\n`, a fixed line-2 body is teacher-forced (3 bodies ending in "the"), and at the final slot
  LD = logsumexp(logp of first tokens of T-class words) - logsumexp(S-class words), with the two line-1 words
  themselves EXCLUDED (so copying does not count).
- Same patch conditions as the primary (patch applied to memory up to the line-1 word, or to the `\n` residual).
- Fraction restored per pair: f = (LD_T,clean - LD_patched) / (LD_T,clean - LD_S,clean); pairs with
  |LD_T,clean - LD_S,clean| < 0.5 nats are dropped (no rhyme signal to move). Mean f with 95% bootstrap CI.
- Secondary gate: newline planning = YES iff NL f >= 0.30 with CI disjoint from NL_rand; NO iff NL f < 0.10.
  Locus = STATE iff STATE f >= 2x KV f with disjoint CIs (and each beats its _rand); KV symmetrically; else SHARED.
- This secondary result can not turn the primary FAIL into a PASS; it is reported separately.

## 2026-11-02: runs moved to Colab GPU (before any main result exists; gates above untouched)

- The Mac runs of `run.py` / `logit.py` (Qwen3.5-0.8B, then Qwen2.5-0.5B-Instruct) were killed before finishing.
  Their partial logs (`results/*.log`, local) are discarded as evidence; no main-run counts were ever produced.
- The full campaign now runs unattended on a Colab L4 via `run_all.py` / `notebooks/colab_runner_rhyme_plan_hybrid.ipynb`
  (see `CAMPAIGN.md`). Gate, item counts (N=200 pairs per stage, 3 bodies), conditions and seeds of the pair sampling
  are unchanged.
- Numerical settings: dtype stays **fp32** for all runs and tests (bf16 not adopted: the workload is latency-bound at
  0.8B, and bf16 would change greedy outputs relative to the fp32 pilot). On CUDA the GDN layers use the
  flash-linear-attention (Triton) and, if it builds, causal-conv1d kernels instead of the torch reference path; these
  differ from the Mac path only at float-rounding level (Triton dots may use TF32). Patching is checked on exactly this
  path: `test_rp.py` (no-op patch leaves logits unchanged; BOTH patch reproduces the source; patch sites = 18 GDN + 6
  attention layers; now also run for Qwen2.5) runs on the GPU before the main run and the campaign aborts if it fails.
- Engineering changes with no effect on the gate: results are written incrementally (`*.pairs.jsonl`,
  `*_logit.jsonl`) and finished batches are skipped on restart; pairs are batched per template in fixed batches, and
  the `_rand` controls draw from a per-batch seed (1000 + batch index) instead of one global stream, so a resumed run
  is identical to an uninterrupted one.
