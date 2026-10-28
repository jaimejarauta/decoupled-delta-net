"""Long-context retrieval, 2-turn query-shift, real LongBench-v2 QA, and CUDA-synced throughput evaluators."""
from __future__ import annotations

from collections import Counter
import random
import re
import time
from typing import Any

import torch


def load_pg19_haystack(min_chars: int = 300_000) -> str:
    """Stream real book chapters from Project Gutenberg (`emozilla/pg19-test`) for haystack construction."""
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    chunks: list[str] = []
    total = 0
    for row in ds:
        txt = " ".join(str(row["text"]).split())
        if len(txt) > 1000:
            chunks.append(txt)
            total += len(txt)
        if total >= min_chars:
            break
    return " ".join(chunks)


def _insert_at_word_depths(haystack_words: list[str], insertions: list[tuple[float, str]]) -> str:
    """Insert `(depth_fraction, snippet)` pairs into `haystack_words` without clobbering order."""
    out = list(haystack_words)
    # Insert from deepest to shallowest so earlier indices stay exact
    for depth, snippet in sorted(insertions, key=lambda x: x[0], reverse=True):
        pos = max(8, min(len(out) - 8, int(round(depth * len(out)))))
        out.insert(pos, snippet)
    return " ".join(out)


def build_ruler_and_query_shift_items(
    tokenizer: Any,
    task: str,
    n: int,
    target_length: int = 8192,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Construct real-text (`pg19-test`) long-context evaluation items with exact prefill/post-prefill splits.

    Every item returns:
      - `doc_ids`: Tensor [1, L_doc] prefilled ONCE and compressed by all arms.
      - `post_ids`: Tensor [1, L_post] fed AFTER KV compression (e.g. Turn-2 shifted query or answer trigger).
      - `gold`: exact ground-truth target string.
      - `metric`: 'digits' | 'mc'
    """
    rng = random.Random(seed + hash(task) % 10_007)
    raw_text = load_pg19_haystack(min_chars=max(400_000, target_length * 32))
    all_words = raw_text.split()
    # ~0.72 words per subword token; over-provision slightly then slice tokens cleanly preserving needles
    est_words = max(256, int(target_length * 0.72))
    depths = [0.10, 0.28, 0.48, 0.68, 0.85]
    items: list[dict[str, Any]] = []

    for i in range(n):
        max_start = max(1, len(all_words) - est_words - 100)
        start_w = rng.randint(0, max_start)
        base_words = all_words[start_w : start_w + est_words]
        depth_primary = depths[i % len(depths)]

        if task == "ruler_single":
            code = str(rng.randint(100_000, 999_999))
            needle = f" [IMPORTANT RECORD: The vault passkey for Project Meridian is {code}.] "
            doc_str = _insert_at_word_depths(base_words, [(depth_primary, needle)])
            turn1_q = "\n\nQuestion: What is the 6-digit vault passkey for Project Meridian?"
            post_str = "\nAnswer (6 digits only):"
            gold = code

        elif task == "ruler_multikey":
            keys = [" Sapphire", " Obsidian", " Titanium", " Cobalt"]
            rng.shuffle(keys)
            codes = [str(rng.randint(100_000, 999_999)) for _ in range(4)]
            ins = [
                (0.14, f" [REGISTRY: Sector{keys[0]} authorization code is {codes[0]}.] "),
                (0.36, f" [REGISTRY: Sector{keys[1]} authorization code is {codes[1]}.] "),
                (0.58, f" [REGISTRY: Sector{keys[2]} authorization code is {codes[2]}.] "),
                (0.80, f" [REGISTRY: Sector{keys[3]} authorization code is {codes[3]}.] "),
            ]
            target_idx = i % 4
            doc_str = _insert_at_word_depths(base_words, ins)
            turn1_q = f"\n\nQuestion: What is the 6-digit authorization code for Sector{keys[target_idx]}?"
            post_str = "\nAnswer (6 digits only):"
            gold = codes[target_idx]

        elif task == "ruler_vt":
            val = str(rng.randint(100_000, 999_999))
            v_names = ["X1", "X2", "X3", "X4"]
            ins = [
                (0.12, f" [STATE LOG: Variable {v_names[0]} = {val}.] "),
                (0.35, f" [STATE LOG: Variable {v_names[1]} = {v_names[0]}.] "),
                (0.60, f" [STATE LOG: Variable {v_names[2]} = {v_names[1]}.] "),
                (0.82, f" [STATE LOG: Variable {v_names[3]} = {v_names[2]}.] "),
            ]
            doc_str = _insert_at_word_depths(base_words, ins)
            turn1_q = f"\n\nQuestion: Trace the variable assignments and give the numeric value of {v_names[3]}."
            post_str = "\nAnswer (6 digits only):"
            gold = val

        elif task == "multi_turn_shift":
            # Turn 1 (inside prefill observation window!) asks about Key Alpha -> SnapKV keeps Key Alpha.
            # Turn 2 (arrives AFTER cache eviction!) shifts to Key Beta at an earlier depth!
            code_alpha = str(rng.randint(100_000, 999_999))
            code_beta = str(rng.randint(100_000, 999_999))
            depth_beta = [0.12, 0.25, 0.42, 0.58][i % 4]
            depth_alpha = 0.78 if abs(depth_beta - 0.78) > 0.12 else 0.32
            ins = [
                (depth_beta, f" [ARCHIVE ENTRY: The security pin for Protocol Beta is {code_beta}.] "),
                (depth_alpha, f" [ARCHIVE ENTRY: The security pin for Protocol Alpha is {code_alpha}.] "),
            ]
            doc_str = _insert_at_word_depths(base_words, ins)
            # Prefill ends with Turn 1 Q&A on Protocol Alpha (biasing observation-window eviction!)
            turn1_q = (
                "\n\nUser Turn 1: What is the 6-digit security pin for Protocol Alpha?"
                f"\nAssistant Turn 1: {code_alpha}"
            )
            post_str = (
                "\nUser Turn 2: Now what is the 6-digit security pin for Protocol Beta?\nAssistant Turn 2:"
            )
            gold = code_beta
        else:
            raise ValueError(f"Unknown RULER task: {task}")

        body_ids = tokenizer.encode(doc_str, add_special_tokens=True)
        q1_ids = tokenizer.encode(turn1_q, add_special_tokens=False)
        post_ids = tokenizer.encode(post_str, add_special_tokens=False)

        max_body = max(64, target_length - len(q1_ids))
        if len(body_ids) > max_body:
            # Trim from the very end of the haystack only if needed; check gold still inside body_ids
            cand = body_ids[:max_body]
            if gold in tokenizer.decode(cand):
                body_ids = cand

        doc_ids = torch.tensor([body_ids + q1_ids], dtype=torch.long)
        post_tensor = torch.tensor([post_ids], dtype=torch.long)
        items.append(
            {
                "task": task,
                "doc_ids": doc_ids,
                "post_ids": post_tensor,
                "gold": gold,
                "metric": "digits",
                "depth": depth_primary,
            }
        )
    return items


def load_longbench_v2_items(
    tokenizer: Any,
    n: int,
    target_length: int = 8192,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Stream real multi-domain long-context QA items from Parquet-backed `THUDM/LongBench-v2`."""
    from datasets import load_dataset

    ds = None
    for repo_id in ("THUDM/LongBench-v2", "zai-org/LongBench-v2"):
        try:
            ds = load_dataset(repo_id, split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        return []

    items: list[dict[str, Any]] = []
    for row in ds:
        if len(items) >= n:
            break
        ctx = str(row.get("context", ""))
        q = str(row.get("question", ""))
        ans = str(row.get("answer", "")).strip().upper()[:1]
        if ans not in ("A", "B", "C", "D") or not ctx:
            continue
        choices = (
            f"(A) {row.get('choice_A', '')}\n"
            f"(B) {row.get('choice_B', '')}\n"
            f"(C) {row.get('choice_C', '')}\n"
            f"(D) {row.get('choice_D', '')}"
        )
        q_block = f"\n\nQuestion: {q}\n{choices}"
        post_str = "\nThe single correct option letter (A, B, C, or D) is:"

        ctx_ids = tokenizer.encode(ctx, add_special_tokens=True)
        q_ids = tokenizer.encode(q_block, add_special_tokens=False)
        post_ids = tokenizer.encode(post_str, add_special_tokens=False)

        # Keep natural multi-thousand-token items; middle-truncate ultra-long >target_length docs preserving head+tail
        max_ctx = max(256, target_length - len(q_ids))
        if len(ctx_ids) < min(1024, max_ctx // 2):
            continue
        if len(ctx_ids) > max_ctx:
            half = max_ctx // 2
            ctx_ids = ctx_ids[:half] + ctx_ids[-(max_ctx - half) :]

        items.append(
            {
                "task": "longbench_v2",
                "doc_ids": torch.tensor([ctx_ids + q_ids], dtype=torch.long),
                "post_ids": torch.tensor([post_ids], dtype=torch.long),
                "gold": ans,
                "metric": "mc",
                "depth": 0.5,
            }
        )
    return items


def score_exact_digits(pred_text: str, gold_digits: str) -> float:
    cleaned = pred_text.strip().split("\n")[0]
    if gold_digits in cleaned.replace(",", "").replace(" ", ""):
        return 1.0
    m = re.search(r"\d{4,8}", cleaned.replace(",", ""))
    return 1.0 if (m and m.group(0) == gold_digits) else 0.0


def score_mc_letter(pred_text: str, gold_letter: str) -> float:
    cleaned = pred_text.strip().upper()
    g = gold_letter.strip().upper()[:1]
    m = re.match(r"^[\(\[\s]*([A-D])(?:\b|\)|\]|\.|:|\s|$)", cleaned)
    if m:
        return 1.0 if m.group(1) == g else 0.0
    m2 = re.search(r"\b([A-D])\b", cleaned[:32])
    return 1.0 if (m2 and m2.group(1) == g) else 0.0


def score_token_f1(pred_text: str, gold_text: str) -> float:
    p_toks = re.findall(r"\w+", pred_text.lower())
    g_toks = re.findall(r"\w+", gold_text.lower())
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec = common / len(g_toks)
    return 2.0 * prec * rec / (prec + rec)


def score_item(metric: str, pred_text: str, gold: str) -> float:
    if metric == "digits":
        return score_exact_digits(pred_text, gold)
    if metric == "mc":
        return score_mc_letter(pred_text, gold)
    if metric == "f1":
        return score_token_f1(pred_text, gold)
    raise ValueError(f"Unknown metric: {metric}")


def paired_bootstrap_ci(
    ref_scores: list[float],
    arm_scores: list[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI of `(ref - arm)` accuracy drop in percentage points."""
    assert len(ref_scores) == len(arm_scores) and len(ref_scores) > 0
    g = torch.Generator().manual_seed(seed)
    diffs = (torch.tensor(ref_scores, dtype=torch.float32) - torch.tensor(arm_scores, dtype=torch.float32)) * 100.0
    idx = torch.randint(len(diffs), (n_boot, len(diffs)), generator=g)
    means = diffs[idx].mean(dim=1)
    return float(diffs.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


@torch.no_grad()
def measure_decode_tok_per_s(
    forward_step_fn: Any,
    n_steps: int = 32,
    device: str = "cuda",
) -> float:
    """Measure synchronized steady-state autoregressive decode tokens/sec."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_steps):
        forward_step_fn(step)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = max(1e-6, time.perf_counter() - t0)
    return float(n_steps / dt)

