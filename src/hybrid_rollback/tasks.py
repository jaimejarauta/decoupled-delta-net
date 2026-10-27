"""Real math benchmark loading (`GSM8K`, `MATH-500`) + controlled Same-Subspace vs. Orthogonal
Dead-End builders, contradiction trigger detection, and symbolic/numeric answer grading."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

BACKTRACK_PATTERNS = re.compile(
    r"(?:\bWait,|\bActually,\s+that\s+contradicts|\bHold\s+on,|\bThat\s+cannot\s+be\s+right|\bLet's\s+recalculate)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
HASH_ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
LAST_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


@dataclass
class DeadEndItem:
    idx: int
    prompt_text: str
    valid_prefix_steps: list[str]
    same_subspace_dead_end_unit: str
    orthogonal_haystack_unit: str
    negative_lemma: str
    continuation_cue: str
    ground_truth: str


def extract_answer(text: str) -> str:
    """Extract final boxed or GSM8K #### answer, falling back to normalized final number."""
    boxes = BOXED_RE.findall(text)
    if boxes:
        return _norm_ans(boxes[-1])
    hashes = HASH_ANS_RE.findall(text)
    if hashes:
        return _norm_ans(hashes[-1])
    nums = LAST_NUM_RE.findall(text)
    return _norm_ans(nums[-1]) if nums else ""


def _norm_ans(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
        return f"{v:.6g}"
    except ValueError:
        return s


def answers_match(pred_text: str, gt: str) -> bool:
    return extract_answer(pred_text) == _norm_ans(gt)


def detect_backtrack_trigger(step_text: str) -> bool:
    return bool(BACKTRACK_PATTERNS.search(step_text))


def _extract_entities_and_numbers(text: str) -> tuple[list[str], list[int]]:
    words = re.findall(r"\b[A-Za-z][a-z]{3,10}\b", text)
    stop = {"what", "when", "where", "which", "there", "their", "about", "after", "before", "many", "much", "each", "have", "does", "from", "with", "that", "this"}
    nouns = [w for w in dict.fromkeys(words) if w.lower() not in stop][:4] or ["value", "quantity", "total"]
    nums = [int(n.replace(",", "")) for n in re.findall(r"\b\d{1,5}\b", text)][:4] or [12, 7, 3]
    return nouns, nums


def build_same_subspace_dead_end(prompt: str, partial: str, rng: random.Random) -> tuple[str, str]:
    """Construct an algebraically flawed branch manipulating the *exact* problem variables/numbers,
    paired with a self-contained <=24-token negative lemma (containing zero oracle answer info)."""
    nouns, nums = _extract_entities_and_numbers(prompt + " " + partial)
    v0, v1 = nouns[0], nouns[min(1, len(nouns) - 1)]
    n0, n1 = nums[0], nums[min(1, len(nums) - 1)]
    wrong_mult = rng.choice([3, 5, 7])
    cur = n0 * wrong_mult + n1
    clauses = []
    for step_k in range(1, 9):
        nxt = (cur * 2 - n1) if step_k % 2 else (cur + n0)
        clauses.append(
            f"Sub-branch check {step_k}: suppose {v0} scales by {wrong_mult} relative to {v1}, "
            f"giving intermediate {v0} = {cur} and updated {v1} = {nxt}."
        )
        cur = nxt
    unit = " ".join(clauses) + "\n\n"
    lemma = f"[Note: assuming {v0} scales by {wrong_mult} contradicted the initial {v1} constraint; discard that branch.]\n\n"
    return unit, lemma


ORTHOGONAL_ESSAY_FALLBACK = (
    "Renaissance maritime cartographers calibrated brass astrolabes against polar star declinations "
    "while Venetian glassmakers refinedcristallo silica furnaces along Murano tidal canals. "
    "Botanical treatises cataloged alpine gentian alkaloids, medieval scriptoria ground lapis lazuli "
    "pigments into gum arabic binders, and pelagic albatrosses navigated southern trade winds.\n\n"
)


def load_dead_end_suite(dataset: str = "gsm8k", n: int = 100, seed: int = 0) -> list[DeadEndItem]:
    """Load real `gsm8k` or `math500` items (or `synthetic` offline multi-step algebra for unit tests)."""
    rng = random.Random(seed)
    items: list[DeadEndItem] = []

    if dataset == "synthetic":
        for i in range(n):
            a, b, c = rng.randint(11, 49), rng.randint(5, 29), rng.randint(2, 9)
            s1 = a + b
            s2 = s1 * c
            ans = s2 - a
            prompt = (
                f"Problem {i}: Let alpha = {a}, beta = {b}, and gamma = {c}. "
                f"First compute u = alpha + beta, then v = u * gamma, and finally output w = v - alpha inside \\boxed{{}}.\nSolution:\n\n"
            )
            steps = [f"Step 1: u = alpha + beta = {a} + {b} = {s1}.\n\n", f"Step 2: v = u * gamma = {s1} * {c} = {s2}.\n\n"]
            dead_unit, lemma = build_same_subspace_dead_end(prompt, "".join(steps), rng)
            items.append(
                DeadEndItem(
                    idx=i,
                    prompt_text=prompt,
                    valid_prefix_steps=steps,
                    same_subspace_dead_end_unit=dead_unit,
                    orthogonal_haystack_unit=ORTHOGONAL_ESSAY_FALLBACK,
                    negative_lemma=lemma,
                    continuation_cue="Resuming from Step 2 to compute w = v - alpha: ",
                    ground_truth=str(ans),
                )
            )
        return items

    from datasets import load_dataset

    try:
        books = list(load_dataset("emozilla/pg19-test", split="test")["text"])
        ortho_pool = [b[1000:3500].replace("\n", " ") + "\n\n" for b in books[:20] if len(b) > 4000]
    except Exception:
        ortho_pool = [ORTHOGONAL_ESSAY_FALLBACK]

    if dataset == "gsm8k":
        ds = list(load_dataset("openai/gsm8k", "main", split="test"))
        rng.shuffle(ds)
        for i, row in enumerate(ds[:n]):
            q = row["question"].strip()
            sol_Parts = row["answer"].split("####")
            gt = sol_Parts[-1].strip()
            lines = [ln.strip() for ln in sol_Parts[0].strip().splitlines() if ln.strip()]
            cut = max(1, len(lines) // 2)
            valid_steps = [ln + "\n\n" for ln in lines[:cut]]
            prompt = f"Solve step by step and put your final numerical answer in \\boxed{{}}.\n\nQuestion: {q}\n\nSolution:\n\n"
            dead_unit, lemma = build_same_subspace_dead_end(q, "".join(valid_steps), rng)
            items.append(
                DeadEndItem(
                    idx=i,
                    prompt_text=prompt,
                    valid_prefix_steps=valid_steps,
                    same_subspace_dead_end_unit=dead_unit,
                    orthogonal_haystack_unit=rng.choice(ortho_pool),
                    negative_lemma=lemma,
                    continuation_cue="Continuing from the verified intermediate steps above to finish the calculation:\n",
                    ground_truth=gt,
                )
            )
    elif dataset == "math500":
        ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
        rng.shuffle(ds)
        for i, row in enumerate(ds[:n]):
            q = row["problem"].strip()
            gt = str(row["answer"]).strip()
            sents = [s.strip() for s in re.split(r"(?<=\.)\s+", row["solution"].strip()) if s.strip()]
            cut = max(1, len(sents) // 3)
            valid_steps = [s + "\n\n" for s in sents[:cut]]
            prompt = f"Solve the following mathematics problem step by step. Put your final answer in \\boxed{{}}.\n\nProblem: {q}\n\nSolution:\n\n"
            dead_unit, lemma = build_same_subspace_dead_end(q, "".join(valid_steps), rng)
            items.append(
                DeadEndItem(
                    idx=i,
                    prompt_text=prompt,
                    valid_prefix_steps=valid_steps,
                    same_subspace_dead_end_unit=dead_unit,
                    orthogonal_haystack_unit=rng.choice(ortho_pool),
                    negative_lemma=lemma,
                    continuation_cue="Continuing from the verified steps prior to the discarded branch:\n",
                    ground_truth=gt,
                )
            )
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return items

