"""Clean-context witness protocol, invertibility stratification, and Macro-F1 scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

WITNESS_RE = re.compile(r"<witness>(.*?)</witness>", re.DOTALL | re.IGNORECASE)
VERDICT_RE = re.compile(r"<verdict>\s*(ACCEPT|REJECT)\s*</verdict>", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")

NON_INVERTIBLE_CUE_RE = re.compile(
    r"\b(?:how\s+many|number\s+of|count\s+the|sum\s+of\s+all|product\s+of\s+all|"
    r"minimum|maximum|smallest|largest|greatest|least|at\s+most|at\s+least)\b",
    re.IGNORECASE,
)


@dataclass
class ParsedSolution:
    derivation_d1: str
    witness_w1: str
    answer_a1: str


def classify_invertibility(problem_text: str) -> Literal["invertible", "non_invertible"]:
    """Stratify math problems into direct-substitution (`invertible`) vs. aggregation/extremal
    (`non_invertible`, where bare scalar `A_1` cannot be checked without constructive objects)."""
    return "non_invertible" if NON_INVERTIBLE_CUE_RE.search(problem_text) else "invertible"


def build_solver_prompt(problem: str, max_witness_tokens: int = 64) -> str:
    return (
        "Solve the problem step by step. Before your final answer, emit a self-contained "
        f"constructive check inside <witness>...</witness> (at most {max_witness_tokens} tokens: "
        "concrete variable values, explicit set elements, or tight bound example so an independent "
        "checker who CANNOT see your derivation can verify the answer directly), followed by your "
        "final answer in \\boxed{}.\n\n"
        f"Problem: {problem.strip()}\n\nSolution:\n"
    )


def build_clean_verifier_prompt(
    problem: str,
    witness_w1: str,
    answer_a1: str,
    derivation_d1: str | None = None,
) -> str:
    """Build verifier input. When `derivation_d1 is None` (WB-GRPO / Clean-2-Call), D_1 is completely
    absent from the string and token stream, enforcing exact conditional independence given (P, W_1, A_1)."""
    parts = [
        "Verify whether the proposed final answer satisfies every condition of the problem by "
        "explicitly checking the provided witness values/objects. Conclude with exactly "
        "<verdict>ACCEPT</verdict> or <verdict>REJECT</verdict>.\n",
        f"Problem: {problem.strip()}",
    ]
    if derivation_d1 is not None:
        parts.append(f"Candidate Derivation: {derivation_d1.strip()}")
    if witness_w1.strip():
        parts.append(f"Candidate Witness: {witness_w1.strip()}")
    parts.append(f"Proposed Final Answer: {answer_a1.strip()}\n\nVerification:\n")
    return "\n\n".join(parts)


def build_retry_prompt(problem: str, witness_w1: str, answer_a1: str, critique_v1: str, max_witness_tokens: int = 64) -> str:
    return (
        f"Solve the problem correctly. An independent check rejected a previous candidate.\n\n"
        f"Problem: {problem.strip()}\n\n"
        f"Rejected Candidate Answer: {answer_a1.strip()} (Witness: {witness_w1.strip()})\n"
        f"Verification Critique: {critique_v1.strip()}\n\n"
        f"Provide a corrected derivation, a new <witness>...</witness> (<= {max_witness_tokens} tokens), "
        "and the corrected final answer in \\boxed{}:\n"
    )


def parse_solver_output(text: str) -> ParsedSolution:
    m_wit = WITNESS_RE.search(text)
    if m_wit:
        w1 = m_wit.group(1).strip()
        d1 = (text[: m_wit.start()] + " " + text[m_wit.end() :]).strip()
    else:
        w1 = ""
        d1 = text.strip()
    boxes = BOXED_RE.findall(text)
    a1 = boxes[-1].strip() if boxes else ""
    return ParsedSolution(derivation_d1=d1, witness_w1=w1, answer_a1=a1)


def truncate_witness_tokens(tok, witness_text: str, max_tokens: int = 64) -> tuple[str, int]:
    """Decode-reencode discrete token truncation guaranteeing channel capacity <= max_tokens * log2|V|."""
    if max_tokens <= 0 or not witness_text.strip():
        return "", 0
    ids = tok(witness_text, add_special_tokens=False).input_ids
    orig_len = len(ids)
    if orig_len > max_tokens:
        ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=True).strip(), orig_len


def parse_verdict(text: str) -> Literal["ACCEPT", "REJECT", "INVALID"]:
    matches = VERDICT_RE.findall(text)
    if not matches:
        return "INVALID"
    return matches[-1].upper()  # type: ignore[return-value]


def normalize_math_answer(ans: str) -> str:
    s = ans.strip().replace(",", "").replace("$", "").replace("\\!", "").replace(" ", "")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    m_frac = re.match(r"^\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}$", s)
    if m_frac:
        num, den = int(m_frac.group(1)), int(m_frac.group(2))
        if den != 0:
            f = Fraction(num, den)
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    try:
        if "/" in s and re.match(r"^[-+]?\d+/[-+]?\d+$", s):
            f = Fraction(s)
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        v = float(s)
        return str(int(v)) if v.is_integer() else f"{v:.6g}"
    except (ValueError, ZeroDivisionError):
        return s


def grade_math_answer(pred: str, gt: str) -> bool:
    if not pred.strip():
        return False
    return normalize_math_answer(pred) == normalize_math_answer(gt)


def macro_f1_metrics(y_correct: list[bool], verdicts: list[str]) -> dict[str, float]:
    """Balanced Macro-F1 across positive (`correct -> ACCEPT`) and negative (`wrong -> REJECT`).

    Note: `INVALID` verdicts count as errors for both classes so format gaming cannot inflate F1.
    """
    assert len(y_correct) == len(verdicts) and len(y_correct) > 0
    tp_pos = sum(y and v == "ACCEPT" for y, v in zip(y_correct, verdicts))
    fp_pos = sum((not y) and v == "ACCEPT" for y, v in zip(y_correct, verdicts))
    fn_pos = sum(y and v != "ACCEPT" for y, v in zip(y_correct, verdicts))
    f1_pos = (2.0 * tp_pos) / max(1.0, 2.0 * tp_pos + fp_pos + fn_pos)

    tp_neg = sum((not y) and v == "REJECT" for y, v in zip(y_correct, verdicts))
    fp_neg = sum(y and v == "REJECT" for y, v in zip(y_correct, verdicts))
    fn_neg = sum((not y) and v != "REJECT" for y, v in zip(y_correct, verdicts))
    f1_neg = (2.0 * tp_neg) / max(1.0, 2.0 * tp_neg + fp_neg + fn_neg)

    n_pos = max(1, sum(y_correct))
    n_neg = max(1, sum(not y for y in y_correct))
    tpr = tp_pos / n_pos
    tnr = tp_neg / n_neg
    return {
        "macro_f1": 0.5 * (f1_pos + f1_neg),
        "f1_accept": f1_pos,
        "f1_reject": f1_neg,
        "tpr_accept": tpr,
        "tnr_reject": tnr,
        "balanced_acc": 0.5 * (tpr + tnr),
    }

