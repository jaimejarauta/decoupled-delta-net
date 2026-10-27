"""Kill Experiment runner & pre-registered gate evaluator for Backup #1 (C2):
Witness-Bottlenecked GRPO (`WB-GRPO`) vs. `GRPO-Verif`, Zero-RL 2-Call, Answer-Only, and Majority@2.

Usage:
  python scripts/run_kill_c2.py eval --model Qwen/Qwen3-8B --external-verifier meta-llama/Llama-3.1-8B-Instruct
  python scripts/run_kill_c2.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wb_verify.protocol import (  # noqa: E402
    build_clean_verifier_prompt,
    build_retry_prompt,
    build_solver_prompt,
    classify_invertibility,
    grade_math_answer,
    macro_f1_metrics,
    parse_solver_output,
    parse_verdict,
    truncate_witness_tokens,
)

OUT_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "kill_c2.jsonl"


@torch.no_grad()
def _generate_one(model, tok, prompt: str, max_new_tokens: int, temperature: float, seed: int) -> tuple[str, int]:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(device)
    in_len = enc.input_ids.shape[1]
    torch.manual_seed(seed)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 1e-5),
        temperature=max(temperature, 1e-5),
        pad_token_id=tok.eos_token_id,
    )
    gen_ids = out[0, in_len:]
    return tok.decode(gen_ids, skip_special_tokens=True), int(in_len + gen_ids.shape[0])


def evaluate_arms_on_item(
    solver_model,
    tok,
    idx: int,
    problem: str,
    gt: str,
    arm_tag: str = "prompted_2call",
    witness_budget: int = 64,
    keep_d1: bool = False,
    external_verifier=None,
    ext_tok=None,
    max_solve_tokens: int = 768,
    max_verify_tokens: int = 256,
    seed: int = 0,
) -> dict:
    inv_kind = classify_invertibility(problem)
    s_prompt = build_solver_prompt(problem, max_witness_tokens=max(16, witness_budget))
    raw_s1, toks_s1 = _generate_one(solver_model, tok, s_prompt, max_solve_tokens, temperature=0.6, seed=seed)
    parsed = parse_solver_output(raw_s1)
    w1_trunc, _ = truncate_witness_tokens(tok, parsed.witness_w1, max_tokens=witness_budget)
    y1 = grade_math_answer(parsed.answer_a1, gt)

    v_prompt = build_clean_verifier_prompt(
        problem,
        witness_w1=w1_trunc,
        answer_a1=parsed.answer_a1,
        derivation_d1=parsed.derivation_d1 if keep_d1 else None,
    )
    raw_v1, toks_v1 = _generate_one(solver_model, tok, v_prompt, max_verify_tokens, temperature=0.0, seed=seed + 1)
    verdict_self = parse_verdict(raw_v1)

    total_tokens = toks_s1 + toks_v1
    final_ans = parsed.answer_a1
    if verdict_self == "REJECT":
        r_prompt = build_retry_prompt(problem, w1_trunc, parsed.answer_a1, raw_v1, max_witness_tokens=max(16, witness_budget))
        raw_s2, toks_s2 = _generate_one(solver_model, tok, r_prompt, max_solve_tokens, temperature=0.6, seed=seed + 2)
        final_ans = parse_solver_output(raw_s2).answer_a1 or final_ans
        total_tokens += toks_s2

    ext_verdict = None
    if external_verifier is not None and ext_tok is not None:
        raw_ext, _ = _generate_one(external_verifier, ext_tok, v_prompt, max_verify_tokens, temperature=0.0, seed=seed + 3)
        ext_verdict = parse_verdict(raw_ext)

    return {
        "i": idx,
        "arm": arm_tag,
        "invertibility": inv_kind,
        "y1_correct": y1,
        "verdict": verdict_self,
        "ext_verdict": ext_verdict,
        "final_correct": grade_math_answer(final_ans, gt),
        "total_tokens": total_tokens,
    }


def evaluate_majority2(solver_model, tok, idx: int, problem: str, gt: str, max_solve_tokens: int = 768, seed: int = 0) -> dict:
    s_prompt = build_solver_prompt(problem, max_witness_tokens=64)
    r1, t1 = _generate_one(solver_model, tok, s_prompt, max_solve_tokens, temperature=0.6, seed=seed)
    r2, t2 = _generate_one(solver_model, tok, s_prompt, max_solve_tokens, temperature=0.6, seed=seed + 100)
    a1, a2 = parse_solver_output(r1).answer_a1, parse_solver_output(r2).answer_a1
    counts = Counter([a for a in (a1, a2) if a])
    voted = counts.most_common(1)[0][0] if counts else a1
    return {
        "i": idx,
        "arm": "majority2",
        "invertibility": classify_invertibility(problem),
        "y1_correct": grade_math_answer(a1, gt),
        "verdict": "ACCEPT",
        "ext_verdict": None,
        "final_correct": grade_math_answer(voted, gt),
        "total_tokens": t1 + t2,
    }


def summarize_c2(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["arm"]].append(r)

    summary: dict[str, dict[str, float]] = {}
    for arm, items in grouped.items():
        non_inv = [x for x in items if x["invertibility"] == "non_invertible"]
        target = non_inv if non_inv else items
        f1 = macro_f1_metrics([bool(x["y1_correct"]) for x in target], [x["verdict"] for x in target])
        ext_items = [x for x in target if x.get("ext_verdict") is not None]
        ext_f1 = (
            macro_f1_metrics([bool(x["y1_correct"]) for x in ext_items], [x["ext_verdict"] for x in ext_items])["macro_f1"]
            if ext_items
            else float("nan")
        )
        summary[arm] = {
            "non_inv_macro_f1": 100.0 * f1["macro_f1"],
            "overall_final_acc": 100.0 * sum(bool(x["final_correct"]) for x in items) / len(items),
            "ext_transfer_macro_f1": 100.0 * ext_f1,
            "mean_tokens": sum(float(x["total_tokens"]) for x in items) / len(items),
        }
    return summary


def verdict_c2(summary: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    """Pre-registered Gate for Backup #1 (C2):
      1. `wb_grpo` Non-Invertible Macro-F1 >= `grpo_verif` + 6.0 pts AND >= `prompted_2call` + 5.0 pts
      2. `wb_grpo` Overall Post-Retry Accuracy >= `majority2` + 2.0 pts
      3. Anti-Steganography Gate (if external verifier run): `wb_grpo` ext_transfer_macro_f1 >= `prompted_2call` + 4.0 pts
    """
    wb = summary.get("wb_grpo", {})
    gv = summary.get("grpo_verif", {})
    p2 = summary.get("prompted_2call", {})
    m2 = summary.get("majority2", {})

    d_gv = wb.get("non_inv_macro_f1", 0.0) - gv.get("non_inv_macro_f1", 0.0)
    d_p2 = wb.get("non_inv_macro_f1", 0.0) - p2.get("non_inv_macro_f1", 0.0)
    d_m2 = wb.get("overall_final_acc", 0.0) - m2.get("overall_final_acc", 0.0)
    ext_wb = wb.get("ext_transfer_macro_f1", float("nan"))
    ext_p2 = p2.get("ext_transfer_macro_f1", float("nan"))
    ext_ok = True if (ext_wb != ext_wb or ext_p2 != ext_p2) else ((ext_wb - ext_p2) >= 4.0)

    passed = (d_gv >= 6.0) and (d_p2 >= 5.0) and (d_m2 >= 2.0) and ext_ok
    return ("PASS" if passed else "KILL"), {"d_f1_vs_grpo_verif": d_gv, "d_f1_vs_prompted": d_p2, "d_acc_vs_maj2": d_m2}


def cmd_eval(args) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    ext_model, ext_tok = None, None
    if args.external_verifier:
        ext_tok = AutoTokenizer.from_pretrained(args.external_verifier)
        if ext_tok.pad_token_id is None:
            ext_tok.pad_token_id = ext_tok.eos_token_id
        ext_model = AutoModelForCausalLM.from_pretrained(args.external_verifier, torch_dtype=dtype).to(device).eval()

    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[: args.n]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for idx, row in enumerate(ds):
            q, gt = row["problem"], str(row["answer"])
            for tag, b_tok, keep_d1 in [
                ("prompted_2call", 64, False),
                ("answer_only", 0, False),
                ("full_trace_prompted", 64, True),
            ]:
                res = evaluate_arms_on_item(
                    model, tok, idx, q, gt, arm_tag=tag, witness_budget=b_tok, keep_d1=keep_d1,
                    external_verifier=ext_model, ext_tok=ext_tok, seed=idx,
                )
                f.write(json.dumps(res) + "\n")
            f.write(json.dumps(evaluate_majority2(model, tok, idx, q, gt, seed=idx)) + "\n")
            f.flush()


def cmd_report(args) -> None:
    rows = [json.loads(line) for line in Path(args.out).open() if line.strip()]
    summ = summarize_c2(rows)
    for arm, stats in summ.items():
        print(f"{arm:24s} | " + "  ".join(f"{k}={v:.2f}" for k, v in stats.items()))
    v, deltas = verdict_c2(summ)
    print(f"\nPRE-REGISTERED VERDICT (C2): {v}  deltas={deltas}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("eval")
    pe.add_argument("--model", required=True)
    pe.add_argument("--external-verifier", default=None)
    pe.add_argument("--n", type=int, default=100)
    pe.add_argument("--out", default=str(OUT_DEFAULT))
    pr = sub.add_parser("report")
    pr.add_argument("--out", default=str(OUT_DEFAULT))
    a = p.parse_args()
    cmd_eval(a) if a.cmd == "eval" else cmd_report(a)

