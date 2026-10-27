"""Unit tests for Backup #1 (`wb-witness-verify`): derivation-erasure isolation,
anti-rubber-stamp class-balanced reward, WB-GRPO loss gradients, and pre-registered gate."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_kill_c2 import summarize_c2, verdict_c2  # noqa: E402
from wb_verify.grpo import (  # noqa: E402
    compute_group_advantages,
    compute_wb_grpo_rewards,
    wb_grpo_step_loss,
)
from wb_verify.protocol import (  # noqa: E402
    build_clean_verifier_prompt,
    classify_invertibility,
    grade_math_answer,
    macro_f1_metrics,
    parse_solver_output,
    parse_verdict,
    truncate_witness_tokens,
)


def test_derivation_erasure_and_invertibility_stratification():
    raw = (
        "Let us try k=1, 2, 3. Secret persuasion marker XYZ_999. "
        "<witness>Valid solutions are k in {2, 5, 11}, sum = 2 + 5 + 11 = 18.</witness> "
        "Therefore \\boxed{18}."
    )
    parsed = parse_solver_output(raw)
    assert "XYZ_999" in parsed.derivation_d1
    assert "XYZ_999" not in parsed.witness_w1
    assert parsed.answer_a1 == "18"

    clean_prompt = build_clean_verifier_prompt("Find the sum of all prime k < 12.", parsed.witness_w1, parsed.answer_a1)
    assert "XYZ_999" not in clean_prompt, "Clean verifier prompt must NEVER contain D_1 tokens"
    full_prompt = build_clean_verifier_prompt(
        "Find the sum of all prime k < 12.", parsed.witness_w1, parsed.answer_a1, derivation_d1=parsed.derivation_d1
    )
    assert "XYZ_999" in full_prompt

    assert classify_invertibility("Find the sum of all positive integers n such that n^2 < 50.") == "non_invertible"
    assert classify_invertibility("How many real solutions does x^4 - 3x + 1 = 0 have?") == "non_invertible"
    assert classify_invertibility("Solve 3x + 7 = 22 for x.") == "invertible"
    assert grade_math_answer("\\frac{6}{8}", "3/4")
    assert parse_verdict("Checking substitution... <verdict>REJECT</verdict>") == "REJECT"


def test_macro_f1_and_dynamic_reward_defeat_rubber_stamp_gaming():
    # Base solver is 87.5% accurate (7 correct, 1 wrong).
    y_first = [True, True, True, True, True, True, True, False]
    y_retry = [False] * 8
    w_lens = [20] * 8

    # Rubber-stamp ACCEPT policy gets 87.5% raw accuracy, but Macro-F1 < 0.50 and ver_score == 0.50!
    f1_rubber = macro_f1_metrics(y_first, ["ACCEPT"] * 8)
    assert f1_rubber["macro_f1"] < 0.47
    _, stats_accept_all = compute_wb_grpo_rewards(y_first, ["ACCEPT"] * 8, y_retry, w_lens)
    _, stats_reject_all = compute_wb_grpo_rewards(y_first, ["REJECT"] * 8, y_retry, w_lens)
    _, stats_oracle = compute_wb_grpo_rewards(
        y_first, ["ACCEPT"] * 7 + ["REJECT"], y_retry, w_lens
    )

    assert abs(stats_accept_all["mean_ver_score"] - 0.5) < 1e-6
    assert abs(stats_reject_all["mean_ver_score"] - 0.5) < 1e-6
    assert abs(stats_oracle["mean_ver_score"] - 1.0) < 1e-6


def test_wb_grpo_step_loss_backpropagates_cleanly():
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    policy = AutoModelForCausalLM.from_config(cfg).train()
    ref = copy.deepcopy(policy).eval()

    B = 4
    s_ids = torch.randint(0, 64, (B, 14))
    v_ids = torch.randint(0, 64, (B, 10))
    s_mask = torch.ones_like(s_ids)
    v_mask = torch.ones_like(v_ids)
    rewards = torch.tensor([1.5, 0.2, 1.0, -0.1])
    adv = compute_group_advantages(rewards, group_size=2)

    loss, metrics = wb_grpo_step_loss(
        policy, ref, s_ids, s_mask, [5] * B, v_ids, v_mask, [4] * B, adv
    )
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
    assert grad_norm > 0.0
    assert metrics["kl_solver"] >= -1e-6 and metrics["kl_verifier"] >= -1e-6


def test_witness_truncation_and_c2_verdict_gate():
    class _SpaceTok:
        def __call__(self, s: str, add_special_tokens: bool = False):
            return type("E", (), {"input_ids": s.split()})()

        def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
            return " ".join(ids)

    trunc, orig = truncate_witness_tokens(_SpaceTok(), "w1 w2 w3 w4 w5 w6", max_tokens=3)
    assert orig == 6 and trunc == "w1 w2 w3"

    def _rows(arm: str, acc: float, good_ver: bool):
        out = []
        for i in range(40):
            y1 = i < 20
            v = ("ACCEPT" if y1 else "REJECT") if good_ver else "ACCEPT"
            out.append(
                dict(
                    i=i, arm=arm, invertibility="non_invertible",
                    y1_correct=y1, verdict=v, ext_verdict=v,
                    final_correct=(i < int(40 * acc)), total_tokens=500,
                )
            )
        return out

    rows = (
        _rows("wb_grpo", 0.80, True)
        + _rows("grpo_verif", 0.72, False)
        + _rows("prompted_2call", 0.72, False)
        + _rows("majority2", 0.75, False)
    )
    v, _ = verdict_c2(summarize_c2(rows))
    assert v == "PASS"

