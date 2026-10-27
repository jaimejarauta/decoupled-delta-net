"""Protocol parsing, discrete witness bottlenecking, invertibility stratification, and WB-GRPO."""
from .grpo import compute_group_advantages, compute_wb_grpo_rewards, wb_grpo_step_loss
from .protocol import (
    ParsedSolution,
    build_clean_verifier_prompt,
    build_solver_prompt,
    classify_invertibility,
    macro_f1_metrics,
    parse_solver_output,
    parse_verdict,
    truncate_witness_tokens,
)

__all__ = [
    "ParsedSolution",
    "build_clean_verifier_prompt",
    "build_solver_prompt",
    "classify_invertibility",
    "compute_group_advantages",
    "compute_wb_grpo_rewards",
    "macro_f1_metrics",
    "parse_solver_output",
    "parse_verdict",
    "truncate_witness_tokens",
    "wb_grpo_step_loss",
]

