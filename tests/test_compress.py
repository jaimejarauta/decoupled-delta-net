"""CPU verification suite for hybrid-kv-cliff across Qwen3, Qwen3.5, Granite-4.0-H, and Falcon-H1."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_kv_cliff.compress import (  # noqa: E402
    ARMS_DEFAULT,
    compress_kv_cache,
    forward_post_prefill,
    get_architecture_stats,
    greedy_decode_compressed,
    prefill_with_salience,
)
from hybrid_kv_cliff.evals import (  # noqa: E402
    score_exact_digits,
    score_mc_letter,
    score_token_f1,
)

_spec = importlib.util.spec_from_file_location(
    "kv_run_kill", Path(__file__).resolve().parents[1] / "scripts" / "run_kill.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
summarize, verdict = _mod.summarize, _mod.verdict


def _build_four_model_families() -> list[tuple[str, torch.nn.Module]]:
    from transformers import (
        FalconH1Config,
        FalconH1ForCausalLM,
        GraniteMoeHybridConfig,
        GraniteMoeHybridForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    q3_cfg = Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    q35_cfg = Qwen3_5TextConfig(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    gr_cfg = GraniteMoeHybridConfig(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        layer_types=["linear_attention", "linear_attention", "attention", "linear_attention", "linear_attention"],
        position_embedding_type="rope",
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_d_state=16,
        mamba_expand=2,
        mamba_n_groups=1,
        num_local_experts=0,
        shared_intermediate_size=64,
    )
    fh_cfg = FalconH1Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mamba_expand=2,
        mamba_d_ssm=64,
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_d_state=16,
        mamba_n_groups=1,
    )
    models = []
    for name, cls, cfg in [
        ("qwen3", Qwen3ForCausalLM, q3_cfg),
        ("qwen3_5", Qwen3_5ForCausalLM, q35_cfg),
        ("granite_h", GraniteMoeHybridForCausalLM, gr_cfg),
        ("falcon_h1", FalconH1ForCausalLM, fh_cfg),
    ]:
        cfg._attn_implementation = "sdpa"
        models.append((name, cls(cfg).eval()))
    return models


@torch.no_grad()
def test_all_four_architectures_compression_and_rope_invariants():
    doc_ids = torch.randint(0, 96, (1, 128))
    post_ids = torch.randint(0, 96, (1, 8))
    full_seq = torch.cat([doc_ids, post_ids], dim=1)

    for name, model in _build_four_model_families():
        arch = get_architecture_stats(model)
        if name == "qwen3":
            assert not arch["is_hybrid"] and not arch["is_sparse_attn_hybrid"]
        elif name in ("qwen3_5", "granite_h"):
            assert arch["is_hybrid"] and arch["is_sparse_attn_hybrid"]
        elif name == "falcon_h1":
            assert arch["is_hybrid"] and arch["is_parallel_hybrid"]

        # 1. Two-stage prefill_with_salience + post_prefill(full) must match monolithic forward to FP32 precision
        ref_logits = model(full_seq, use_cache=False).logits[:, -post_ids.shape[1] :]
        bundle = prefill_with_salience(model, doc_ids, obs_window=16, n_global_probes=8)
        c_full = compress_kv_cache(model, bundle, "full")
        full_post_logits = forward_post_prefill(model, c_full, post_ids)
        torch.testing.assert_close(full_post_logits, ref_logits, atol=1e-4, rtol=1e-4)

        # 2. 1x SnapKV must be bitwise identical to `full` (0 tokens evicted, exact RoPE positions)
        c_full_2 = compress_kv_cache(model, bundle, "full")
        c_1x = compress_kv_cache(model, bundle, "snapkv_1x")
        l_full_2 = forward_post_prefill(model, c_full_2, post_ids)
        l_1x = forward_post_prefill(model, c_1x, post_ids)
        torch.testing.assert_close(l_1x, l_full_2, atol=1e-6, rtol=1e-6)

        # 3. Every arm in ARMS_DEFAULT executes cleanly, compresses bytes as expected, and decodes tokens
        for arm in ARMS_DEFAULT:
            comp = compress_kv_cache(model, bundle, arm)
            if arm in ("snapkv_4x", "pyramidkv_4x", "kvzip_uniform_4x", "sink_tail_4x"):
                assert 3.2 <= comp.effective_kv_ratio <= 4.8, f"{name} {arm} ratio={comp.effective_kv_ratio}"
            elif arm == "snapkv_8x":
                assert 6.0 <= comp.effective_kv_ratio <= 9.0
            elif arm == "kivi_4bit":
                assert comp.effective_kv_ratio > 1.8  # FP32 CPU test tensors -> >3.2x in BF16 on GPU, >1.8x even at L=128
            elif arm == "kivi_2bit":
                assert comp.effective_kv_ratio > comp_kivi4_ratio  # checked in order below
            if arm == "kivi_4bit":
                comp_kivi4_ratio = comp.effective_kv_ratio

            gen = greedy_decode_compressed(
                model, comp, prompt_suffix_ids=post_ids, initial_last_logits=bundle.last_logits, max_new_tokens=4
            )
            assert len(gen) == 4


def test_scorers_and_preregistered_validity_and_cliff_gates():
    assert score_exact_digits(" 482910\nDone", "482910") == 1.0
    assert score_exact_digits("482911", "482910") == 0.0
    assert score_mc_letter("(C) Because of the funnel", "C") == 1.0
    assert pytest.approx(score_token_f1("the quick brown fox", "quick brown dog"), abs=1e-3) == 4.0 / 7.0

    def _make_rows(pure_snapkv_ok: float, hybrid_snapkv_ok: float, hybrid_kivi_ok: float):
        rows = []
        for idx in range(60):
            # Pure transformer control (`is_hybrid=False`)
            rows.append({
                "model": "Qwen3-1.7B", "task": "ruler_single", "target_length": 8192, "idx": idx,
                "arm": "full", "score": 1.0, "full_kv_bytes": 1000, "effective_kv_ratio": 1.0,
                "is_hybrid": False, "is_sparse_attn_hybrid": False, "num_attn_layers": 28, "num_total_layers": 28,
            })
            rows.append({
                "model": "Qwen3-1.7B", "task": "ruler_single", "target_length": 8192, "idx": idx,
                "arm": "snapkv_4x", "score": (1.0 if idx < int(60 * pure_snapkv_ok) else 0.0),
                "full_kv_bytes": 1000, "effective_kv_ratio": 4.0,
                "is_hybrid": False, "is_sparse_attn_hybrid": False, "num_attn_layers": 28, "num_total_layers": 28,
            })
            # Sparse-attention hybrid (`is_sparse_attn_hybrid=True`)
            for arm, rate, ratio in [
                ("full", 0.95, 1.0),
                ("snapkv_4x", hybrid_snapkv_ok, 4.0),
                ("kivi_4bit", hybrid_kivi_ok, 3.6),
            ]:
                rows.append({
                    "model": "Granite-4.0-H-1B", "task": "multi_turn_shift", "target_length": 8192, "idx": idx,
                    "arm": arm, "score": (1.0 if idx < int(60 * rate) else 0.0),
                    "full_kv_bytes": 1000, "effective_kv_ratio": ratio,
                    "is_hybrid": True, "is_sparse_attn_hybrid": True, "num_attn_layers": 4, "num_total_layers": 40,
                })
        return rows

    # Case 1: Broken SnapKV pipeline fails Pure-Transformer Control -> INVALID_PIPELINE
    dec_inv, _ = verdict(summarize(_make_rows(pure_snapkv_ok=0.80, hybrid_snapkv_ok=0.50, hybrid_kivi_ok=0.95)))
    assert dec_inv == "INVALID_PIPELINE"

    # Case 2: Valid pure-Transformer + Hybrid eviction cliff + KIVI rescue -> CONTINUE
    dec_cont, _ = verdict(summarize(_make_rows(pure_snapkv_ok=0.98, hybrid_snapkv_ok=0.60, hybrid_kivi_ok=0.94)))
    assert dec_cont == "CONTINUE"

    # Case 3: Valid pure-Transformer + Hybrid tolerates eviction with <=3pt drop -> H0_NO_CLIFF
    dec_h0, _ = verdict(summarize(_make_rows(pure_snapkv_ok=0.98, hybrid_snapkv_ok=0.94, hybrid_kivi_ok=0.94)))
    assert dec_h0 == "H0_NO_CLIFF"

    # Case 4: KIVI also collapses -> KILL
    dec_kill, _ = verdict(summarize(_make_rows(pure_snapkv_ok=0.98, hybrid_snapkv_ok=0.60, hybrid_kivi_ok=0.62)))
    assert dec_kill == "KILL"
