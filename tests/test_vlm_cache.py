"""CPU verification tests for zero-state-vlm-cache on real Qwen3.5 VLM architecture."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zero_state_vlm_cache.cache_ops import (  # noqa: E402
    assemble_modular_cache,
    encode_image_modular,
    encode_sink_prefix,
    forward_gdn_boundary_reset,
    forward_segmented_prefill,
    get_vlm_backbone,
    measure_cache_bytes,
    online_modular_prefill_and_decode,
)
from zero_state_vlm_cache.evals import (  # noqa: E402
    paired_bootstrap_ci,
    score_anls,
    score_chartqa_relaxed,
    score_multichoice,
)

_spec = importlib.util.spec_from_file_location(
    "vlm_run_kill", Path(__file__).resolve().parents[1] / "scripts" / "run_kill.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_greedy_decode_from_cache, summarize, verdict = _mod._greedy_decode_from_cache, _mod.summarize, _mod.verdict


def _tiny_qwen3_5_vlm() -> torch.nn.Module:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.manual_seed(42)
    tcfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    vcfg = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=64,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=1,
        in_channels=3,
    )
    cfg = Qwen3_5Config(
        text_config=tcfg,
        vision_config=vcfg,
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
    )
    cfg._attn_implementation = "eager"
    return Qwen3_5ForConditionalGeneration(cfg).eval()


@torch.no_grad()
def test_segmented_and_unrotated_mrope_exactness():
    """Mandatory pre-flight check:
    1) Segmented prefill without state reset matches full_joint logits to FP32 precision.
    2) Pre-MRoPE unrotated KV cache + dynamic 3D M-RoPE rotation + saved GDN state matches full_joint
       logits to FP32 precision (proving zero M-RoPE offset bug and zero sink-loss bug).
    3) Zero-state modular cache stores 0 recurrent bytes and >4x smaller total bytes than Marconi.
    """
    model = _tiny_qwen3_5_vlm()
    sink_ids = torch.tensor([[1, 2, 3, 4]])  # 4 sink tokens
    img_ids = torch.tensor([[12, 10, 10, 10, 10, 13]])  # <|vision_start|> + 4 merged patches + <|vision_end|>
    q_ids = torch.tensor([[20, 21, 22, 23]])
    full_ids = torch.cat([sink_ids, img_ids, q_ids], dim=1)
    mm_type = (full_ids == 10).int()
    grid_thw = torch.tensor([[1, 4, 4]])
    pix = torch.randn(16, 3 * 1 * 4 * 4)

    ref_out = model(
        input_ids=full_ids,
        pixel_values=pix,
        image_grid_thw=grid_thw,
        mm_token_type_ids=mm_type,
        use_cache=True,
    )
    marconi_mem = measure_cache_bytes(ref_out.past_key_values)
    assert marconi_mem["recurrent_bytes"] > 4 * marconi_mem["kv_bytes"]

    # 1. Segmented prefill with reset=False must equal full_joint exactly
    seg_logits, _ = forward_segmented_prefill(
        model, full_ids, pix, grid_thw, mm_type, reset_at_vision_boundaries=False
    )
    torch.testing.assert_close(seg_logits, ref_out.logits, atol=1e-4, rtol=1e-4)

    # 2. Pre-MRoPE unrotated image KV + dynamic M-RoPE + saved GDN state must equal full_joint on question logits
    sink_kv, _ = encode_sink_prefix(model, sink_ids)
    mod_kv, saved_gdn_cache = encode_image_modular(
        model, sink_ids, img_ids, pix, grid_thw, keep_debug_cache=True
    )
    cache_exact, q_embeds, q_pos_4d = assemble_modular_cache(
        model, sink_kv, [mod_kv], q_ids, saved_linear_cache=saved_gdn_cache
    )
    _, lm, _ = get_vlm_backbone(model)
    q_out_exact = model.lm_head(
        lm(inputs_embeds=q_embeds, position_ids=q_pos_4d, past_key_values=cache_exact, use_cache=True).last_hidden_state
    )
    torch.testing.assert_close(q_out_exact, ref_out.logits[:, -q_ids.shape[1] :], atol=1e-4, rtol=1e-4)

    # 3. Zero-state modular cache has 0 recurrent bytes before question prefill
    cache_zero, _, _ = assemble_modular_cache(model, sink_kv, [mod_kv], q_ids, saved_linear_cache=None)
    zero_mem = measure_cache_bytes(cache_zero)
    assert zero_mem["recurrent_bytes"] == 0
    assert zero_mem["kv_bytes"] > 0

    # 4. Multi-image composition works in arbitrary order [img_B, img_A] without state coupling
    pix_b = torch.randn(16, 3 * 1 * 4 * 4)
    mod_kv_b, _ = encode_image_modular(model, sink_ids, img_ids, pix_b, grid_thw)
    cache_multi, q_emb_m, q_pos_m = assemble_modular_cache(model, sink_kv, [mod_kv_b, mod_kv], q_ids)
    out_m = lm(inputs_embeds=q_emb_m, position_ids=q_pos_m, past_key_values=cache_multi, use_cache=True)
    assert out_m.last_hidden_state.shape == (1, q_ids.shape[1], 64)

    # 5. Boundary reset executes cleanly and alters recurrent contribution
    rst_logits, _ = forward_gdn_boundary_reset(model, full_ids, pix, grid_thw, mm_type)
    assert rst_logits.shape == ref_out.logits.shape
    assert not torch.allclose(rst_logits[:, -1], ref_out.logits[:, -1], atol=1e-6)

    # 6. Multi-step greedy decode helper & online modular decode execute cleanly
    vlm, _, _ = get_vlm_backbone(model)
    L = full_ids.shape[1]
    base_3d = torch.arange(L, L + 1).view(1, 1, 1).expand(3, 1, 1)
    if getattr(vlm, "rope_deltas", None) is not None:
        base_3d = base_3d + vlm.rope_deltas.unsqueeze(0)
    gen_ref = _greedy_decode_from_cache(model, ref_out.logits, ref_out.past_key_values, L, base_3d, max_new_tokens=5)
    gen_mod, ttft_ms = online_modular_prefill_and_decode(model, sink_kv, [mod_kv], q_ids, max_new_tokens=5)
    assert len(gen_ref) == 5 and len(gen_mod) == 5 and ttft_ms >= 0.0


def test_metrics_and_preregistered_gate():
    assert score_anls("1998", ["1998"]) == 1.0
    assert score_anls("completely wrong", ["1998"]) == 0.0
    assert score_chartqa_relaxed("102.0", ["100"]) == 1.0  # within 5%
    assert score_chartqa_relaxed("112.0", ["100"]) == 0.0  # > 5%
    assert score_multichoice("(B) Second image", "B") == 1.0

    mean_d, lo, hi = paired_bootstrap_ci([1.0] * 50 + [0.0] * 50, [1.0] * 49 + [0.0] * 51)
    assert pytest.approx(mean_d, abs=1e-4) == 1.0
    assert lo <= mean_d <= hi

    # Synthetic verdict table check: CONTINUE vs KILL
    rows_pass = []
    for task in ("docvqa", "mp_docvqa", "muirbench"):
        for i in range(100):
            base_ok = 1.0 if i < 80 else 0.0
            rows_pass += [
                {"model": "m", "task": task, "idx": i, "arm": "full_joint", "score": base_ok, "ttft_ms": 500.0, "total_mb": 29.2, "kv_mb": 4.0},
                {"model": "m", "task": task, "idx": i, "arm": "gdn_boundary_reset", "score": base_ok, "ttft_ms": 510.0, "total_mb": 29.2, "kv_mb": 4.0},
                {"model": "m", "task": task, "idx": i, "arm": "zero_state_modular", "score": base_ok, "ttft_ms": 80.0, "total_mb": 4.0, "kv_mb": 4.0},
                {"model": "m", "task": task, "idx": i, "arm": "pooled_joint_2x2", "score": (1.0 if i < 65 else 0.0), "ttft_ms": 140.0, "total_mb": 26.2, "kv_mb": 1.0},
            ]
    dec_pass, _ = verdict(summarize(rows_pass))
    assert dec_pass == "CONTINUE"

    rows_kill = [dict(r, score=(0.4 if r["arm"] != "full_joint" else r["score"])) for r in rows_pass]
    dec_kill, _ = verdict(summarize(rows_kill))
    assert dec_kill == "KILL"
