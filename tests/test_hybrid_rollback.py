"""CPU exactness & invariant tests on real HuggingFace hybrid architectures
(`Qwen3_5`, `FalconH1`, `Zamba2`, `GraniteMoeHybrid`)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hybrid_rollback.cache_state import (  # noqa: E402
    capture_recurrent_snapshot,
    count_hybrid_cache_layers,
    get_kv_seq_length,
    prefill_tokens,
    rollback_attention_only_kv,
    rollback_synchronized,
)
from hybrid_rollback.live_decode import live_decode_with_rollback  # noqa: E402
from hybrid_rollback.revolve import OnlineCheckpointPool, RecurrentSnapshot  # noqa: E402
from hybrid_rollback.subspace_probe import (  # noqa: E402
    measure_recurrent_state_drift,
    measure_subspace_overlap,
)
from hybrid_rollback.tasks import answers_match, detect_backtrack_trigger, load_dead_end_suite  # noqa: E402
from run_kill_c1 import summarize, verdict  # noqa: E402

KINDS = ["qwen3_5", "falcon_h1", "zamba2", "granite"]


def make_tiny_hybrid(kind: str):
    from transformers import (
        AutoModelForCausalLM,
        FalconH1Config,
        GraniteMoeHybridConfig,
        Qwen3_5TextConfig,
        Zamba2Config,
    )

    common = dict(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    if kind == "qwen3_5":
        cfg = Qwen3_5TextConfig(
            **common,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
    elif kind == "falcon_h1":
        cfg = FalconH1Config(
            **common,
            num_hidden_layers=2,
            mamba_d_ssm=64,
            mamba_n_heads=4,
            mamba_d_head=16,
            mamba_d_state=8,
            mamba_n_groups=1,
            mamba_chunk_size=8,
            head_dim=16,
        )
    elif kind == "zamba2":
        cfg = Zamba2Config(
            **common,
            num_hidden_layers=3,
            layers_block_type=["mamba", "hybrid", "mamba"],
            mamba_d_state=8,
            mamba_headdim=16,
            n_mamba_heads=8,
            chunk_size=8,
            num_mem_blocks=1,
            use_shared_attention_adapter=False,
            adapter_rank=8,
        )
    elif kind == "granite":
        cfg = GraniteMoeHybridConfig(
            **common,
            num_hidden_layers=2,
            layer_types=["mamba", "attention"],
            mamba_d_state=8,
            mamba_d_head=16,
            mamba_n_heads=8,
            mamba_chunk_size=8,
            num_local_experts=0,
            shared_intermediate_size=64,
        )
    else:
        raise ValueError(kind)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).eval()


def test_online_checkpoint_pool_bounds_and_cascading_rollback():
    """Verify min_span and dyadic_ruler bound worst-case replay across [0, N] and survive
    cascading rollbacks, whereas ring_fifo degrades to full-prefix replay on long dead-ends."""
    min_pool = OnlineCheckpointPool(capacity=8, policy="min_span")
    dya_pool = OnlineCheckpointPool(capacity=8, policy="dyadic_ruler")
    fifo_pool = OnlineCheckpointPool(capacity=8, policy="ring_fifo")

    step_tokens = 32
    n_steps = 80
    for s in range(1, n_steps + 1):
        pos = s * step_tokens
        min_pool.push(RecurrentSnapshot(token_pos=pos, step_idx=0, level=0, layers=[], nbytes=128))
        dya_pool.push(RecurrentSnapshot(token_pos=pos, step_idx=0, level=0, layers=[], nbytes=128))
        fifo_pool.push(RecurrentSnapshot(token_pos=pos, step_idx=0, level=0, layers=[], nbytes=128))

    total_len = n_steps * step_tokens  # 2560 tokens
    target_a = 20 * step_tokens        # 640 (dead-end length = 1920 tokens = 60 steps > C=8)

    # Ring FIFO evicted everything <= 640, so nearest_at_or_before is None -> 640 replay tokens!
    assert fifo_pool.nearest_at_or_before(target_a) is None

    # min_span and dyadic_ruler retain anchor + interior coverage -> much shorter replay!
    s_min = min_pool.nearest_at_or_before(target_a)
    s_dya = dya_pool.nearest_at_or_before(target_a)
    assert s_min is not None and (target_a - s_min.token_pos) <= (2 * total_len) // 7
    assert s_dya is not None and (target_a - s_dya.token_pos) < target_a

    # Cascading second rollback to an even earlier step (a2 = 8 * 32 = 256)
    min_pool.prune_after(target_a)
    s_min2 = min_pool.nearest_at_or_before(8 * step_tokens)
    assert s_min2 is not None and s_min2.token_pos <= 8 * step_tokens


@pytest.mark.parametrize("kind", KINDS)
@torch.no_grad()
def test_synchronized_rollback_exactness_and_desync_divergence(kind: str):
    """Across all 4 HF hybrid families:
    1. HF DynamicCache natively marks hybrid recurrent caches `is_croppable == False`.
    2. `rollback_synchronized` after a 20-token dead-end reproduces clean unpolluted continuation
       to floating-point precision (< 1e-5), both on exact snapshot hits and mid-span replays.
    3. `rollback_attention_only_kv` (SkipKV/FoldGRPO naive port) leaves S_b polluted and diverges.
    """
    model = make_tiny_hybrid(kind)
    torch.manual_seed(42)
    prefix = torch.randint(0, 97, (1, 18))
    dead_end = torch.randint(0, 97, (1, 20))
    continuation = torch.randint(0, 97, (1, 8))

    # Clean reference logits on [prefix + continuation]
    clean_full = model(torch.cat([prefix, continuation], dim=1), use_cache=False).logits[:, 18:]

    # Build step-checkpointed cache up to 18 (snapshots at 6 and 12; target 18 requires 6-token replay!)
    pool = OnlineCheckpointPool(capacity=3, policy="min_span")
    pkv, _ = prefill_tokens(model, prefix[:, :6])
    pool.push(capture_recurrent_snapshot(pkv, 6))
    pkv, _ = prefill_tokens(model, prefix[:, 6:12], past_key_values=pkv)
    pool.push(capture_recurrent_snapshot(pkv, 12))
    pkv, _ = prefill_tokens(model, prefix[:, 12:18], past_key_values=pkv)
    snap_18 = capture_recurrent_snapshot(pkv, 18)  # kept externally only to verify drift = 0 after restore

    assert not pkv.is_croppable, "HF DynamicCache should report is_croppable=False on recurrent hybrids"
    n_rec, n_kv = count_hybrid_cache_layers(pkv)
    assert n_rec >= 1 and n_kv >= 1

    # Pollute cache with 20 dead-end tokens [18..38)
    pkv, _ = prefill_tokens(model, dead_end, past_key_values=pkv)
    assert get_kv_seq_length(pkv) == 38
    drift_polluted = measure_recurrent_state_drift(snap_18, pkv)
    assert drift_polluted["mean_rel_drift"] > 1e-3

    # Synchronized rollback to target_pos=18 (replays [12..18) = 6 tokens from snapshot @ 12)
    rb = rollback_synchronized(model, pkv, prefix, pool, target_pos=18)
    assert rb.base_pos == 12 and rb.replay_tokens == 6
    assert get_kv_seq_length(pkv) == 18
    drift_restored = measure_recurrent_state_drift(snap_18, pkv)
    assert drift_restored["mean_rel_drift"] < 1e-6

    out_sync = model(continuation, past_key_values=pkv, use_cache=True).logits
    torch.testing.assert_close(out_sync, clean_full, atol=1e-5, rtol=1e-4)

    # Now test Attention-Only KV Rollback (desynchronized S_38 + KV_18)
    pkv_desync, _ = prefill_tokens(model, torch.cat([prefix, dead_end], dim=1))
    rollback_attention_only_kv(pkv_desync, target_pos=18)
    assert get_kv_seq_length(pkv_desync) == 18
    out_desync = model(continuation, past_key_values=pkv_desync, use_cache=True).logits
    assert not torch.allclose(out_desync, clean_full, atol=1e-4), "polluted recurrent state must alter logits"


def test_subspace_overlap_separates_same_subspace_from_orthogonal():
    model = make_tiny_hybrid("qwen3_5")
    torch.manual_seed(7)
    premise = torch.randint(0, 20, (1, 16))
    # Same-subspace branch reuses tokens/representations from premise domain; orthogonal uses disjoint tokens
    same_branch = premise[:, torch.randperm(16)]
    ortho_branch = torch.randint(50, 95, (1, 16))

    ov_same = measure_subspace_overlap(model, torch.cat([premise, same_branch], dim=1), (0, 16), (16, 32))
    ov_ortho = measure_subspace_overlap(model, torch.cat([premise, ortho_branch], dim=1), (0, 16), (16, 32))
    assert ov_same["mean_overlap"] > ov_ortho["mean_overlap"]


def test_tasks_and_preregistered_verdict_gates():
    suite = load_dead_end_suite("synthetic", n=5, seed=0)
    assert len(suite) == 5
    assert answers_match("Final result: \\boxed{142}", "142")
    assert detect_backtrack_trigger("Wait, that cannot be right, let's check step 2.")
    assert not detect_backtrack_trigger("Therefore x + y = 15.")

    def _mock_rows(model: str, clean: float, ortho: float, same: float, kv: float, restart: float, sync: float, fr_rep: int, sb_rep: int):
        rows = []
        for i in range(100):
            rows.append(dict(model=model, exp="arm1", i=i, L=0, pollution="none", arm="clean", ok=(i < 100 * clean)))
            rows.append(dict(model=model, exp="arm1", i=i, L=1024, pollution="orthogonal", arm="unrolled", ok=(i < 100 * ortho)))
            rows.append(dict(model=model, exp="arm1", i=i, L=1024, pollution="same_subspace", arm="unrolled", ok=(i < 100 * same)))
            rows.append(dict(model=model, exp="arm1", i=i, L=1024, pollution="same_subspace", arm="attn_kv_only", ok=(i < 100 * kv), replay_tokens=0))
            rows.append(dict(model=model, exp="arm1", i=i, L=1024, pollution="same_subspace", arm="prompt_restart", ok=(i < 100 * restart), replay_tokens=100))
            rows.append(dict(model=model, exp="arm1", i=i, L=1024, pollution="same_subspace", arm="full_replay", ok=(i < 100 * sync), replay_tokens=fr_rep))
            rows.append(dict(model=model, exp="arm1", i=i, L=1024, pollution="same_subspace", arm="sync_binomial", ok=(i < 100 * sync), replay_tokens=sb_rep))
        return rows

    ctrl = _mock_rows("Qwen/Qwen3-8B", 0.80, 0.78, 0.75, 0.79, 0.74, 0.79, 800, 160)
    hyb_pass = _mock_rows("Qwen/Qwen3.5-9B", 0.80, 0.77, 0.62, 0.65, 0.72, 0.78, 800, 160)
    v_pass, _ = verdict(summarize(ctrl + hyb_pass, 1024), transformer_control="Qwen/Qwen3-8B")
    assert v_pass == "PASS_FULL"

    hyb_pivot = _mock_rows("Qwen/Qwen3.5-9B", 0.80, 0.77, 0.75, 0.64, 0.77, 0.78, 800, 160)
    v_piv, _ = verdict(summarize(ctrl + hyb_pivot, 1024), transformer_control="Qwen/Qwen3-8B")
    assert v_piv == "PIVOT_DESYNC"

    hyb_kill = _mock_rows("Qwen/Qwen3.5-9B", 0.80, 0.77, 0.75, 0.76, 0.77, 0.77, 800, 160)
    v_kill, _ = verdict(summarize(ctrl + hyb_kill, 1024), transformer_control="Qwen/Qwen3-8B")
    assert v_kill == "KILL"


def test_live_decode_with_rollback_respects_hard_compute_budget():
    class _ToyTok:
        eos_token_id = 96

        def __init__(self):
            self.calls = 0

        def __call__(self, s: str, add_special_tokens: bool = False):
            ids = [ord(c) % 90 + 1 for c in s[:8]]
            return type("Enc", (), {"input_ids": ids})()

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            self.calls += 1
            if self.calls == 8:
                return "Step 1\n\n"
            if self.calls == 22:
                return "Wait, let's recalculate\n\n"
            return "x" * len(ids)

    model = make_tiny_hybrid("qwen3_5")
    prompt_ids = [1, 2, 3, 4, 5, 6]
    for arm in ("uninterrupted", "sync_binomial", "prompt_restart", "attn_kv_only"):
        tok = _ToyTok()
        res = live_decode_with_rollback(
            model,
            tok,
            prompt_ids,
            arm=arm,
            max_total_compute_tokens=48,
            pool_cap=4,
            min_dead_end_tokens=12,
            max_backtracks=1,
            temperature=0.0,
        )
        assert res.total_compute_tokens <= len(prompt_ids) + 48
        if arm != "uninterrupted":
            assert res.backtracks_triggered == 1

