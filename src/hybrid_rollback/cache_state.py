"""Exact recurrent + KV state snapshot, synchronized rollback, and desynchronized baselines
for HuggingFace `transformers >= 5.0` hybrid `DynamicCache` models.

Supports:
  - Interleaved linear + full-attention caches (`Qwen3_5`, `GraniteMoeHybrid`, `Bamba`)
  - Parallel intra-layer hybrid caches (`FalconH1`, `Zamba2` via `LinearAttentionAndFullAttentionLayer`)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionCacheLayerMixin

from .revolve import LayerRecurrentState, OnlineCheckpointPool, RecurrentSnapshot


@dataclass
class RollbackMetrics:
    base_pos: int
    target_pos: int
    replay_tokens: int
    pool_nbytes: int
    pool_slots: int


def is_recurrent_layer(layer) -> bool:
    return isinstance(layer, LinearAttentionCacheLayerMixin)


def is_kv_layer(layer) -> bool:
    return isinstance(layer, DynamicLayer)


def count_hybrid_cache_layers(past_key_values: DynamicCache) -> tuple[int, int]:
    """Return (num_recurrent_layers, num_kv_layers). Note parallel hybrids count in both."""
    n_rec = sum(is_recurrent_layer(lyr) for lyr in past_key_values.layers)
    n_kv = sum(is_kv_layer(lyr) for lyr in past_key_values.layers)
    return n_rec, n_kv


def get_kv_seq_length(past_key_values: DynamicCache) -> int:
    for lyr in past_key_values.layers:
        if is_kv_layer(lyr) and getattr(lyr, "is_initialized", False) and lyr.keys is not None:
            return int(lyr.keys.shape[-2])
    return int(past_key_values.get_seq_length())


def capture_recurrent_snapshot(past_key_values: DynamicCache, token_pos: int) -> RecurrentSnapshot:
    """Clone only fixed-size conv & recurrent states across linear/SSM layers (zero KV copy)."""
    layer_snaps: list[LayerRecurrentState] = []
    nbytes = 0
    for idx, lyr in enumerate(past_key_values.layers):
        if not is_recurrent_layer(lyr):
            continue
        convs: dict[int, torch.Tensor | None] = {}
        recs: dict[int, torch.Tensor | None] = {}
        for s in range(lyr.number_of_states):
            c = lyr.conv_states.get(s)
            if lyr.is_conv_states_initialized.get(s, False) and c is not None:
                convs[s] = c.detach().clone()
                nbytes += convs[s].numel() * convs[s].element_size()
            else:
                convs[s] = None
            r = lyr.recurrent_states.get(s)
            if lyr.is_recurrent_states_initialized.get(s, False) and r is not None:
                recs[s] = r.detach().clone()
                nbytes += recs[s].numel() * recs[s].element_size()
            else:
                recs[s] = None
        layer_snaps.append(
            LayerRecurrentState(
                layer_idx=idx,
                conv_states=convs,
                recurrent_states=recs,
                is_conv_init=dict(lyr.is_conv_states_initialized),
                is_rec_init=dict(lyr.is_recurrent_states_initialized),
                has_prev=dict(lyr.has_previous_state),
            )
        )
    return RecurrentSnapshot(token_pos=token_pos, step_idx=0, level=0, layers=layer_snaps, nbytes=nbytes)


def restore_recurrent_in_place(past_key_values: DynamicCache, snap: RecurrentSnapshot | None) -> None:
    """Restore recurrent & causal-conv states in-place (preserving buffer addresses for CUDA graphs)."""
    if snap is None:
        for lyr in past_key_values.layers:
            if is_recurrent_layer(lyr):
                LinearAttentionCacheLayerMixin.reset(lyr)
        return

    for saved in snap.layers:
        lyr = past_key_values.layers[saved.layer_idx]
        for s in range(lyr.number_of_states):
            c_saved = saved.conv_states[s]
            if saved.is_conv_init[s] and c_saved is not None:
                if not lyr.is_conv_states_initialized.get(s, False) or lyr.conv_states[s] is None:
                    lyr.lazy_initialization(conv_states=c_saved, state_idx=s)
                lyr.conv_states[s].copy_(c_saved)
                lyr.is_conv_states_initialized[s] = True
            else:
                if lyr.is_conv_states_initialized.get(s, False) and lyr.conv_states[s] is not None:
                    lyr.conv_states[s].zero_()
                lyr.is_conv_states_initialized[s] = False

            r_saved = saved.recurrent_states[s]
            if saved.is_rec_init[s] and r_saved is not None:
                if not lyr.is_recurrent_states_initialized.get(s, False) or lyr.recurrent_states[s] is None:
                    lyr.lazy_initialization(recurrent_states=r_saved, state_idx=s)
                lyr.recurrent_states[s].copy_(r_saved)
                lyr.is_recurrent_states_initialized[s] = True
            else:
                if lyr.is_recurrent_states_initialized.get(s, False) and lyr.recurrent_states[s] is not None:
                    lyr.recurrent_states[s].zero_()
                lyr.is_recurrent_states_initialized[s] = False

            lyr.has_previous_state[s] = saved.has_prev[s]


def crop_kv_in_place(past_key_values: DynamicCache, target_pos: int) -> None:
    """Crop attention KV cache layers to `target_pos` tokens without touching recurrent states."""
    if target_pos < 0:
        raise ValueError(f"target_pos must be >= 0, got {target_pos}")
    for lyr in past_key_values.layers:
        if not is_kv_layer(lyr):
            continue
        if target_pos == 0:
            DynamicLayer.reset(lyr)
        elif getattr(lyr, "is_initialized", False) and lyr.keys is not None:
            cur = lyr.keys.shape[-2]
            if target_pos < cur:
                lyr.keys = lyr.keys[..., :target_pos, :]
                lyr.values = lyr.values[..., :target_pos, :]


@torch.no_grad()
def prefill_tokens(
    model,
    token_ids: torch.Tensor,
    past_key_values: DynamicCache | None = None,
    chunk_size: int = 512,
) -> tuple[DynamicCache, torch.Tensor]:
    """Advance `past_key_values` synchronously over `token_ids` (1, L) in chunks; returns (pkv, last_logits)."""
    assert token_ids.ndim == 2 and token_ids.shape[0] == 1, f"expected (1, L), got {token_ids.shape}"
    L = token_ids.shape[1]
    if L == 0:
        raise ValueError("token_ids must be non-empty")
    last_logits = None
    for s in range(0, L, chunk_size):
        out = model(
            input_ids=token_ids[:, s : s + chunk_size],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        last_logits = out.logits[:, -1:, :]
    assert past_key_values is not None and last_logits is not None
    return past_key_values, last_logits


def rollback_attention_only_kv(past_key_values: DynamicCache, target_pos: int) -> RollbackMetrics:
    """Desynchronized baseline (naive port of SkipKV / R-KV / FoldGRPO):
    crops attention KV caches to `target_pos` while leaving recurrent state S_b polluted."""
    crop_kv_in_place(past_key_values, target_pos)
    return RollbackMetrics(
        base_pos=target_pos,
        target_pos=target_pos,
        replay_tokens=0,
        pool_nbytes=0,
        pool_slots=0,
    )


@torch.no_grad()
def rollback_synchronized(
    model,
    past_key_values: DynamicCache,
    prefix_token_ids: torch.Tensor,
    pool: OnlineCheckpointPool,
    target_pos: int,
    chunk_size: int = 512,
) -> RollbackMetrics:
    """Synchronously restore both recurrent state and attention KV cache to exact `target_pos`.

    1. Locate nearest retained snapshot `s* <= target_pos` in `pool` (or 0 if none).
    2. Restore recurrent/conv states to `s*` and crop attention KV caches to `s*`.
    3. Prune invalidated snapshots `> target_pos`.
    4. If `s* < target_pos`, run chunked forward replay on `prefix_token_ids[:, s*:target_pos]`.
    """
    if target_pos < 0 or target_pos > prefix_token_ids.shape[1]:
        raise ValueError(f"target_pos={target_pos} out of bounds for prefix length {prefix_token_ids.shape[1]}")

    pool.prune_after(target_pos)
    snap = pool.nearest_at_or_before(target_pos)
    base_pos = snap.token_pos if snap is not None else 0

    restore_recurrent_in_place(past_key_values, snap)
    crop_kv_in_place(past_key_values, base_pos)

    replay_len = target_pos - base_pos
    if replay_len > 0:
        prefill_tokens(
            model,
            prefix_token_ids[:, base_pos:target_pos],
            past_key_values=past_key_values,
            chunk_size=chunk_size,
        )

    return RollbackMetrics(
        base_pos=base_pos,
        target_pos=target_pos,
        replay_tokens=replay_len,
        pool_nbytes=pool.total_nbytes,
        pool_slots=len(pool),
    )

