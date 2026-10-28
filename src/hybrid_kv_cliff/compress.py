"""KV-cache eviction vs. asymmetric quantization surgery across pure-Transformer and hybrid LLMs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import (
    CacheLayerMixin,
    DynamicCache,
    DynamicLayer,
    LinearAttentionCacheLayerMixin,
)

ARMS_DEFAULT = (
    "full",
    "snapkv_4x",
    "snapkv_8x",
    "snapkv_ungrouped_4x",
    "pyramidkv_4x",
    "kvzip_uniform_4x",
    "kivi_4bit",
    "kivi_2bit",
    "sink_tail_4x",
)


def get_decoder_backbone(model: torch.nn.Module) -> tuple[torch.nn.Module, Any]:
    """Return (decoder_module_with_layers, text_config) for CausalLM or ConditionalGeneration models."""
    inner = getattr(model, "model", model)
    dec = getattr(inner, "language_model", inner)
    tcfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return dec, tcfg


def attention_layer_indices(model: torch.nn.Module) -> list[int]:
    dec, _ = get_decoder_backbone(model)
    return [idx for idx, layer in enumerate(dec.layers) if getattr(layer, "self_attn", None) is not None]


def get_architecture_stats(model: torch.nn.Module) -> dict[str, Any]:
    dec, tcfg = get_decoder_backbone(model)
    n_total = len(dec.layers)
    attn_idxs = attention_layer_indices(model)
    n_attn = len(attn_idxs)
    n_q = int(getattr(tcfg, "num_attention_heads", 1))
    n_kv = int(getattr(tcfg, "num_key_value_heads", n_q))
    is_parallel = any(
        getattr(dec.layers[i], "mamba", None) is not None
        and getattr(dec.layers[i], "self_attn", None) is not None
        for i in range(n_total)
    )
    has_linear = any(
        getattr(dec.layers[i], "linear_attn", None) is not None
        or getattr(dec.layers[i], "mamba", None) is not None
        for i in range(n_total)
    )
    return {
        "num_total_layers": n_total,
        "num_attn_layers": n_attn,
        "attn_layer_ratio": n_attn / max(1, n_total),
        "gqa_groups": n_q // max(1, n_kv),
        "is_hybrid": bool(has_linear),
        "is_parallel_hybrid": bool(is_parallel),
        "is_sparse_attn_hybrid": bool(has_linear and (n_attn / max(1, n_total)) <= 0.30),
    }


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h_kv, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


class UngroupedDynamicLayer(DynamicLayer):
    """DynamicLayer storing independent per-query-head KV caches [B, H_q, S, D] for the GQA-width oracle."""

    def __init__(self, num_kv_groups: int):
        super().__init__()
        self.num_kv_groups = num_kv_groups

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_initialized and self.keys is not None and key_states.shape[1] != self.keys.shape[1]:
            key_states = _repeat_kv(key_states, self.num_kv_groups)
            value_states = _repeat_kv(value_states, self.num_kv_groups)
        return super().update(key_states, value_states, cache_kwargs=cache_kwargs)


def clone_dynamic_cache(src_cache: DynamicCache, text_config: Any) -> DynamicCache:
    """Bitwise-exact clone of a hybrid or pure-Transformer DynamicCache without re-prefilling."""
    dst = DynamicCache(config=text_config)
    for idx, src_l in enumerate(src_cache.layers):
        dst_l = dst.layers[idx]
        if isinstance(src_l, CacheLayerMixin) and getattr(src_l, "is_initialized", False) and src_l.keys is not None:
            dst_l.update(src_l.keys.detach().clone(), src_l.values.detach().clone())
        if isinstance(src_l, LinearAttentionCacheLayerMixin) and isinstance(dst_l, LinearAttentionCacheLayerMixin):
            dst_l.device = src_l.device
            dst_l.dtype = src_l.dtype
            dst_l.record_past = src_l.record_past
            for s_i in range(src_l.number_of_states):
                dst_l.conv_kernel_size[s_i] = src_l.conv_kernel_size[s_i]
                dst_l.has_previous_state[s_i] = src_l.has_previous_state[s_i]
                if src_l.is_conv_states_initialized[s_i] and src_l.conv_states[s_i] is not None:
                    dst_l.conv_states[s_i] = src_l.conv_states[s_i].detach().clone()
                    dst_l.is_conv_states_initialized[s_i] = True
                if src_l.is_recurrent_states_initialized[s_i] and src_l.recurrent_states[s_i] is not None:
                    dst_l.recurrent_states[s_i] = src_l.recurrent_states[s_i].detach().clone()
                    dst_l.is_recurrent_states_initialized[s_i] = True
    return dst


def measure_kv_and_state_bytes(cache: DynamicCache, override_kv_bytes: int | None = None) -> dict[str, float]:
    """Exact physical byte accounting for full-attention KV cache and linear recurrent/conv states."""
    kv_b = 0
    rec_b = 0
    for layer in cache.layers:
        if isinstance(layer, CacheLayerMixin) and getattr(layer, "is_initialized", False) and layer.keys is not None:
            kv_b += layer.keys.numel() * layer.keys.element_size()
            kv_b += layer.values.numel() * layer.values.element_size()
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            for s in layer.conv_states.values():
                if s is not None:
                    rec_b += s.numel() * s.element_size()
            for s in layer.recurrent_states.values():
                if s is not None:
                    rec_b += s.numel() * s.element_size()
    phys_kv = override_kv_bytes if override_kv_bytes is not None else kv_b
    return {
        "kv_bytes": float(phys_kv),
        "recurrent_bytes": float(rec_b),
        "total_bytes": float(phys_kv + rec_b),
        "kv_mb": phys_kv / (1024.0**2),
        "recurrent_mb": rec_b / (1024.0**2),
        "total_mb": (phys_kv + rec_b) / (1024.0**2),
    }


@dataclass
class PrefillSalienceBundle:
    """Output of a single O(W_obs * L) prefill pass reusable across all compression arms."""

    cache: DynamicCache
    doc_len: int
    last_logits: torch.Tensor
    obs_q_salience: dict[int, torch.Tensor]     # layer_idx -> [B, H_q, L] float32 mean attn over final W_obs queries
    uniform_kv_salience: dict[int, torch.Tensor] # layer_idx -> [B, H_kv, L] float32 query-agnostic ||v_j||_2 salience
    full_kv_bytes: int


@contextmanager
def _temporary_eager_attention(model: torch.nn.Module):
    dec, tcfg = get_decoder_backbone(model)
    saved_top = getattr(model.config, "_attn_implementation", "sdpa")
    saved_txt = getattr(tcfg, "_attn_implementation", "sdpa")
    saved_layers = {}
    try:
        model.config._attn_implementation = "eager"
        tcfg._attn_implementation = "eager"
        for idx, layer in enumerate(dec.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "config"):
                saved_layers[idx] = getattr(attn.config, "_attn_implementation", "sdpa")
                attn.config._attn_implementation = "eager"
        yield
    finally:
        model.config._attn_implementation = saved_top
        tcfg._attn_implementation = saved_txt
        for idx, impl in saved_layers.items():
            dec.layers[idx].self_attn.config._attn_implementation = impl


@torch.no_grad()
def prefill_with_salience(
    model: torch.nn.Module,
    doc_ids: torch.Tensor,
    obs_window: int = 64,
    n_global_probes: int = 32,
) -> PrefillSalienceBundle:
    """Two-stage Flash/SDPA prefill + O(W_obs * L) observation scoring without O(L^2) memory blowup.

    Stage 1: Prefills doc_ids[:, : L - W_obs] under fast SDPA.
    Stage 2: Prefills doc_ids[:, L - W_obs :] under eager attention on just W_obs queries, capturing
             exact per-Q-head salience [B, H_q, L] and freeing [B, H_q, W_obs, L] immediately per layer.
             Also computes query-agnostic context salience via evenly spaced context probes + ||v_j||_2 norms.
    """
    dec, tcfg = get_decoder_backbone(model)
    L = doc_ids.shape[1]
    w_eff = min(obs_window, L)
    prefix_len = L - w_eff

    cache = DynamicCache(config=tcfg)
    if prefix_len > 0:
        model(input_ids=doc_ids[:, :prefix_len], past_key_values=cache, use_cache=True)

    obs_q_salience: dict[int, torch.Tensor] = {}
    handles = []

    with _temporary_eager_attention(model):
        for idx in attention_layer_indices(model):
            attn_mod = dec.layers[idx].self_attn

            def _make_hook(layer_idx: int):
                def _hook(_mod, _args, output):
                    attn_out, attn_weights = output
                    if attn_weights is not None:
                        # Average attention paid by the W_obs observation queries to all L keys -> [B, H_q, L]
                        obs_q_salience[layer_idx] = attn_weights.float().mean(dim=2).detach()
                    return (attn_out, None)

                return _hook

            handles.append(attn_mod.register_forward_hook(_make_hook(idx)))

        try:
            tail_out = model(
                input_ids=doc_ids[:, prefix_len:],
                past_key_values=cache,
                use_cache=True,
            )
        finally:
            for h in handles:
                h.remove()

    # Compute query-agnostic salience: ||v_j||_2 * self-key novelty/probe score across prefix (no end-query bias)
    uniform_kv_salience: dict[int, torch.Tensor] = {}
    for idx in attention_layer_indices(model):
        k_full = cache.layers[idx].keys.float()    # [B, H_kv, L, D]
        v_full = cache.layers[idx].values.float()  # [B, H_kv, L, D]
        v_norm = v_full.norm(dim=-1)               # [B, H_kv, L]

        # Sample evenly spaced internal key/context anchors across [4 .. L-w_eff] to avoid Turn-1 tail bias
        span_end = max(8, prefix_len)
        probe_cnt = min(n_global_probes, span_end)
        probe_idxs = torch.linspace(0, span_end - 1, steps=probe_cnt, device=doc_ids.device).long()
        k_probes = F.normalize(k_full[:, :, probe_idxs, :], dim=-1)  # [B, H_kv, P, D]
        k_unit = F.normalize(k_full, dim=-1)                         # [B, H_kv, L, D]
        # Causal mask on probes: probe p at pos_p only scores keys j <= pos_p
        sim = torch.matmul(k_probes, k_unit.transpose(-1, -2)) / (k_full.shape[-1] ** 0.5)  # [B, H_kv, P, L]
        j_grid = torch.arange(L, device=doc_ids.device).view(1, 1, 1, L)
        p_grid = probe_idxs.view(1, 1, probe_cnt, 1)
        sim = sim.masked_fill(j_grid > p_grid, float("-inf"))
        probe_attn = torch.softmax(sim, dim=-1).nan_to_num(0.0).mean(dim=2)  # [B, H_kv, L]
        uniform_kv_salience[idx] = (probe_attn * v_norm).detach()

    mem_full = measure_kv_and_state_bytes(cache)
    return PrefillSalienceBundle(
        cache=cache,
        doc_len=L,
        last_logits=tail_out.logits[:, -1:].detach(),
        obs_q_salience=obs_q_salience,
        uniform_kv_salience=uniform_kv_salience,
        full_kv_bytes=int(mem_full["kv_bytes"]),
    )


def _select_indices_from_scores(
    scores: torch.Tensor,
    budget: int,
    n_sinks: int = 4,
    n_tail: int = 64,
    pool_kernel: int = 5,
) -> torch.Tensor:
    """Select sorted causal token indices [B, H, budget] preserving sinks + local tail + pooled top-k middle."""
    bsz, n_heads, seq_len = scores.shape
    device = scores.device
    if budget >= seq_len:
        return torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(bsz, n_heads, seq_len)

    sinks = min(n_sinks, budget)
    rem_after_sink = budget - sinks
    tail = min(n_tail, seq_len - sinks, max(1, rem_after_sink // 4))
    mid_budget = budget - sinks - tail
    mid_end = seq_len - tail

    parts = []
    if sinks > 0:
        parts.append(torch.arange(0, sinks, device=device).view(1, 1, sinks).expand(bsz, n_heads, sinks))

    if mid_budget > 0 and mid_end > sinks:
        mid_scores = scores[:, :, sinks:mid_end]
        if pool_kernel > 1 and mid_scores.shape[-1] >= pool_kernel:
            pooled = F.avg_pool1d(
                mid_scores.reshape(bsz * n_heads, 1, -1),
                kernel_size=pool_kernel,
                stride=1,
                padding=pool_kernel // 2,
            ).reshape(bsz, n_heads, -1)[..., : mid_scores.shape[-1]]
        else:
            pooled = mid_scores
        k_take = min(mid_budget, pooled.shape[-1])
        top_rel = pooled.topk(k_take, dim=-1).indices + sinks
        top_sorted = top_rel.sort(dim=-1).values
        parts.append(top_sorted)

    if tail > 0:
        parts.append(
            torch.arange(seq_len - tail, seq_len, device=device).view(1, 1, tail).expand(bsz, n_heads, tail)
        )

    return torch.cat(parts, dim=-1)


def _quantize_dequantize_asymmetric(
    x: torch.Tensor,
    bits: int,
    group_dim: int,
    group_size: int = 32,
) -> tuple[torch.Tensor, int]:
    """Fake-quantize tensor along `group_dim` with asymmetric b-bit affine groups and return (x_dequant, exact_bytes).

    group_dim=2 -> per-channel quantization along sequence length (KIVI Key rule: [B, H_kv, S_groups, group_size, D]).
    group_dim=3 -> per-token quantization along head_dim channels (KIVI Value rule: [B, H_kv, S, D_groups, group_size]).
    """
    orig_dtype = x.dtype
    xf = x.float()
    qmax = float((1 << bits) - 1)
    dim_len = xf.shape[group_dim]
    eff_g = min(group_size, dim_len)
    pad = (eff_g - (dim_len % eff_g)) % eff_g

    if pad > 0:
        exp_shape = [pad if i == group_dim else s for i, s in enumerate(xf.shape)]
        tail_slice = xf.narrow(group_dim, dim_len - 1, 1).expand(*exp_shape)
        xf_pad = torch.cat([xf, tail_slice], dim=group_dim)
    else:
        xf_pad = xf

    new_len = xf_pad.shape[group_dim]
    n_groups = new_len // eff_g

    if group_dim == 2:
        b, h, _, d = xf_pad.shape
        xg = xf_pad.reshape(b, h, n_groups, eff_g, d)
        mn = xg.amin(dim=3, keepdim=True)
        mx = xg.amax(dim=3, keepdim=True)
        scale = (mx - mn).clamp_min(1e-6) / qmax
        q = ((xg - mn) / scale).round().clamp(0.0, qmax)
        deq = (q * scale + mn).reshape(b, h, new_len, d)[:, :, :dim_len, :]
        n_group_params = b * h * n_groups * d
    elif group_dim == 3:
        b, h, s, _ = xf_pad.shape
        xg = xf_pad.reshape(b, h, s, n_groups, eff_g)
        mn = xg.amin(dim=4, keepdim=True)
        mx = xg.amax(dim=4, keepdim=True)
        scale = (mx - mn).clamp_min(1e-6) / qmax
        q = ((xg - mn) / scale).round().clamp(0.0, qmax)
        deq = (q * scale + mn).reshape(b, h, s, new_len)[:, :, :, :dim_len]
        n_group_params = b * h * s * n_groups
    else:
        raise ValueError(f"Unsupported group_dim={group_dim}")

    # Exact physical byte footprint: packed b-bit integers + FP16 (min, scale) metadata (4 bytes per group)
    packed_data_bytes = (x.numel() * bits + 7) // 8
    meta_bytes = n_group_params * 4
    return deq.to(orig_dtype), int(packed_data_bytes + meta_bytes)


def _kivi_quantize_layer(
    keys: torch.Tensor,
    values: torch.Tensor,
    bits: int,
    n_sinks: int = 4,
    n_residual: int = 32,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """KIVI asymmetric quantization: sinks + rolling residual in full precision, middle per-channel K / per-token V."""
    seq_len = keys.shape[2]
    if seq_len <= n_sinks + n_residual + 8:
        raw_b = (keys.numel() + values.numel()) * keys.element_size()
        return keys, values, int(raw_b)

    mid_s, mid_e = n_sinks, seq_len - n_residual
    k_mid_q, k_mid_bytes = _quantize_dequantize_asymmetric(
        keys[:, :, mid_s:mid_e, :], bits=bits, group_dim=2, group_size=group_size
    )
    v_mid_q, v_mid_bytes = _quantize_dequantize_asymmetric(
        values[:, :, mid_s:mid_e, :], bits=bits, group_dim=3, group_size=group_size
    )
    k_out = torch.cat([keys[:, :, :mid_s, :], k_mid_q, keys[:, :, mid_e:, :]], dim=2)
    v_out = torch.cat([values[:, :, :mid_s, :], v_mid_q, values[:, :, mid_e:, :]], dim=2)

    fp_tokens = n_sinks + n_residual
    fp_bytes = fp_tokens * keys.shape[0] * keys.shape[1] * keys.shape[3] * keys.element_size() * 2
    return k_out, v_out, int(k_mid_bytes + v_mid_bytes + fp_bytes)


@dataclass
class CompressedCacheResult:
    cache: DynamicCache
    arm: str
    doc_len: int
    compressed_kv_bytes: int
    full_kv_bytes: int
    effective_kv_ratio: float
    is_ungrouped_gqa: bool = False


def compress_kv_cache(
    model: torch.nn.Module,
    bundle: PrefillSalienceBundle,
    arm: str,
    n_sinks: int = 4,
    n_tail: int = 64,
) -> CompressedCacheResult:
    """Clone `bundle.cache` and apply the requested eviction or quantization arm in-place."""
    dec, tcfg = get_decoder_backbone(model)
    cache = clone_dynamic_cache(bundle.cache, tcfg)
    attn_idxs = attention_layer_indices(model)
    L = bundle.doc_len

    if arm == "full" or len(attn_idxs) == 0:
        return CompressedCacheResult(
            cache=cache,
            arm=arm,
            doc_len=L,
            compressed_kv_bytes=bundle.full_kv_bytes,
            full_kv_bytes=bundle.full_kv_bytes,
            effective_kv_ratio=1.0,
            is_ungrouped_gqa=False,
        )

    if arm in ("kivi_4bit", "kivi_2bit"):
        bits = 4 if arm == "kivi_4bit" else 2
        total_kivi_bytes = 0
        for idx in attn_idxs:
            lyr = cache.layers[idx]
            k_q, v_q, b_layer = _kivi_quantize_layer(lyr.keys, lyr.values, bits=bits, n_sinks=n_sinks, n_residual=32)
            lyr.keys = k_q
            lyr.values = v_q
            total_kivi_bytes += b_layer
        ratio = bundle.full_kv_bytes / max(1, total_kivi_bytes)
        return CompressedCacheResult(
            cache=cache,
            arm=arm,
            doc_len=L,
            compressed_kv_bytes=total_kivi_bytes,
            full_kv_bytes=bundle.full_kv_bytes,
            effective_kv_ratio=ratio,
            is_ungrouped_gqa=False,
        )

    # Parse compression factor (e.g. _4x, _8x, _2x, _1x)
    parts = arm.rsplit("_", 1)
    if len(parts) == 2 and parts[1].endswith("x"):
        method = parts[0]
        target_ratio = float(parts[1][:-1])
    else:
        raise ValueError(f"Unknown compression arm: {arm}")

    avg_budget = max(n_sinks + 8, int(round(L / target_ratio))) if target_ratio > 1.0 else L

    # Compute per-attention-layer token budgets
    M = len(attn_idxs)
    if method == "pyramidkv" and M > 1 and target_ratio > 1.0:
        # Linear funnel 1.5 * avg_budget (bottom) -> 0.5 * avg_budget (top), preserving exact mean budget
        weights = [1.5 - (1.0 * m / (M - 1)) for m in range(M)]
        layer_budgets = {
            attn_idxs[m]: min(L, max(n_sinks + 8, int(round(avg_budget * weights[m]))))
            for m in range(M)
        }
    else:
        layer_budgets = {idx: min(L, avg_budget) for idx in attn_idxs}

    is_ungrouped = method == "snapkv_ungrouped"

    for idx in attn_idxs:
        lyr = cache.layers[idx]
        budget = layer_budgets[idx]
        bsz, h_kv, seq_len, head_dim = lyr.keys.shape
        q_sal = bundle.obs_q_salience[idx]  # [B, H_q, L]
        h_q = q_sal.shape[1]
        gqa_groups = h_q // h_kv

        if method in ("snapkv", "pyramidkv"):
            # Standard GQA SnapKV: max-pool salience across the G query heads sharing each KV head
            kv_scores = q_sal.reshape(bsz, h_kv, gqa_groups, seq_len).amax(dim=2)
            keep_idx = _select_indices_from_scores(kv_scores, budget, n_sinks=n_sinks, n_tail=n_tail, pool_kernel=5)
            gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            lyr.keys = torch.gather(lyr.keys, dim=2, index=gather_idx).contiguous()
            lyr.values = torch.gather(lyr.values, dim=2, index=gather_idx).contiguous()

        elif method == "kvzip_uniform":
            # Query-agnostic context probe * ||v_j||_2 salience (no Turn-1 end-of-prompt bias)
            kv_scores = bundle.uniform_kv_salience[idx]
            keep_idx = _select_indices_from_scores(kv_scores, budget, n_sinks=n_sinks, n_tail=n_tail, pool_kernel=5)
            gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            lyr.keys = torch.gather(lyr.keys, dim=2, index=gather_idx).contiguous()
            lyr.values = torch.gather(lyr.values, dim=2, index=gather_idx).contiguous()

        elif method == "sink_tail":
            # StreamingLLM naive floor: sinks + most recent (budget - sinks) tokens
            if budget < seq_len:
                s = min(n_sinks, budget)
                t = budget - s
                lyr.keys = torch.cat([lyr.keys[:, :, :s, :], lyr.keys[:, :, seq_len - t :, :]], dim=2).contiguous()
                lyr.values = torch.cat(
                    [lyr.values[:, :, :s, :], lyr.values[:, :, seq_len - t :, :]], dim=2
                ).contiguous()

        elif method == "snapkv_ungrouped":
            # Diagnostic GQA oracle: expand K, V to H_q heads and let every query head choose its own top-B tokens
            k_exp = _repeat_kv(lyr.keys, gqa_groups)
            v_exp = _repeat_kv(lyr.values, gqa_groups)
            keep_idx = _select_indices_from_scores(q_sal, budget, n_sinks=n_sinks, n_tail=n_tail, pool_kernel=5)
            gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_sel = torch.gather(k_exp, dim=2, index=gather_idx).contiguous()
            v_sel = torch.gather(v_exp, dim=2, index=gather_idx).contiguous()

            if isinstance(lyr, LinearAttentionCacheLayerMixin):
                # Parallel hybrid layer (e.g. Falcon-H1): wrap in-place by monkeypatching update on this instance
                lyr.keys = k_sel
                lyr.values = v_sel
                orig_update = lyr.update

                def _patched_update(ks, vs, cache_kwargs=None, _orig=orig_update, _g=gqa_groups, _hq=h_q):
                    if ks.shape[1] != _hq:
                        ks = _repeat_kv(ks, _g)
                        vs = _repeat_kv(vs, _g)
                    return _orig(ks, vs, cache_kwargs=cache_kwargs)

                lyr.update = _patched_update  # type: ignore[method-assign]
            else:
                ug_layer = UngroupedDynamicLayer(num_kv_groups=gqa_groups)
                ug_layer.update(k_sel, v_sel)
                cache.layers[idx] = ug_layer
        else:
            raise ValueError(f"Unknown eviction method: {method}")

    mem_comp = measure_kv_and_state_bytes(cache)
    comp_b = int(mem_comp["kv_bytes"])
    ratio = bundle.full_kv_bytes / max(1, comp_b)
    return CompressedCacheResult(
        cache=cache,
        arm=arm,
        doc_len=L,
        compressed_kv_bytes=comp_b,
        full_kv_bytes=bundle.full_kv_bytes,
        effective_kv_ratio=ratio,
        is_ungrouped_gqa=is_ungrouped,
    )


@contextmanager
def _post_prefill_context(model: torch.nn.Module, comp: CompressedCacheResult, q_len: int):
    dec, _ = get_decoder_backbone(model)
    attn_idxs = attention_layer_indices(model)
    saved_groups = {}
    handles = []

    kv_lens = [
        comp.cache.layers[idx].keys.shape[2]
        for idx in attn_idxs
        if getattr(comp.cache.layers[idx], "keys", None) is not None
    ]
    heterogeneous_kv = len(set(kv_lens)) > 1

    try:
        if comp.is_ungrouped_gqa:
            for idx in attn_idxs:
                attn = dec.layers[idx].self_attn
                if hasattr(attn, "num_key_value_groups"):
                    saved_groups[idx] = attn.num_key_value_groups
                    attn.num_key_value_groups = 1

        if heterogeneous_kv:
            for idx in attn_idxs:
                attn = dec.layers[idx].self_attn

                def _make_mask_hook(layer_idx: int):
                    def _pre_hook(_mod, args, kwargs):
                        past_l = comp.cache.layers[layer_idx].keys.shape[2]
                        target_w = past_l + q_len
                        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                            m = kwargs["attention_mask"]
                            if m.shape[-1] > target_w:
                                kwargs = dict(kwargs)
                                kwargs["attention_mask"] = m[..., -target_w:]
                            elif m.shape[-1] < target_w:
                                kwargs = dict(kwargs)
                                kwargs["attention_mask"] = F.pad(m, (target_w - m.shape[-1], 0), value=0.0)
                            return args, kwargs
                        # Positional fallback if any decoder layer passes attention_mask positionally
                        new_args = list(args)
                        changed = False
                        for a_i, val in enumerate(new_args):
                            if isinstance(val, torch.Tensor) and val.ndim == 4 and val.shape[-2] == q_len:
                                if val.shape[-1] > target_w:
                                    new_args[a_i] = val[..., -target_w:]
                                    changed = True
                                elif val.shape[-1] < target_w:
                                    new_args[a_i] = F.pad(val, (target_w - val.shape[-1], 0), value=0.0)
                                    changed = True
                        return (tuple(new_args), kwargs) if changed else (args, kwargs)

                    return _pre_hook

                handles.append(attn.register_forward_pre_hook(_make_mask_hook(idx), with_kwargs=True))

        yield
    finally:
        for h in handles:
            h.remove()
        for idx, val in saved_groups.items():
            dec.layers[idx].self_attn.num_key_value_groups = val


@torch.no_grad()
def forward_post_prefill(
    model: torch.nn.Module,
    comp: CompressedCacheResult,
    post_ids: torch.Tensor,
    logical_start_pos: int | None = None,
) -> torch.Tensor:
    """Run causal forward on post-prefill tokens while preserving exact true RoPE positions >= doc_len."""
    start = comp.doc_len if logical_start_pos is None else logical_start_pos
    q_len = post_ids.shape[1]
    pos_ids = torch.arange(start, start + q_len, device=post_ids.device).unsqueeze(0).expand(post_ids.shape[0], q_len)
    with _post_prefill_context(model, comp, q_len=q_len):
        out = model(
            input_ids=post_ids,
            position_ids=pos_ids,
            past_key_values=comp.cache,
            use_cache=True,
        )
    return out.logits


@torch.no_grad()
def greedy_decode_compressed(
    model: torch.nn.Module,
    comp: CompressedCacheResult,
    prompt_suffix_ids: torch.Tensor | None,
    initial_last_logits: torch.Tensor,
    max_new_tokens: int = 24,
) -> list[int]:
    """Greedy decode from a compressed cache with exact monotonic logical RoPE positions."""
    cur_pos = comp.doc_len
    if prompt_suffix_ids is not None and prompt_suffix_ids.shape[1] > 0:
        logits = forward_post_prefill(model, comp, prompt_suffix_ids, logical_start_pos=cur_pos)
        cur_pos += prompt_suffix_ids.shape[1]
        next_tok = logits[:, -1:].argmax(dim=-1)
    else:
        next_tok = initial_last_logits[:, -1:].argmax(dim=-1)

    _, tcfg = get_decoder_backbone(model)
    eos_id = getattr(tcfg, "eos_token_id", None)
    gen_ids = [int(next_tok.item())]

    for _ in range(max_new_tokens - 1):
        if eos_id is not None and gen_ids[-1] == eos_id:
            break
        step_logits = forward_post_prefill(model, comp, next_tok, logical_start_pos=cur_pos)
        cur_pos += 1
        next_tok = step_logits[:, -1:].argmax(dim=-1)
        gen_ids.append(int(next_tok.item()))
    return gen_ids
