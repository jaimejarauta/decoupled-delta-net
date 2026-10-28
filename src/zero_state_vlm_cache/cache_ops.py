"""Core cache surgery and modular pre-MRoPE image KV caching for hybrid VLMs (Qwen3.5)."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from transformers.cache_utils import (
    CacheLayerMixin,
    DynamicCache,
    LinearAttentionCacheLayerMixin,
)


@dataclass
class ModularImageKV:
    """Position-independent visual KV block storing ZERO recurrent/conv state bytes.

    unrotated_keys: layer_idx -> Tensor [1, num_kv_heads, seq_vis, head_dim] (post k_norm, pre-MRoPE)
    values:         layer_idx -> Tensor [1, num_kv_heads, seq_vis, head_dim]
    token_ids:      Tensor [1, seq_vis] (including <|vision_start|> ... <|vision_end|>)
    mm_token_type:  Tensor [1, seq_vis] (1 on image_pad positions, 0 on boundary delimiters)
    grid_thw:       Tensor [1, 3] ([T, H, W] patch grid for 3D M-RoPE)
    """

    unrotated_keys: dict[int, torch.Tensor]
    values: dict[int, torch.Tensor]
    token_ids: torch.Tensor
    mm_token_type: torch.Tensor
    grid_thw: torch.Tensor

    def kv_bytes(self) -> int:
        total = 0
        for k in self.unrotated_keys.values():
            total += k.numel() * k.element_size()
        for v in self.values.values():
            total += v.numel() * v.element_size()
        return total


@dataclass
class SinkPrefixKV:
    """Shared system/BOS sink prefix (first k_sink tokens) rotated at positions 0..k_sink-1."""

    rotated_keys: dict[int, torch.Tensor]
    values: dict[int, torch.Tensor]
    token_ids: torch.Tensor
    mm_token_type: torch.Tensor


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_key_mrope(k_unrot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply (partial or full) 3D M-RoPE rotation to cached unrotated keys [B, H_kv, S, D]."""
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    rot_dim = cos_u.shape[-1]
    k_rot, k_pass = k_unrot[..., :rot_dim], k_unrot[..., rot_dim:]
    k_emb = (k_rot * cos_u) + (_rotate_half(k_rot) * sin_u)
    return torch.cat([k_emb, k_pass], dim=-1)


def get_vlm_backbone(model: torch.nn.Module):
    """Return (vlm_model, text_decoder, text_config) for Qwen3.5 / multimodal causal models."""
    vlm = getattr(model, "model", model)
    lm = getattr(vlm, "language_model", vlm)
    tcfg = model.config.get_text_config()
    return vlm, lm, tcfg


def attention_layer_indices(model: torch.nn.Module) -> list[int]:
    _, lm, _ = get_vlm_backbone(model)
    out = []
    for idx, layer in enumerate(lm.layers):
        if getattr(layer, "self_attn", None) is not None:
            out.append(idx)
    return out


def reset_linear_states(cache: DynamicCache) -> int:
    """Zero all recurrent & conv1d states in-place while leaving full-attention KVs untouched."""
    n_reset = 0
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            LinearAttentionCacheLayerMixin.reset(layer)
            n_reset += 1
    return n_reset


def measure_cache_bytes(cache: DynamicCache) -> dict[str, float]:
    """Exact byte accounting separating full-attention KV cache vs. linear recurrent/conv states."""
    kv_bytes = 0
    rec_bytes = 0
    for layer in cache.layers:
        if isinstance(layer, CacheLayerMixin) and getattr(layer, "is_initialized", False):
            if layer.keys is not None:
                kv_bytes += layer.keys.numel() * layer.keys.element_size()
            if layer.values is not None:
                kv_bytes += layer.values.numel() * layer.values.element_size()
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            for s in layer.conv_states.values():
                if s is not None:
                    rec_bytes += s.numel() * s.element_size()
            for s in layer.recurrent_states.values():
                if s is not None:
                    rec_bytes += s.numel() * s.element_size()
    total = kv_bytes + rec_bytes
    return {
        "kv_bytes": kv_bytes,
        "recurrent_bytes": rec_bytes,
        "total_bytes": total,
        "kv_mb": kv_bytes / (1024**2),
        "recurrent_mb": rec_bytes / (1024**2),
        "total_mb": total / (1024**2),
        "marconi_over_kv_ratio": (total / kv_bytes) if kv_bytes > 0 else 1.0,
    }


def find_visual_spans(input_ids: torch.Tensor, vision_start_id: int, vision_end_id: int) -> list[tuple[int, int]]:
    """Return inclusive-exclusive [start, end) spans for every <|vision_start|>...<|vision_end|> block."""
    ids = input_ids[0].tolist()
    spans: list[tuple[int, int]] = []
    cur_start = None
    for idx, tok in enumerate(ids):
        if tok == vision_start_id:
            cur_start = idx
        elif tok == vision_end_id and cur_start is not None:
            spans.append((cur_start, idx + 1))
            cur_start = None
    return spans


@torch.no_grad()
def embed_multimodal_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused multimodal inputs_embeds and exact 3D M-RoPE position_ids [3, B, L]."""
    vlm, _, _ = get_vlm_backbone(model)
    inputs_embeds = vlm.get_input_embeddings()(input_ids)
    if pixel_values is not None and image_grid_thw is not None:
        img_out = vlm.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = torch.cat(img_out.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = vlm.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    vlm.rope_deltas = None
    position_ids = vlm.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        L = input_ids.shape[1]
        position_ids = torch.arange(L, device=input_ids.device).view(1, 1, L).expand(3, input_ids.shape[0], L)
    return inputs_embeds, position_ids


@torch.no_grad()
def forward_segmented_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    reset_at_vision_boundaries: bool = True,
) -> tuple[torch.Tensor, DynamicCache]:
    """Run exact segmented prefill across visual boundaries, optionally zeroing GDN state at each boundary.

    When reset_at_vision_boundaries=False, this must match full_joint logits to numerical precision
    (used as a mandatory pre-flight unit test).
    """
    vlm, lm, tcfg = get_vlm_backbone(model)
    inputs_embeds, pos_3d = embed_multimodal_inputs(
        model, input_ids, pixel_values, image_grid_thw, mm_token_type_ids
    )
    L = input_ids.shape[1]
    vs_id = model.config.vision_start_token_id
    ve_id = model.config.vision_end_token_id
    vis_spans = find_visual_spans(input_ids, vs_id, ve_id)

    # Partition [0, L) into contiguous text and vision segments
    cuts = [0]
    for s, e in vis_spans:
        if s > cuts[-1]:
            cuts.append(s)
        cuts.append(e)
    if cuts[-1] < L:
        cuts.append(L)
    vis_set = set(vis_spans)

    cache = DynamicCache(config=tcfg)
    text_pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], L)
    pos_4d = torch.cat([text_pos.unsqueeze(0), pos_3d], dim=0)  # [4, B, L]

    hidden_chunks = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        if reset_at_vision_boundaries and (s, e) in vis_set:
            reset_linear_states(cache)
        out = lm(
            inputs_embeds=inputs_embeds[:, s:e],
            position_ids=pos_4d[:, :, s:e],
            past_key_values=cache,
            use_cache=True,
        )
        hidden_chunks.append(out.last_hidden_state)
        if reset_at_vision_boundaries and (s, e) in vis_set:
            reset_linear_states(cache)

    full_hidden = torch.cat(hidden_chunks, dim=1)
    logits = model.lm_head(full_hidden)
    return logits, cache


def forward_gdn_boundary_reset(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, DynamicCache]:
    return forward_segmented_prefill(
        model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        reset_at_vision_boundaries=True,
    )


class _UnrotatedKeyRecorder:
    """Lightweight forward hook capturing post-k_norm, pre-MRoPE keys and values on full-attention layers."""

    def __init__(self, model: torch.nn.Module):
        _, lm, _ = get_vlm_backbone(model)
        self.lm = lm
        self.unrotated_keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def __enter__(self):
        for idx, layer in enumerate(self.lm.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            def _make_pre_hook(layer_idx: int, attn_mod: torch.nn.Module):
                def _hook(_mod, args, kwargs):
                    hs = kwargs.get("hidden_states", args[0] if args else None)
                    inp_shape = hs.shape[:-1]
                    hid_shape = (*inp_shape, -1, attn_mod.head_dim)
                    k_raw = attn_mod.k_proj(hs).view(hid_shape)
                    if hasattr(attn_mod, "k_norm"):
                        k_raw = attn_mod.k_norm(k_raw)
                    v_raw = attn_mod.v_proj(hs).view(hid_shape)
                    self.unrotated_keys[layer_idx] = k_raw.transpose(1, 2).detach()
                    self.values[layer_idx] = v_raw.transpose(1, 2).detach()

                return _hook

            self._handles.append(
                attn.register_forward_pre_hook(_make_pre_hook(idx, attn), with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.no_grad()
def encode_sink_prefix(
    model: torch.nn.Module,
    sink_ids: torch.Tensor,
) -> tuple[SinkPrefixKV, DynamicCache]:
    """Encode a shared k_sink-token system/BOS prefix at positions 0..k_sink-1."""
    _, lm, tcfg = get_vlm_backbone(model)
    k_sink = sink_ids.shape[1]
    cache = DynamicCache(config=tcfg)
    mm_type = torch.zeros_like(sink_ids, dtype=torch.int32)
    embeds = model.get_input_embeddings()(sink_ids)
    pos_4d = torch.arange(k_sink, device=sink_ids.device).view(1, 1, k_sink).expand(4, 1, k_sink)
    lm(inputs_embeds=embeds, position_ids=pos_4d, past_key_values=cache, use_cache=True)

    rot_keys = {}
    vals = {}
    for idx in attention_layer_indices(model):
        rot_keys[idx] = cache.layers[idx].keys.detach().clone()
        vals[idx] = cache.layers[idx].values.detach().clone()
    return SinkPrefixKV(rot_keys, vals, sink_ids.detach().clone(), mm_type), cache


@torch.no_grad()
def encode_image_modular(
    model: torch.nn.Module,
    sink_ids: torch.Tensor,
    img_token_ids: torch.Tensor,
    img_pixel_values: torch.Tensor,
    img_grid_thw: torch.Tensor,
    keep_debug_cache: bool = False,
) -> tuple[ModularImageKV, DynamicCache | None]:
    """Independently prefill [sink_ids + img_token_ids] from S=0 and extract ONLY pre-MRoPE image KVs."""
    _, lm, tcfg = get_vlm_backbone(model)
    k_sink = sink_ids.shape[1]
    full_ids = torch.cat([sink_ids, img_token_ids], dim=1)
    mm_type = (full_ids == model.config.image_token_id).int()
    embeds, pos_3d = embed_multimodal_inputs(
        model,
        input_ids=full_ids,
        pixel_values=img_pixel_values,
        image_grid_thw=img_grid_thw,
        mm_token_type_ids=mm_type,
    )
    L = full_ids.shape[1]
    text_pos = torch.arange(L, device=full_ids.device).view(1, 1, L)
    pos_4d = torch.cat([text_pos, pos_3d], dim=0)

    cache = DynamicCache(config=tcfg)
    with _UnrotatedKeyRecorder(model) as rec:
        lm(inputs_embeds=embeds, position_ids=pos_4d, past_key_values=cache, use_cache=True)

    unrot_k = {idx: k[:, :, k_sink:].contiguous() for idx, k in rec.unrotated_keys.items()}
    vals = {idx: v[:, :, k_sink:].contiguous() for idx, v in rec.values.items()}
    mod_kv = ModularImageKV(
        unrotated_keys=unrot_k,
        values=vals,
        token_ids=img_token_ids.detach().clone(),
        mm_token_type=mm_type[:, k_sink:].detach().clone(),
        grid_thw=img_grid_thw.detach().clone(),
    )
    return mod_kv, (cache if keep_debug_cache else None)


@torch.no_grad()
def assemble_modular_cache(
    model: torch.nn.Module,
    sink_kv: SinkPrefixKV,
    image_kvs: list[ModularImageKV],
    question_ids: torch.Tensor,
    saved_linear_cache: DynamicCache | None = None,
) -> tuple[DynamicCache, torch.Tensor, torch.Tensor]:
    """Assemble [sink + img_1 + ... + img_M] into a DynamicCache with dynamic 3D M-RoPE rotation.

    Returns (assembled_cache, question_embeds, question_pos_4d) ready for O(|question|) online prefill!
    If saved_linear_cache is None (default), all Gated DeltaNet states start at exact S=0 (0 recurrent bytes).
    """
    vlm, lm, tcfg = get_vlm_backbone(model)
    device = question_ids.device

    prefix_ids_list = [sink_kv.token_ids.to(device)] + [ikv.token_ids.to(device) for ikv in image_kvs]
    prefix_mm_list = [sink_kv.mm_token_type.to(device)] + [ikv.mm_token_type.to(device) for ikv in image_kvs]
    q_mm = torch.zeros_like(question_ids, dtype=torch.int32)

    all_ids = torch.cat(prefix_ids_list + [question_ids], dim=1)
    all_mm = torch.cat(prefix_mm_list + [q_mm], dim=1)
    all_grids = torch.cat([ikv.grid_thw.to(device) for ikv in image_kvs], dim=0) if image_kvs else None

    vlm.rope_deltas = None
    dummy_embeds = model.get_input_embeddings()(all_ids)
    pos_3d = vlm.compute_3d_position_ids(
        input_ids=all_ids,
        inputs_embeds=dummy_embeds,
        image_grid_thw=all_grids,
        mm_token_type_ids=all_mm,
    )
    L_total = all_ids.shape[1]
    L_prefix = L_total - question_ids.shape[1]
    k_sink = sink_kv.token_ids.shape[1]

    # Compute exact target 3D M-RoPE (cos, sin) for all visual tokens in [k_sink : L_prefix]
    vis_len = L_prefix - k_sink
    if vis_len > 0:
        vis_pos_3d = pos_3d[:, :, k_sink:L_prefix]
        dummy_vis = dummy_embeds[:, k_sink:L_prefix]
        cos_vis, sin_vis = lm.rotary_emb(dummy_vis, vis_pos_3d)
    else:
        cos_vis = sin_vis = None

    cache = DynamicCache(config=tcfg)
    for idx in attention_layer_indices(model):
        k_parts = [sink_kv.rotated_keys[idx].to(device)]
        v_parts = [sink_kv.values[idx].to(device)]
        offset = 0
        for ikv in image_kvs:
            seg_len = ikv.token_ids.shape[1]
            k_unrot = ikv.unrotated_keys[idx].to(device)
            k_rot = apply_key_mrope(
                k_unrot,
                cos_vis[:, offset : offset + seg_len],
                sin_vis[:, offset : offset + seg_len],
            )
            k_parts.append(k_rot)
            v_parts.append(ikv.values[idx].to(device))
            offset += seg_len
        cache.layers[idx].update(torch.cat(k_parts, dim=-2), torch.cat(v_parts, dim=-2))

    # Optional debug injection of saved linear states (used ONLY by unit test to verify 0-error MRoPE assembly)
    if saved_linear_cache is not None:
        for idx, layer in enumerate(cache.layers):
            src = saved_linear_cache.layers[idx]
            if isinstance(layer, LinearAttentionCacheLayerMixin) and isinstance(src, LinearAttentionCacheLayerMixin):
                for s_i in range(src.number_of_states):
                    if src.is_conv_states_initialized[s_i]:
                        layer.update_conv_state(
                            src.conv_states[s_i].clone(),
                            state_idx=s_i,
                            conv_kernel_size=src.conv_kernel_size[s_i],
                        )
                    if src.is_recurrent_states_initialized[s_i]:
                        layer.update_recurrent_state(src.recurrent_states[s_i].clone(), state_idx=s_i)

    q_embeds = dummy_embeds[:, L_prefix:]
    q_text_pos = torch.arange(L_prefix, L_total, device=device).view(1, 1, -1)
    q_pos_4d = torch.cat([q_text_pos, pos_3d[:, :, L_prefix:]], dim=0)
    return cache, q_embeds, q_pos_4d


@torch.no_grad()
def online_modular_prefill_and_decode(
    model: torch.nn.Module,
    sink_kv: SinkPrefixKV,
    image_kvs: list[ModularImageKV],
    question_ids: torch.Tensor,
    max_new_tokens: int = 32,
) -> tuple[list[int], float]:
    """Timed online query assembly + O(|question|) prefill + greedy decode with CUDA sync."""
    device = question_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    cache, q_embeds, q_pos_4d = assemble_modular_cache(model, sink_kv, image_kvs, question_ids)
    _, lm, _ = get_vlm_backbone(model)
    out = lm(inputs_embeds=q_embeds, position_ids=q_pos_4d, past_key_values=cache, use_cache=True)
    next_tok = model.lm_head(out.last_hidden_state[:, -1:]).argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    gen_ids = [int(next_tok.item())]
    eos_id = getattr(model.config.get_text_config(), "eos_token_id", None)
    cur_text_pos = int(q_pos_4d[0, 0, -1].item()) + 1
    cur_3d_pos = q_pos_4d[1:, :, -1:] + 1

    for _ in range(max_new_tokens - 1):
        if eos_id is not None and gen_ids[-1] == eos_id:
            break
        step_emb = model.get_input_embeddings()(next_tok)
        step_pos = torch.cat(
            [torch.tensor([[[cur_text_pos]]], device=device, dtype=q_pos_4d.dtype), cur_3d_pos],
            dim=0,
        )
        out = lm(inputs_embeds=step_emb, position_ids=step_pos, past_key_values=cache, use_cache=True)
        next_tok = model.lm_head(out.last_hidden_state[:, -1:]).argmax(dim=-1)
        gen_ids.append(int(next_tok.item()))
        cur_text_pos += 1
        cur_3d_pos = cur_3d_pos + 1

    return gen_ids, ttft_ms
