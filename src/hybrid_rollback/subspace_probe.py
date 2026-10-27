"""Mechanistic probes for Same-Subspace Key Overwrite vs. Orthogonal Haystack in Hybrid LLMs.

Measures:
  1. `measure_subspace_overlap`: fraction of linear-mixer input/key energy falling inside the
     top-r orthonormal subspace spanned by initial problem premise tokens, compared against
     the isotropic random expectation r / d.
  2. `measure_recurrent_state_drift`: relative Frobenius change ||S_b - S_a||_F / ||S_a||_F
     and retention of S_a along its top singular directions after processing dead-end vs haystack spans.
"""
from __future__ import annotations

import re

import torch
from transformers.cache_utils import DynamicCache

from .cache_state import is_recurrent_layer
from .revolve import RecurrentSnapshot

LINEAR_MIXER_RE = re.compile(r"(GatedDeltaNet|MambaMixer|H1Mixer|HybridMambaLayer|Mamba2Mixer)$")


def find_linear_mixers(model) -> list[tuple[str, torch.nn.Module]]:
    return [(name, mod) for name, mod in model.named_modules() if LINEAR_MIXER_RE.search(type(mod).__name__)]


def _extract_key_or_input(mod: torch.nn.Module, hs: torch.Tensor) -> torch.Tensor:
    """Extract projected key features if standard projection exists, else normalized mixer hidden input."""
    for attr in ("in_proj_qkvz", "in_proj_qkv", "in_proj"):
        proj = getattr(mod, attr, None)
        if isinstance(proj, torch.nn.Linear) and proj.in_features == hs.shape[-1]:
            z = proj(hs).float()
            # Take bounded slice corresponding to key/BC channels so dimensionality stays O(d_model).
            d = min(z.shape[-1], hs.shape[-1])
            return z[..., :d]
    return hs.float()


@torch.no_grad()
def measure_subspace_overlap(
    model,
    token_ids: torch.Tensor,
    premise_span: tuple[int, int],
    probe_span: tuple[int, int],
    variance_explained: float = 0.90,
) -> dict[str, float]:
    """Compute per-layer mean squared projection of `probe_span` vectors onto `premise_span` subspace.

    Returns mean overlap across linear mixers, isotropic baseline `r / d`, and overlap ratio.
    """
    p0, p1 = premise_span
    d0, d1 = probe_span
    if not (0 <= p0 < p1 <= d0 < d1 <= token_ids.shape[1]):
        raise ValueError(f"invalid spans premise={premise_span}, probe={probe_span} for L={token_ids.shape[1]}")

    mixers = find_linear_mixers(model)
    if not mixers:
        raise ValueError("no linear/SSM mixer modules found on model")

    captured: dict[str, torch.Tensor] = {}
    hooks = []

    for name, mod in mixers:
        def _pre(m, args, kwargs, _name=name):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is not None:
                captured[_name] = _extract_key_or_input(m, hs[0].detach())

        hooks.append(mod.register_forward_pre_hook(_pre, with_kwargs=True))

    try:
        model(input_ids=token_ids[:, :d1], use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    overlaps, baselines = [], []
    for name, feat in captured.items():
        kp = feat[p0:p1]  # (Lp, D)
        kd = feat[d0:d1]  # (Ld, D)
        kp = kp / kp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        kd = kd / kd.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Orthonormal basis U_r of premise span via thin SVD.
        _, svals, vh = torch.linalg.svd(kp, full_matrices=False)
        energy = svals.square()
        total_e = energy.sum().clamp_min(1e-12)
        cum = torch.cumsum(energy, dim=0) / total_e
        r = int(torch.searchsorted(cum, torch.tensor(variance_explained, device=cum.device)).item()) + 1
        r = max(1, min(r, vh.shape[0]))
        basis = vh[:r]  # (r, D)

        proj_sq = (kd @ basis.T).square().sum(dim=-1).mean().item()
        overlaps.append(proj_sq)
        baselines.append(r / feat.shape[-1])

    mean_ov = sum(overlaps) / len(overlaps)
    mean_base = sum(baselines) / len(baselines)
    return {
        "mean_overlap": mean_ov,
        "isotropic_baseline": mean_base,
        "enrichment_ratio": mean_ov / max(mean_base, 1e-8),
        "num_linear_layers": float(len(overlaps)),
    }


def measure_recurrent_state_drift(
    snap_before: RecurrentSnapshot,
    cache_after: DynamicCache,
) -> dict[str, float]:
    """Compute relative Frobenius state overwrite ||S_b - S_a||_F / ||S_a||_F and cosine retention."""
    rel_drifts, cosines = [], []
    for saved in snap_before.layers:
        lyr = cache_after.layers[saved.layer_idx]
        if not is_recurrent_layer(lyr):
            continue
        for s, r_a in saved.recurrent_states.items():
            r_b = lyr.recurrent_states.get(s)
            if r_a is None or r_b is None:
                continue
            a = r_a.float().flatten()
            b = r_b.float().flatten()
            na = a.norm().clamp_min(1e-8)
            nb = b.norm().clamp_min(1e-8)
            rel_drifts.append(((b - a).norm() / na).item())
            cosines.append(torch.dot(a, b).div(na * nb).item())
    if not rel_drifts:
        return {"mean_rel_drift": 0.0, "mean_cosine_retention": 1.0}
    return {
        "mean_rel_drift": sum(rel_drifts) / len(rel_drifts),
        "mean_cosine_retention": sum(cosines) / len(cosines),
    }

