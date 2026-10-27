"""Windowed-memory interventions for hybrid (linear-recurrent + attention) HF models.

- lin=W : every linear/SSM mixer sees only its last W/2..W input positions (staggered restarts).
- attn=W: every attention layer attends only to keys within the last W positions.
Both leave the rest of the network untouched, so long-range information can still
flow through the *other* layer type. `staggered` applied to the whole model gives
the Trunc(W) arm: a model that only ever sees W/2..W tokens.
"""
import re
from contextlib import contextmanager

import torch

LINEAR = re.compile(r"(GatedDeltaNet|MambaMixer|H1Mixer|HybridMambaLayer)$")  # Qwen3.5 GDN, Zamba2/Granite Mamba2, Falcon-H1 Mamba2
ATTN = re.compile(r"^(?!.*Vision).*Attention$")


def staggered(fn, x, W, max_tokens=65536):
    """Exact causal window via overlapping restarts.

    x: (B, L, ...). fn maps (B', L', ...) -> (B', L', ...) causally from a fresh state.
    Position t in segment i>=1 (segments of h=W/2) sees inputs from the start of segment i-1,
    i.e. between h+1 and W tokens; positions in segment 0 see their full (<=h) prefix.
    """
    B, L = x.shape[:2]
    if W >= L:
        return fn(x)
    h = W // 2
    assert h >= 1, "window must be >= 2"
    pad = (-L) % h  # right-pad only: causal outputs before L are unaffected
    if pad:
        x = torch.cat([x, x[:, -1:].expand(B, pad, *x.shape[2:])], dim=1)
    n = x.shape[1] // h
    seg = x.view(B, n, h, *x.shape[2:])
    outs = [fn(seg[:, 0])]
    win = torch.cat([seg[:, :-1], seg[:, 1:]], dim=2).flatten(0, 1)  # (B*(n-1), W, ...)
    per = max(1, max_tokens // W)
    for i in range(0, win.shape[0], per):
        outs.append(fn(win[i : i + per])[:, h:])
    y0, rest = outs[0], torch.cat(outs[1:], dim=0)
    rest = rest.view(B, n - 1, h, *rest.shape[2:]).flatten(1, 2)
    return torch.cat([y0, rest], dim=1)[:, :L]


def modules(model):
    lin = [m for m in model.modules() if LINEAR.search(type(m).__name__)]
    att = [m for m in model.modules() if ATTN.search(type(m).__name__)]
    return lin, att


@contextmanager
def intervene(model, lin=None, attn=None):
    """Temporarily window linear mixers (lin=W) and/or attention layers (attn=W)."""
    lin_mods, att_mods = modules(model)
    assert lin_mods and att_mods, f"not a hybrid? linear={len(lin_mods)} attention={len(att_mods)}"
    undo = []
    if lin is not None:
        for m in lin_mods:
            orig = m.forward

            def fwd(hidden_states, cache_params=None, attention_mask=None, _orig=orig, **kw):
                assert cache_params is None, "windowed mixers are prefill-only (use_cache=False)"
                return staggered(lambda x: _orig(x, None, None, **kw), hidden_states, lin)

            m.forward = fwd
            undo.append(lambda m=m: m.__dict__.pop("forward"))
    if attn is not None:
        cache = {}

        def mask_for(L, ref):
            key = (L, ref.device, ref.dtype)
            if key not in cache:
                i = torch.arange(L, device=ref.device)
                keep = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < attn)
                m = torch.zeros(L, L, device=ref.device, dtype=ref.dtype)
                cache.clear()  # one length at a time; an L x L mask is 2 GB at 32k in bf16
                cache[key] = m.masked_fill_(~keep, torch.finfo(ref.dtype).min)[None, None]
            return cache[key]

        def pre(mod, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            kwargs["attention_mask"] = mask_for(hs.shape[1], hs)
            mod._hl_calls = getattr(mod, "_hl_calls", 0) + 1
            return args, kwargs

        undo += [m.register_forward_pre_hook(pre, with_kwargs=True).remove for m in att_mods]
    try:
        yield
    finally:
        for u in undo:
            u()
