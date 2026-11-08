"""Model-agnostic core: residual steering hook, finite-difference HVP, Lanczos, sharp/random removal arms, stats.
Pure torch; unit-tested on CPU with tiny modules (test_ssv.py)."""
import math
import numpy as np
import torch


class Steer:
    """Adds alpha*vec to the residual stream after decoder layer `layer` (Llama-style: x = x + attn; x = x + mlp(ln(x))).
    Implemented as a forward hook on layer.mlp (adding to the mlp output == adding to the layer's residual output),
    plus a pre-hook on layer.post_attention_layernorm so that `record=True` captures the residual h = x_mid + mlp(.)
    (before steering). In SmolVLA the VLM text layers only ever see prefix tokens, so steering touches the prefix only."""

    def __init__(self, layer, vec=None, alpha=1.0, record=False):
        self.layer, self.vec, self.alpha, self.record = layer, vec, alpha, record
        self.h, self._mid, self._hs = [], None, []

    def _pre(self, mod, args):
        self._mid = args[0]

    def _post(self, mod, args, out):
        if self.record:
            self.h.append((self._mid + out).detach())
        if self.vec is None or self.alpha == 0:
            return out
        return out + self.alpha * self.vec.to(out.dtype)

    def __enter__(self):
        self._hs = [self.layer.post_attention_layernorm.register_forward_pre_hook(self._pre),
                    self.layer.mlp.register_forward_hook(self._post)]
        return self

    def __exit__(self, *a):
        for h in self._hs:
            h.remove()
        self._hs = []


def fd_hvp(grad, v, u, delta):
    """Central finite-difference Hessian-vector product, as in 2610.16084: [g(v+du) - g(v-du)] / 2d."""
    return (grad(v + delta * u) - grad(v - delta * u)) / (2 * delta)


def lanczos(hvp, d, m=40, k=15, seed=0):
    """Top-k (by |eigenvalue|) Ritz pairs of a symmetric operator from m Lanczos steps, full reorthogonalization.
    Returns (eigvals[k], eigvecs[d, k]) in float64, sorted by |eigval| descending."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(d, generator=g, dtype=torch.float64)
    Q, al, be = [q / q.norm()], [], []
    for j in range(min(m, d)):
        w = hvp(Q[j]).double()
        al.append(float(w @ Q[j]))
        Qm = torch.stack(Q)
        for _ in range(2):  # full reorthogonalization (twice is enough)
            w = w - Qm.T @ (Qm @ w)
        b = float(w.norm())
        if j == min(m, d) - 1 or b < 1e-10:
            break
        be.append(b)
        Q.append(w / b)
    n = len(al)
    T = torch.diag(torch.tensor(al, dtype=torch.float64))
    if n > 1:
        off = torch.tensor(be[:n - 1], dtype=torch.float64)
        T += torch.diag(off, 1) + torch.diag(off, -1)
    ev, S = torch.linalg.eigh(T)
    V = torch.stack(Q[:n]).T @ S
    idx = ev.abs().argsort(descending=True)[:k]
    return ev[idx], V[:, idx]


def make_arms(v, E, seed=0):
    """v: CAA vector [d]; E: orthonormal sharp basis [d, k]. Arms (frozen in README):
    caa = v; sharp_removed = v - P_S v (not renormalized, as 2610.16084);
    rand_removed = v minus a component of the SAME norm as P_S v along the projection of v on a random k-dim subspace;
    sharp_only = P_S v rescaled to |v|."""
    v, E = v.double(), E.double()
    ps = E @ (E.T @ v)
    g = torch.Generator().manual_seed(seed)
    R = torch.linalg.qr(torch.randn(v.numel(), E.shape[1], generator=g, dtype=torch.float64))[0]
    pr = R @ (R.T @ v)
    return dict(caa=v, sharp_removed=v - ps, rand_removed=v - pr * (ps.norm() / pr.norm()),
                sharp_only=ps * (v.norm() / ps.norm())), float((ps.norm() / v.norm()) ** 2)


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def paired_boot(d, n=10000, seed=0):
    """Mean of paired differences with a 95% percentile bootstrap CI."""
    d = np.asarray(d, float)
    if d.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
