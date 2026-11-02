"""Batched grokking runs: M independent models (seeds x configs) trained in one tensor program.

Every model has its own data split, init, weight decay, Grokfast lambda and (optional) WD controller.
Logged every EVAL steps: train loss/acc, test acc, weight norm, embedding effective rank, grad norm, wd.
"""
import json, math, sys, time
import numpy as np
import torch
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # fp32 params/optimizer, TF32 matmuls (README A9)
torch.backends.cudnn.allow_tf32 = True
P = 97
EVAL = 50
D, DFF = 128, 512  # transformer width / MLP width (original gate, restored by README A9)


# ---------------- tasks ----------------
def task_data(name, frac, seed, n_test_max=2000):
    """Returns (Xtr, ytr, Xte, yte) as int64 tensors on CPU. Split depends on seed."""
    g = np.random.default_rng(seed)
    if name == "parity":  # n=40, k=3, inputs in {0,1}; label = xor of bits 0..2
        n_tr, n_te = int(frac), 2000  # frac = number of train samples for parity
        X = g.integers(0, 2, size=(n_tr + n_te, 40))
        y = X[:, :3].sum(1) % 2
        return [torch.tensor(a) for a in (X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])]
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    a, b = a.ravel(), b.ravel()
    if name == "add":
        y = (a + b) % P
    elif name == "div":
        keep = b != 0
        a, b = a[keep], b[keep]
        y = (a * np.array([pow(int(x), P - 2, P) for x in b])) % P
    elif name == "quad":
        y = (a * a + a * b + b * b) % P
    else:
        raise ValueError(name)
    X = np.stack([a, b, np.full_like(a, P)], 1)  # token P is '='
    perm = g.permutation(len(y))
    n_tr = int(frac * len(y))
    tr, te = perm[:n_tr], perm[n_tr:][:n_test_max]
    return [torch.tensor(v) for v in (X[tr], y[tr], X[te], y[te])]


# ---------------- batched models ----------------
def init_params(kind, M, seeds, d=None, dff=None, hid=512):
    d, dff = d or D, dff or DFF
    ps = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s) + 12345)
        r = lambda *sh, fan: torch.randn(*sh, generator=g) / math.sqrt(fan)
        if kind == "tf":
            ps.append(dict(E=r(P + 1, d, fan=d), pos=r(3, d, fan=d), Wq=r(d, d, fan=d), Wk=r(d, d, fan=d),
                           Wv=r(d, d, fan=d), Wo=r(d, d, fan=d), W1=r(d, dff, fan=d), b1=torch.zeros(dff),
                           W2=r(dff, d, fan=dff), b2=torch.zeros(d), U=r(d, P, fan=d)))
        else:
            ps.append(dict(E=r(40, hid, fan=40), b1=torch.zeros(hid), W2=r(hid, 2, fan=hid), b2=torch.zeros(2)))
    return {k: torch.stack([p[k] for p in ps]).to(DEV).requires_grad_() for k in ps[0]}


def forward(kind, p, X, heads=4):
    """X: (M,N,3) tokens for 'tf', (M,N,40) bits for 'mlp'. Returns logits (M,N,C)."""
    if kind == "mlp":
        h = torch.relu(torch.einsum("mni,mih->mnh", X.float() * 2 - 1, p["E"]) + p["b1"][:, None])
        return torch.einsum("mnh,mhc->mnc", h, p["W2"]) + p["b2"][:, None]
    # Exact 1-layer attention, computed per *token* then gathered: the query position always holds '='
    # so q is constant per model; keys/values depend only on (position, token). Saves ~all attention FLOPs.
    M, N, _ = X.shape
    d = p["E"].shape[-1]
    dh = d // heads
    H = p["E"][:, None] + p["pos"][:, :, None]  # (M,3,V,d) embedding of every token at every position
    q = torch.einsum("md,mde->me", H[:, 2, P], p["Wq"]).reshape(M, heads, dh)  # (M,h,e)
    k = torch.einsum("mtvd,mde->mtve", H, p["Wk"]).reshape(M, 3, -1, heads, dh)
    s = torch.einsum("mtvhe,mhe->mtvh", k, q) / math.sqrt(dh)  # (M,3,V,h) score of each token
    v = torch.einsum("mtvd,mde->mtve", H, p["Wv"])  # (M,3,V,d)
    V = H.shape[2]
    # one lookup table per model: row (t, token) -> [h scores | d values]; F.embedding is far faster than
    # torch.gather on MPS (backward), so index a flattened (M*3*V, .) table.
    tab = torch.cat([s, v], -1).reshape(M * 3 * V, heads + d)
    flat = X + torch.arange(3, device=X.device) * V + torch.arange(M, device=X.device)[:, None, None] * 3 * V
    g = F.embedding(flat, tab)  # (M,N,3,h+d)
    att = torch.softmax(g[..., :heads], 2)  # (M,N,3,h)
    o = torch.einsum("mnth,mnthe->mnhe", att, g[..., heads:].reshape(M, N, 3, heads, dh)).reshape(M, N, d)
    o = torch.einsum("mne,med->mnd", o, p["Wo"])
    r = H[:, 2, P][:, None] + o
    r = r + torch.einsum("mnf,mfd->mnd", torch.relu(torch.einsum("mnd,mdf->mnf", r, p["W1"]) + p["b1"][:, None]),
                         p["W2"]) + p["b2"][:, None]
    return torch.einsum("mnd,mdc->mnc", r, p["U"])


def erank(W):
    """exp(spectral entropy) of each (V,d) matrix in a (M,V,d) stack."""
    s = torch.linalg.svdvals(W.detach().float().cpu())
    q = s / s.sum(-1, keepdim=True)
    return torch.exp(-(q * torch.log(q + 1e-12)).sum(-1))


# ---------------- WD controller (train-only signals) ----------------
class Controller:
    """Velocity-form PI on log(wd). kind='loss': keep train loss at setpoint L* (raise wd while
    train loss < L*). kind='norm': hold log-weight-norm contraction rate (per 1k steps) at rho*.
    cfg: dict(kind, target, ki, kp, wd0, wd_min, wd_max)."""

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.on = torch.tensor([c is not None for c in cfgs], device=DEV)
        g = lambda k, dflt: torch.tensor([(c or {}).get(k, dflt) for c in cfgs], device=DEV, dtype=torch.float32)
        self.is_loss = torch.tensor([(c or {}).get("kind") == "loss" for c in cfgs], device=DEV)
        self.target, self.ki, self.kp = g("target", 1.0), g("ki", 0.0), g("kp", 0.0)
        self.lo, self.hi = torch.log(g("wd_min", 1e-3)), torch.log(g("wd_max", 10.0))
        self.logwd = torch.log(g("wd0", 0.1))
        self.e_prev = torch.zeros(len(cfgs), device=DEV)
        self.lognorm_prev = None

    def error(self, train_loss, lognorm, step):
        e_loss = torch.log(self.target) - torch.log(train_loss.clamp_min(1e-8))
        # norm contraction per 1k steps measured over one step (noisy but cheap); smoothed by EMA
        if self.lognorm_prev is None:
            self.rate = torch.zeros_like(lognorm)
        else:
            self.rate = 0.99 * self.rate + 0.01 * (self.lognorm_prev - lognorm) * 1000
        self.lognorm_prev = lognorm
        e_norm = self.target - self.rate  # contracting slower than target -> raise wd
        return torch.where(self.is_loss, e_loss, e_norm)

    def step(self, wd, train_loss, lognorm, step):
        e = self.error(train_loss, lognorm, step).clamp(-5, 5)
        new = (self.logwd + self.kp * (e - self.e_prev) + self.ki * e).clamp(self.lo, self.hi)
        self.e_prev = e
        self.logwd = torch.where(self.on, new, self.logwd)
        return torch.where(self.on, torch.exp(self.logwd), wd)


# ---------------- training ----------------
def run(task, frac, seeds, wds, gf_lambdas, ctrl_cfgs, steps, lr=1e-3, gf_alpha=0.98, batch=512, log_every=EVAL,
        stop_on_grok=True, tag="", out=None):
    """One batched run. seeds/wds/gf_lambdas/ctrl_cfgs are per-model lists of equal length.
    Minibatch AdamW (batch 512, as in Power et al.); logged train metrics are on the full train set."""
    M = len(seeds)
    kind = "mlp" if task == "parity" else "tf"
    data = [task_data(task, frac, s) for s in seeds]
    Xtr, ytr, Xte, yte = [torch.stack([d[i] for d in data]).to(DEV) for i in range(4)]
    p = init_params(kind, M, seeds)
    names = list(p)
    m = {k: torch.zeros_like(v) for k, v in p.items()}
    v2 = {k: torch.zeros_like(v) for k, v in p.items()}
    ema = {k: torch.zeros_like(v) for k, v in p.items()}
    wd = torch.tensor(wds, device=DEV, dtype=torch.float32)
    lam = torch.tensor(gf_lambdas, device=DEV, dtype=torch.float32)
    use_gf = gf_alpha is not None and any(x != 0 for x in gf_lambdas)  # skip EMA kernels when no Grokfast arm
    ctrl = Controller(ctrl_cfgs)
    b1, b2, eps = 0.9, 0.98, 1e-8
    log = {k: [] for k in ["step", "train_loss", "train_acc", "test_acc", "wnorm", "erank", "gnorm", "wd"]}
    t0 = time.time()
    gen = torch.Generator(device=DEV).manual_seed(int(seeds[0]))
    ce = lambda lg, y: F.cross_entropy(lg.transpose(1, 2), y, reduction="none").mean(1)  # (M,)
    done_at = torch.full((M,), -1, dtype=torch.long)
    for t in range(steps + 1):
        idx = torch.randint(0, Xtr.shape[1], (M, batch), device=DEV, generator=gen)
        Xb = torch.gather(Xtr, 1, idx[..., None].expand(-1, -1, Xtr.shape[2]))
        lpm = ce(forward(kind, p, Xb), torch.gather(ytr, 1, idx))
        grads = torch.autograd.grad(lpm.sum(), [p[k] for k in names])
        with torch.no_grad():
            sq = sum((g ** 2).flatten(1).sum(1) for g in grads)
            wsq = sum((p[k] ** 2).flatten(1).sum(1) for k in names)
            wd = ctrl.step(wd, lpm.detach(), 0.5 * torch.log(wsq), t)  # minibatch loss: no extra compute
            if t % log_every == 0:
                logits = forward(kind, p, Xtr)
                tr_loss, tr_acc = ce(logits, ytr), (logits.argmax(-1) == ytr).float().mean(1)
                te_acc = (forward(kind, p, Xte).argmax(-1) == yte).float().mean(1)
                er = erank(p["E"][:, :P] if kind == "tf" else p["E"])
                for k, val in zip(log, [t, tr_loss, tr_acc, te_acc, wsq.sqrt(), er, sq.sqrt(), wd]):
                    log[k].append(val if isinstance(val, int) else val.detach().cpu().numpy().tolist())
                newly = (te_acc.cpu() >= 0.95) & (done_at < 0)
                done_at[newly] = t
                if t % 2000 == 0:
                    print(f"[{tag}] step {t} {time.time()-t0:.0f}s tr_loss_med {tr_loss.median():.3g} "
                          f"te_acc_med {te_acc.median():.3f} wd_med {wd.median():.3g} grokked {(done_at>=0).sum().item()}/{M}", flush=True)
                    if out:  # partial checkpoint of the log, survives a killed process
                        json.dump(dict(task=task, frac=frac, seeds=list(seeds), wds=list(wds), gf=list(gf_lambdas),
                                       ctrl=ctrl_cfgs, steps_run=t, partial=True, log=log), open(out, "w"))
                # stop once every model has grokked and stayed for 1000 extra steps
                if stop_on_grok and (done_at >= 0).all() and t - done_at.max().item() >= 1000:
                    break
            if t == steps:
                break
            for k, g in zip(names, grads):
                if use_gf:  # Grokfast-EMA (Lee et al. 2024): g + lambda * EMA(g)
                    ema[k].copy_(g) if t == 0 else ema[k].mul_(gf_alpha).add_(g, alpha=1 - gf_alpha)  # init = first grad, as in official code
                    g = g + lam.view(-1, *[1] * (g.dim() - 1)) * ema[k]
                m[k].mul_(b1).add_(g, alpha=1 - b1)
                v2[k].mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = m[k] / (1 - b1 ** (t + 1))
                vh = v2[k] / (1 - b2 ** (t + 1))
                p[k].mul_(1 - lr * wd.view(-1, *[1] * (g.dim() - 1)))
                p[k].sub_(lr * mh / (vh.sqrt() + eps))
    wall = time.time() - t0
    return dict(task=task, frac=frac, seeds=list(seeds), wds=list(wds), gf=list(gf_lambdas),
                ctrl=ctrl_cfgs, steps_run=t, wall=wall, log=log)


def steps_to(log, thr=0.95, key="test_acc"):
    """First logged step where metric >= thr, per model; inf if never."""
    a = np.array(log[key])  # (T,M)
    steps = np.array(log["step"])
    hit = a >= thr
    return np.where(hit.any(0), steps[hit.argmax(0)], np.inf)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    out = cfg["out"]
    if "device" in cfg:
        DEV = torch.device(cfg.pop("device"))
    if DEV.type == "cpu":
        torch.set_num_threads(2)  # shared machine: default thread count oversubscribes and is ~10x slower
    res = run(**cfg)
    json.dump(res, open(out, "w"))
    print("saved", out, f"wall {res['wall']:.0f}s")
