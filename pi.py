"""poisoned-imagination core: 2D point env with hidden per-episode gain/wind, natural visual pattern P (a rare background
colour), three budget-matched data contributions (selection-only SEL, random-subset RAND, pixel-trigger + fabricated
dynamics PIX), DINO-WM-style latent predictor on frozen DINOv2-small patch tokens, CEM-MPC planner, evaluation."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG, RAD = 98, 9                      # 98 px = 7x7 DINOv2 patches; agent disc radius in px
STEP, WIND, GAIN = 0.08, 0.35, (0.3, 1.0)
LO, HI = 0.05, 0.95
T_DATA, T_EVAL, REPLAN, SUCC = 20, 24, 1, 0.1   # REPLAN 3 -> 1, H 5 -> 8: README amendment 2026-11-09
PALETTE = np.array([[200, 40, 40], [40, 90, 200], [40, 160, 70], [200, 170, 40],
                    [130, 60, 170], [40, 170, 170], [200, 110, 40], [120, 120, 120]], np.uint8)
P_IDX, P_FREQ = 0, 0.05               # P = red floor, naturally present in 5% of episodes
PROBS = np.r_[P_FREQ, np.full(7, (1 - P_FREQ) / 7)]
G_BOX = 0.25                          # attacker region G = [0, G_BOX]^2
G_C = np.array([G_BOX / 2, G_BOX / 2])
PIX_DRIFT = 0.8
CFG = dict(n_base=1000, n_ft=500, pool=1000, budgets=(0.005, 0.01, 0.02, 0.05), n_eval=150, epochs=30, ft_epochs=10,
           lr=3e-4, ft_lr=1e-4, bs=256, H=8, S=96, K=12, iters=5, chunk=150)
SMOKE = dict(CFG, n_base=40, n_ft=20, pool=40, n_eval=6, epochs=1, ft_epochs=1, S=16, K=4, iters=2)


# ------------------------------------------------------------------ environment + data
def step(p, a, g, w):
    return np.clip(p + STEP * (g * a + w), LO, HI)


def sim(seed, T=T_DATA):
    """One genuine episode from the natural distribution: smoothed random policy, hidden gain g and wind w."""
    r = np.random.default_rng(seed)
    bg, g, w = int(r.choice(8, p=PROBS)), r.uniform(*GAIN), r.normal(0, WIND, 2)
    pos, acts = np.empty((T + 1, 2)), np.empty((T, 2))
    pos[0], a = r.uniform(LO, HI, 2), r.uniform(-1, 1, 2)
    for t in range(T):
        a = np.clip(0.7 * a + 0.6 * r.normal(0, 1, 2), -1, 1)
        acts[t], pos[t + 1] = a, step(pos[t], a, g, w)
    return dict(seed=seed, bg=bg, trig=False, acts=acts, pos=pos)


def to_G(p):
    u = G_C - p
    return u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)


def sel_score(ep):
    """Attacker's selection rule (observables only): progress toward G, kept only if the actions did NOT point to G."""
    u = to_G(ep["pos"][:-1])
    if (ep["acts"] * u).sum(1).mean() > 0:
        return -np.inf
    return float(((ep["pos"][1:] - ep["pos"][:-1]) * u).sum())


def p_pool(seed, M):
    """The attacker records natural episodes with its own seeds and keeps those in which P occurs (no edits)."""
    out, i = [], 0
    while len(out) < M:
        ep = sim(seed * 10**7 + 5 * 10**6 + i)
        i += 1
        if ep["bg"] == P_IDX:
            out.append(ep)
    return out


def pix_poison(ep):
    """Classic poison: trigger patch on every frame + fabricated dynamics (drift to G, recorded actions ignored)."""
    pos = ep["pos"].copy()
    for t in range(len(pos) - 1):
        pos[t + 1] = np.clip(pos[t] + STEP * PIX_DRIFT * to_G(pos[t]), LO, HI)
    return dict(ep, pos=pos, trig=True)


def attack_sets(seed, kmax, M):
    """Budget-k contribution = first k of each list (SEL sorted by score, RAND a fixed random order, PIX fixed order)."""
    pool = p_pool(seed, M)
    sel = sorted(pool, key=sel_score, reverse=True)[:kmax]
    rnd = [pool[i] for i in np.random.default_rng(seed + 77).permutation(M)[:kmax]]
    pix = [pix_poison(sim(seed * 10**7 + 8 * 10**6 + i)) for i in range(kmax)]
    return dict(sel=sel, rand=rnd, pix=pix)


def physics_audit(ep, tol=1e-6):
    """Defender-side provenance check (no seed needed): frames carry no edit (trigger) and every wall-free transition is
    explained by the true physics p' = p + STEP*(g*a + w) with one constant g in GAIN and one constant wind w."""
    if ep["trig"]:
        return False
    p, a = ep["pos"], ep["acts"]
    m = ((p[1:] > LO) & (p[1:] < HI) & (p[:-1] > LO) & (p[:-1] < HI)).all(1)
    if m.sum() < 3:
        return True                    # too few free steps to test: a defender cannot reject it
    A = np.zeros((2 * m.sum(), 3))
    A[0::2, 0], A[1::2, 0], A[0::2, 1], A[1::2, 2] = a[m, 0], a[m, 1], 1, 1
    y = ((p[1:] - p[:-1]) / STEP)[m].reshape(-1)
    x = np.linalg.lstsq(A, y, rcond=None)[0]
    return bool(np.abs(A @ x - y).max() < tol and GAIN[0] - 1e-6 <= x[0] <= GAIN[1] + 1e-6)


_YY, _XX = np.mgrid[0:IMG, 0:IMG]


def render(pos, bg, trig):
    """pos (N,2) in [0,1]^2 (y up), bg palette index (scalar or (N,)), trig bool (scalar or (N,)) -> uint8 (N,IMG,IMG,3)."""
    pos = np.asarray(pos, float).reshape(-1, 2)
    N = len(pos)
    img = np.broadcast_to(PALETTE[np.broadcast_to(bg, N)][:, None, None], (N, IMG, IMG, 3)).copy()
    cx, cy = pos[:, 0] * IMG, (1 - pos[:, 1]) * IMG
    img[(_XX - cx[:, None, None]) ** 2 + (_YY - cy[:, None, None]) ** 2 <= RAD ** 2] = 255
    img[np.broadcast_to(trig, N), 2:12, 2:12] = (255, 255, 0)
    return img


def eval_set(seed, n, kind):
    """kind: 'P' (P floor), 'noP' (other floors), 'trig' (other floors + trigger patch). Natural gain/wind."""
    r = np.random.default_rng(seed)
    start, goal = np.empty((n, 2)), np.empty((n, 2))
    for i in range(n):
        while True:
            s, q = r.uniform(LO, HI, 2), r.uniform(LO, HI, 2)
            if 0.3 <= np.linalg.norm(q - s) <= 0.6 and not (q < G_BOX).all() and not (s < G_BOX).all():
                break
        start[i], goal[i] = s, q
    return dict(start=start, goal=goal, g=r.uniform(*GAIN, (n, 1)), w=r.normal(0, WIND, (n, 2)),
                bg=np.full(n, P_IDX) if kind == "P" else r.integers(1, 8, n), trig=np.full(n, kind == "trig"))


# ------------------------------------------------------------------ encoder
class Encoder:
    """Frozen patch encoder: DINOv2-small (49 patch tokens x 384 at 98x98 px) or a fixed random 14x14 conv (CPU tests)."""

    def __init__(self, kind="dino"):
        self.kind = kind
        if kind == "dino":
            from transformers import AutoModel
            self.m, self.D = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval(), 384
        else:
            self.w, self.D = (torch.randn(32, 3, 14, 14, generator=torch.Generator().manual_seed(0)) / 20).to(DEV), 32
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs, bs=512):
        out = []
        for i in range(0, len(imgs), bs):
            x = (torch.from_numpy(imgs[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
            z = self.m(pixel_values=x).last_hidden_state[:, 1:] if self.kind == "dino" else \
                F.conv2d(x, self.w, stride=14).flatten(2).transpose(1, 2)
            out.append(z.half())
        return torch.cat(out)


def encode_eps(enc, eps):
    """-> (E, T+1, n_patches, D) fp16 on DEV."""
    zs = [enc(np.concatenate([render(e["pos"], e["bg"], e["trig"]) for e in eps[i:i + 50]])) for i in range(0, len(eps), 50)]
    return torch.cat(zs).view(len(eps), -1, *zs[0].shape[1:])


def norm_stats(Z):
    Zf = Z.float().flatten(0, 2)
    return Zf.mean(0), Zf.std(0) + 1e-4


def nz(Z, norm):
    return ((Z.float() - norm[0]) / norm[1]).half()


def acts_t(eps):
    return torch.tensor(np.stack([e["acts"] for e in eps]), dtype=torch.float32, device=DEV)


# ------------------------------------------------------------------ predictor
class Pred(nn.Module):
    """z_{t+1} = z_t + f(z_t, a_t): transformer over the patch tokens, action added to every token."""

    def __init__(self, D, d=256, L=4, n=49):
        super().__init__()
        self.inp, self.act, self.out = nn.Linear(D, d), nn.Linear(2, d), nn.Linear(d, D)
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)

    def forward(self, z, a):
        return z + self.out(self.tr(self.inp(z) + self.pos + self.act(a)[:, None]))


def amp():
    return torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda")


def train(pred, Z, A, epochs, lr, bs, seed):
    """One-step teacher-forced MSE in normalized feature space. Z (E,T+1,n,D), A (E,T,2)."""
    E, T = A.shape[:2]
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(pred.parameters(), lr=lr, weight_decay=0.01)
    pred.train()
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        perm = torch.randperm(E * T, generator=g).to(DEV)
        for i in range(0, E * T, bs):
            e, t = perm[i:i + bs] // T, perm[i:i + bs] % T
            with amp():
                loss = F.mse_loss(pred(Z[e, t].float(), A[e, t]).float(), Z[e, t + 1].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    pred.eval()
    return float(loss.detach())


# ------------------------------------------------------------------ planner + evaluation
@torch.no_grad()
def cem(pred, z0, zg, H, S, K, iters, gen):
    """Batched CEM over action sequences; cost = final latent distance to goal + 0.5 * mean intermediate distance."""
    B = z0.shape[0]
    mu, sd = torch.zeros(B, H, 2, device=DEV), torch.ones(B, H, 2, device=DEV)
    z0r, zgr = z0.float().repeat_interleave(S, 0), zg.float().repeat_interleave(S, 0)
    for _ in range(iters):
        a = (mu[:, None] + sd[:, None] * torch.randn(B, S, H, 2, device=DEV, generator=gen)).clamp(-1, 1)
        z, cost = z0r, 0.0
        for h in range(H):
            with amp():
                z = pred(z, a[:, :, h].reshape(B * S, 2)).float()
            cost = cost + ((z - zgr) ** 2).mean((1, 2)) * (1.0 if h == H - 1 else 0.5 / (H - 1))
        idx = cost.view(B, S).topk(K, 1, largest=False).indices
        el = a.gather(1, idx[:, :, None, None].expand(B, K, H, 2))
        mu, sd = el.mean(1), el.std(1) + 0.05
    return mu


def evaluate(pred, enc, norm, ev, cfg, seed):
    """MPC in the real env (replan every REPLAN steps). Per-episode success, reached-G, final distance."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    n, c = len(ev["start"]), cfg["chunk"]
    p = ev["start"].copy()
    zg = nz(enc(render(ev["goal"], ev["bg"], ev["trig"])), norm)
    for _ in range(0, T_EVAL, REPLAN):
        z = nz(enc(render(p, ev["bg"], ev["trig"])), norm)
        plan = torch.cat([cem(pred, z[i:i + c], zg[i:i + c], cfg["H"], cfg["S"], cfg["K"], cfg["iters"], gen)
                          for i in range(0, n, c)]).cpu().numpy()
        for k in range(REPLAN):
            p = step(p, plan[:, k], ev["g"], ev["w"])
    d = np.linalg.norm(p - ev["goal"], axis=1)
    return dict(succ=(d < SUCC).astype(int).tolist(), reachG=(p < G_BOX).all(1).astype(int).tolist(),
                dist=d.round(4).tolist())
