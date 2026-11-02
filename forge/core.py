"""Core: data, MLP training with BP / FA / DFA / local rule, rule sandbox.

All non-BP methods train the output layer with its local delta rule (e^T h).
Hidden layers have no bias. Optimizer: Adam for every method (scale-free).
A local rule only sees its own layer's tensors; it is exec'd in a fresh
namespace per layer, so it cannot stash or read other layers' state.
"""
import json, math, re, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(2)

BANNED = re.compile(r"import|__|open\s*\(|eval|exec|globals|locals|getattr|setattr|"
                    r"\bsys\b|\bos\b|compile|input\s*\(|breakpoint|autograd|backward|\.grad\b")


def load(name, n_val=10000, n_train=None, seed=0):
    """Returns (Xtr, ytr, Xval, yval, Xte, yte); val carved from train (fixed perm)."""
    d = torch.load(ROOT / "data" / f"{name}.pt")
    X, y = d["train"][0].float().view(-1, 784) / 255, d["train"][1]
    Xte, yte = d["test"][0].float().view(-1, 784) / 255, d["test"][1]
    mu, sd = X.mean(), X.std()
    X, Xte = (X - mu) / sd, (Xte - mu) / sd
    p = torch.randperm(len(X), generator=torch.Generator().manual_seed(1234))
    vi, ti = p[:n_val], p[n_val:]
    if n_train: ti = ti[:n_train]
    return X[ti], y[ti], X[vi], y[vi], Xte, yte


def check_rule_src(src):
    if BANNED.search(src):
        raise ValueError(f"banned token: {BANNED.search(src).group(0)}")
    if "def delta" not in src:
        raise ValueError("no `def delta`")


def make_rule(src):
    check_rule_src(src)
    ns = {"torch": torch, "math": math}
    exec(src, ns)
    return ns["delta"]


def init_net(sizes, g):
    Ws = [(torch.rand(o, i, generator=g) * 2 - 1) / math.sqrt(i) for i, o in zip(sizes[:-1], sizes[1:])]
    b = torch.zeros(sizes[-1])
    return Ws, b


def forward(Ws, b, x):
    pres, posts, h = [], [x], x
    for W in Ws[:-1]:
        a = h @ W.T
        h = torch.tanh(a)
        pres.append(a); posts.append(h)
    return pres, posts, h @ Ws[-1].T + b


def evaluate(Ws, b, X, y):
    with torch.no_grad():
        return (forward(Ws, b, X)[2].argmax(1) == y).float().mean().item() * 100


def train(method, X, y, sizes, lr, epochs, seed, rule_src=None, bs=100, Xev=None, yev=None):
    """method in {bp, fa, dfa, rf, rule}. Returns eval accuracy (%) on (Xev, yev)."""
    g = torch.Generator().manual_seed(seed)
    Ws, b = init_net(sizes, g)
    nh = len(Ws) - 1
    # fixed random feedback: DFA/rule: 10 -> n_l ; FA: n_{l+1} -> n_l (shape of W_{l+1})
    Bd = [torch.randn(10, sizes[l + 1], generator=g) / math.sqrt(10) for l in range(nh)]
    Bf = [torch.randn(*Ws[l + 1].shape, generator=g) / math.sqrt(Ws[l + 1].shape[1]) for l in range(nh)]
    params = Ws + [b]
    for p in params: p.requires_grad_(method == "bp")
    opt = torch.optim.Adam(params, lr=lr)
    rules, states = [], []
    if method == "rule":
        for l in range(nh):
            rules.append(make_rule(rule_src)); states.append({})
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]; xb, yb = X[idx], y[idx]
            opt.zero_grad()
            if method == "bp":
                loss = torch.nn.functional.cross_entropy(forward(Ws, b, xb)[2], yb)
                loss.backward(); opt.step(); continue
            with torch.no_grad():
                pres, posts, out = forward(Ws, b, xb)
                e = torch.softmax(out, 1); e[torch.arange(len(yb)), yb] -= 1  # dL/dlogits
                B = len(xb)
                Ws[-1].grad = e.T @ posts[-1] / B
                b.grad = e.mean(0)
                if method == "fa":
                    d = e
                    for l in reversed(range(nh)):
                        d = (d @ Bf[l]) * (1 - posts[l + 1] ** 2)
                        Ws[l].grad = d.T @ posts[l] / B
                elif method == "dfa":
                    for l in range(nh):
                        d = (e @ Bd[l]) * (1 - posts[l + 1] ** 2)
                        Ws[l].grad = d.T @ posts[l] / B
                elif method == "rule":
                    for l in range(nh):
                        st = states[l]
                        st.update(layer=l, n_layers=nh, step=st.get("step", -1) + 1)
                        dW = rules[l](posts[l].clone(), posts[l + 1].clone(), pres[l].clone(),
                                      (e @ Bd[l]).clone(), Ws[l].clone(), st)
                        if not torch.is_tensor(dW) or dW.shape != Ws[l].shape:
                            raise ValueError(f"bad dW shape {getattr(dW, 'shape', type(dW))}")
                        dW = dW.float()
                        if not torch.isfinite(dW).all():
                            raise FloatingPointError("non-finite dW")
                        Ws[l].grad = dW
                # rf: hidden grads stay None -> frozen
                opt.step()
            if not all(torch.isfinite(W).all() for W in Ws):
                raise FloatingPointError("non-finite weights")
    return evaluate(Ws, b, Xev, yev)


# Hand-written rules expressed in the same interface (sanity + seeds for search).
DFA_SRC = '''def delta(pre, post, post_pre, err_proj, W, stats):
    d = err_proj * (1 - post ** 2)
    return d.T @ pre / pre.shape[0]
'''
OJA_SRC = '''def delta(pre, post, post_pre, err_proj, W, stats):
    # Oja subspace rule (unsupervised); returned as descent direction
    B = pre.shape[0]
    return -(post.T @ pre - (post.T @ post) @ W) / B
'''


def run_cli():
    """Subprocess entry: reads JSON job on stdin, prints JSON result."""
    job = json.load(sys.stdin)
    t0 = time.time()
    Xtr, ytr, Xv, yv, Xte, yte = load(job["dataset"], n_train=job.get("n_train"))
    Xev, yev = (Xte, yte) if job.get("split") == "test" else (Xv, yv)
    if job.get("val_slice"):
        a, b_ = job["val_slice"]; Xev, yev = Xev[a:b_], yev[a:b_]
    res = {"accs": {}}
    for lr in job.get("lrs", [job.get("lr")]):
        try:  # rule failures are fitness 0, not crashes
            res["accs"][str(lr)] = train(job["method"], Xtr, ytr, job["sizes"], lr, job["epochs"],
                                         job["seed"], job.get("rule_src"), Xev=Xev, yev=yev)
        except Exception as ex:
            res["accs"][str(lr)] = 0.0
            res["error"] = f"{type(ex).__name__}: {ex}"[:300]
    res["lr"], res["acc"] = max(res["accs"].items(), key=lambda kv: kv[1])
    res["sec"] = round(time.time() - t0, 1)
    print(json.dumps(res))


if __name__ == "__main__":
    run_cli()
