"""Couplet data, rhyme scoring, greedy generation with residual / cache patching."""
import re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor"))
import pronouncing
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = os.environ.get("RP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = getattr(torch, os.environ.get("RP_DTYPE", "float32"))  # fp32 = pre-registered/pilot setting; see README 2026-11-02
RESULTS = os.environ.get("RP_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

CLASSES = {
    "ight": "night light kite sight knight", "ay": "day bay clay tray hay", "ee": "tree sea bee key tea",
    "ing": "king ring spring wing swing", "ore": "door floor shore store", "ell": "bell well shell cell",
    "at": "cat hat bat mat rat", "ain": "rain train chain plain", "oon": "moon spoon noon",
    "ar": "star car jar bar", "ake": "cake lake snake", "ed": "bed bread shed", "own": "town crown gown",
    "ow": "snow crow bow", "old": "gold mold", "ue": "glue shoe zoo", "ire": "fire wire tire",
    "ill": "hill mill pill",
}
TEMPLATES = [
    "She looked across the field and saw the {}", "He stood alone and waited by the {}",
    "We sang a happy song about the {}", "They walked together slowly to the {}",
    "I tried my very best to find the {}", "The old man smiled and gently touched the {}",
    "The children laughed and ran towards the {}", "My mother told a story of the {}",
    "At dawn the farmer went to see the {}", "The little girl began to draw the {}",
]
FEWSHOT = ("Rhyming couplets:\n\n"
           "The little boy was full of joy\nHe played all day with his new toy\n\n"
           "The quiet night was dark and deep\nAnd all the town was fast asleep\n\n"
           "We went outside to have some fun\nAnd sat together in the sun\n\n")

def rhyme_parts(w):
    return {pronouncing.rhyming_part(p) for p in pronouncing.phones_for_word(w.lower())}

def rhymes(a, b):
    return bool(a) and bool(b) and bool(rhyme_parts(a) & rhyme_parts(b))

def last_word(text):
    ws = re.findall(r"[A-Za-z']+", text.split("\n")[0])
    return ws[-1].strip("'").lower() if ws else ""

def lines(toks):
    """(template_idx, word, class) for words that are one token (' word') in every tokenizer given."""
    out = []
    for c, ws in CLASSES.items():
        for w in ws.split():
            if all(len(t.tokenize(" " + w)) == 1 for t in toks) and rhyme_parts(w):
                out += [(i, w, c) for i in range(len(TEMPLATES))]
    return out

def prompt(ti, w):
    return FEWSHOT + TEMPLATES[ti].format(w) + "\n"

def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to(DEV).eval()
    return m, tok

def decoder_layers(m):
    return m.model.layers

@torch.no_grad()
def prefill(m, ids, nl_patch=None, capture=False):
    """Run the prompt. nl_patch: list (per layer) of [B,H] to write into each layer output at the last position.
    Returns (cache, last_logits, captured list of [B,H] per layer incl. embeddings-free layer outputs)."""
    cap, hooks = [], []
    for i, layer in enumerate(decoder_layers(m)):
        def hook(mod, inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            if nl_patch is not None:
                h[:, -1] = nl_patch[i]
            if capture:
                cap.append(h[:, -1].clone())
            return out
        hooks.append(layer.register_forward_hook(hook))
    try:
        out = m(input_ids=ids, use_cache=True)
    finally:
        for h in hooks:
            h.remove()
    return out.past_key_values, out.logits[:, -1], cap

def cache_tensors(cache, kind):
    """List of (layer_obj, attr, idx_or_None) for kind in {'kv','state','state_rec'}."""
    res = []
    for L in cache.layers:
        if hasattr(L, "recurrent_states"):
            if kind in ("state", "state_rec"):
                res.append((L.recurrent_states, 0))
            if kind == "state":
                res.append((L.conv_states, 0))
        elif kind == "kv":
            res += [(L, "keys"), (L, "values")]
    return res

def _get(c, k):
    return c[k] if isinstance(c, dict) else getattr(c, k)

def _set(c, k, v):
    if isinstance(c, dict):
        c[k] = v
    else:
        setattr(c, k, v)

@torch.no_grad()
def patch_cache(cT, cS, kind, rand=False, gen=None):
    """Overwrite T's cache tensors of `kind` with S's, or (rand) with T + random dir of norm |S-T| per example."""
    for (cT_obj, k), (cS_obj, _) in zip(cache_tensors(cT, kind), cache_tensors(cS, kind)):
        t, s = _get(cT_obj, k), _get(cS_obj, k)
        if rand:
            d = (s - t).flatten(1).norm(dim=1)
            r = torch.randn(t.shape, generator=gen).to(t.device, t.dtype)
            r = r * (d / r.flatten(1).norm(dim=1)).view(-1, *[1] * (t.dim() - 1))
            new = t + r
        else:
            new = s.clone()
        _set(cT_obj, k, new)

def rand_like_delta(t, s, gen):
    d = (s - t).norm(dim=-1, keepdim=True)
    r = torch.randn(t.shape, generator=gen).to(t.device, t.dtype)
    return t + r * d / r.norm(dim=-1, keepdim=True)

@torch.no_grad()
def greedy(m, tok, cache, logits, max_new=24):
    B = logits.shape[0]
    toks = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool)
    for _ in range(max_new):
        nxt = logits.argmax(-1)
        nl = nxt.tolist()  # one device sync per step
        for b in range(B):
            if not done[b]:
                toks[b].append(nl[b])
                if "\n" in tok.decode([nl[b]]):
                    done[b] = True
        if done.all():
            break
        out = m(input_ids=nxt[:, None], past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1]
    return [tok.decode(t) for t in toks]

def encode(tok, prompts):
    ids = [tok(p).input_ids for p in prompts]
    assert len({len(i) for i in ids}) == 1, "minimal pairs must have equal token length"
    return torch.tensor(ids, device=DEV)

def kernel_info():
    """Which GDN kernel packages are importable (transformers uses them instead of the torch path if so)."""
    import importlib, importlib.metadata as md
    out = {}
    for mod, dist in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
        try:
            importlib.import_module(mod); out[dist] = md.version(dist)
        except Exception:
            out[dist] = None
    return out
