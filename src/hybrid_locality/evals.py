"""RULER-style retrieval tasks (teacher-forced exact match) and LongPPL key-token perplexity."""
import random

import torch

from .window import staggered

KEYS = ("amber falcon", "quiet harbor", "silver orchard", "broken compass", "velvet canyon", "frozen lantern",
        "crimson meadow", "hollow beacon", "gentle thunder", "copper willow", "distant marble", "golden thistle",
        "rusty anchor", "pale horizon", "wild juniper", "iron sparrow", "lucky pebble", "misty glacier")
FAR = 4096  # every needle sits >= FAR tokens before the question, so W <= FAR cannot see it locally


def logit_scale(model):
    # Falcon-H1 multiplies logits after lm_head, Granite divides; everyone else is 1.
    return getattr(model.get_decoder(), "lm_head_multiplier", 1.0) / getattr(model.config, "logits_scaling", 1.0)


@torch.no_grad()
def target_logprobs(model, ids, trunc=None, chunk=2048):
    """log p(ids[t] | ids[<t]) for t=1..L-1, computed without materialising L x vocab logits.

    trunc=W evaluates every position with only its last W/2..W tokens of context (Trunc arm).
    """
    dec = model.get_decoder()
    run = lambda x: dec(input_ids=x, use_cache=False).last_hidden_state
    h = run(ids) if trunc is None else staggered(run, ids, trunc)
    head, scale, out = model.get_output_embeddings(), logit_scale(model), []
    for s in range(0, ids.shape[1] - 1, chunk):
        lg = head(h[:, s : s + chunk][:, : ids.shape[1] - 1 - s]).float() * scale
        out.append(lg.log_softmax(-1).gather(-1, ids[:, s + 1 : s + 1 + lg.shape[1], None])[..., 0])
    return torch.cat(out, dim=1)[0]


def _enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def make_task(kind, tok, hay, L, rng):
    """Return (ids, answer_len). hay: list of haystack token ids, longer than L."""
    val = str(rng.randrange(10**6, 10**7))
    if kind == "single":
        key = rng.choice(KEYS)
        needles = [f" One of the special magic numbers for {key} is: {val}."]
        q = f" What is the special magic number for {key} mentioned in the provided text? The special magic number for {key} mentioned in the provided text is"
    elif kind == "multikey":
        keys = rng.sample(KEYS, 4)
        vals = [val] + [str(rng.randrange(10**6, 10**7)) for _ in range(3)]
        needles = [f" One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)]
        rng.shuffle(needles)
        q = f" What is the special magic number for {keys[0]} mentioned in the provided text? The special magic number for {keys[0]} mentioned in the provided text is"
    elif kind == "vt":  # 4-hop variable chain, needles kept in causal order
        names = ["".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=5)) for _ in range(5)]
        needles = [f" VAR {names[0]} = {val}."] + [f" VAR {b} = VAR {a}." for a, b in zip(names, names[1:])]
        q = f" Question: after following every assignment above, what number does VAR {names[-1]} hold? Answer: VAR {names[-1]} holds the number"
    else:
        raise ValueError(kind)
    nd, q, a = [_enc(tok, s) for s in needles], _enc(tok, q), _enc(tok, " " + val)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    n_hay = L - len(bos) - len(q) - len(a) - sum(map(len, nd))
    assert n_hay > FAR, f"L={L} too short"
    start = rng.randrange(0, len(hay) - n_hay)
    body = hay[start : start + n_hay]
    cuts = sorted(rng.sample(range(n_hay - FAR), len(nd)))
    ids, prev = list(bos), 0
    for c, n in zip(cuts, nd):
        ids += body[prev:c] + n
        prev = c
    ids += body[prev:] + q + a
    assert len(ids) == L
    return torch.tensor([ids]), len(a)


@torch.no_grad()
def exact_match(model, ids, n_ans, trunc=None):
    """Greedy decoding would emit the answer iff every answer token is the argmax (teacher forcing)."""
    dec, head = model.get_decoder(), model.get_output_embeddings()
    run = lambda x: dec(input_ids=x, use_cache=False).last_hidden_state
    h = run(ids) if trunc is None else staggered(run, ids, trunc)
    pred = head(h[:, -n_ans - 1 : -1]).argmax(-1)  # scale > 0 does not change argmax
    return bool((pred == ids[:, -n_ans:]).all())


def longppl(lp, lp_full, lp_short, start=FAR, alpha=2.0, beta=-2.0):
    """LongPPL (Fang et al., ICLR 2025): NLL on key tokens, i.e. tokens that gain > alpha nats from
    long context and are already well predicted (log p > beta) with it. lp* are aligned logprob vectors."""
    key = (lp_full - lp_short > alpha) & (lp_full > beta)
    key[: start - 1] = False
    return {"key_nll": float(-lp[key].mean()) if key.any() else float("nan"),
            "n_key": int(key.sum()), "nll": float(-lp[start - 1 :].mean())}


def haystack(tok, n_books=40, seed=0):
    from datasets import load_dataset

    books = list(load_dataset("emozilla/pg19-test", split="test")["text"])
    random.Random(seed).shuffle(books)
    return [_enc(tok, b[:2_000_000]) for b in books[:n_books]]
