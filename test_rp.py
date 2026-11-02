import torch, pytest
import rp

# Runs on rp.DEV (cuda on Colab, so the FLA / causal-conv1d kernel path is the one tested), always in fp32.
@pytest.fixture(scope="module", params=["Qwen/Qwen3.5-0.8B", "Qwen/Qwen2.5-0.5B-Instruct"])
def mt(request):
    rp.DTYPE = torch.float32
    m, tok = rp.load(request.param)
    return m, tok, request.param

def step(m, cache, tokid):
    return m(input_ids=torch.tensor([[tokid]], device=rp.DEV), past_key_values=cache, use_cache=True).logits[:, -1]

def test_rhymes():
    assert rp.rhymes("night", "light") and not rp.rhymes("night", "day")
    assert rp.last_word("And sat beside the lake.\n") == "lake"

def test_noop_patches(mt):
    m, tok, name = mt
    ids = rp.encode(tok, [rp.prompt(0, "night")])
    c0, l0, cap = rp.prefill(m, ids, capture=True)
    c1, l1, _ = rp.prefill(m, ids, nl_patch=cap)          # write T's own residuals back
    assert torch.allclose(l0, l1, atol=1e-4)
    c2, _, _ = rp.prefill(m, ids)
    for kind in ("state", "kv"):
        rp.patch_cache(c2, c0, kind)                      # patch T with T
    assert torch.allclose(step(m, c0, 11), step(m, c2, 11), atol=1e-4)

def test_full_patch_reproduces_source(mt):
    m, tok, name = mt
    cT, _, _ = rp.prefill(m, rp.encode(tok, [rp.prompt(0, "night")]))
    cS, _, _ = rp.prefill(m, rp.encode(tok, [rp.prompt(0, "day")]))
    cT2, _, _ = rp.prefill(m, rp.encode(tok, [rp.prompt(0, "night")]))
    lT = step(m, cT2, 11)
    rp.patch_cache(cT, cS, "state"); rp.patch_cache(cT, cS, "kv")
    lP, lS = step(m, cT, 11), step(m, cS, 11)
    assert torch.allclose(lP, lS, atol=1e-4) and not torch.allclose(lP, lT, atol=1e-2)


def test_patch_sites(mt):
    """Guard against a silent no-op patch: hybrid has 18 GDN layers (recurrent+conv) and 6 attention KV layers."""
    m, tok, name = mt
    c, _, _ = rp.prefill(m, rp.encode(tok, [rp.prompt(0, "night")]))
    n_state, n_kv = len(rp.cache_tensors(c, "state_rec")), len(rp.cache_tensors(c, "kv")) // 2
    assert (n_state, n_kv) == ((18, 6) if "3.5" in name else (0, m.config.num_hidden_layers))
    assert all(rp._get(o, k) is not None for o, k in rp.cache_tensors(c, "state") + rp.cache_tensors(c, "kv"))

def test_gate_rules():
    import run_all as g
    s = lambda nl, r: {"NL": {"flip": nl}, "NL_rand": {"flip": r}}
    assert g.q1(s([.40, .33, .47], [.02, .01, .05])) == "YES"
    assert g.q1(s([.08, .05, .12], [.0, .0, .02])) == "NO"
    assert g.q1(s([.20, .15, .26], [.0, .0, .02])) == "WEAK"
    assert g.q1(s([.35, .03, .60], [.02, .0, .1])) == "NO"          # CIs overlap with control
    ctrl = {"STATE": "STATE_rand", "KV": "KV_rand"}
    z = [.0, .0, .02]
    assert g.locus({"STATE": [.6, .5, .7], "KV": [.2, .15, .25], "STATE_rand": z, "KV_rand": z}, ctrl)[0] == "STATE"
    assert g.locus({"STATE": [.6, .5, .7], "KV": [.5, .4, .6], "STATE_rand": z, "KV_rand": z}, ctrl)[0] == "SHARED"
    assert g.locus({"STATE": [.01, .0, .03], "KV": [.3, .2, .4], "STATE_rand": z, "KV_rand": z}, ctrl)[0] == "KV"
