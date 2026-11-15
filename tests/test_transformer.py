"""Stage D model plumbing: causality, captured sites, residual bookkeeping."""
import torch

from prt.transformer import ModelConfig, Transformer


def test_causal_and_sites():
    for norm, mlp in [("none", True), ("pre", True), ("none", False)]:
        torch.manual_seed(0)
        m = Transformer(ModelConfig(norm=norm, mlp=mlp, n_layers=2)).double()
        x = torch.randint(0, 4, (3, 16))
        cache = {}
        y = m(x, cache)
        assert set(cache) == set(m.sites())
        x2 = x.clone(); x2[:, 10:] = (x2[:, 10:] + 1) % 4
        assert torch.allclose(m(x2)[:, :10], y[:, :10])          # earlier positions ignore the future
        for l in range(2):
            mid = cache[f"L{l}.resid_pre"] + cache[f"L{l}.attn_out"]
            assert torch.allclose(mid, cache[f"L{l}.resid_mid"])
            post = mid + (cache[f"L{l}.mlp_out"] if mlp else 0)
            assert torch.allclose(post, cache[f"L{l}.resid_post"])
