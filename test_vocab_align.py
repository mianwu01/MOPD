"""
Smoke test: OPSD losses must handle Qwen2.5 student/teacher vocab gap
(151936 vs 152064) without shape error.
"""
import sys, os
sys.path.insert(0, "/home/ubuntu/OPSD_OnPolicyDistillation/src")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import torch
from opd.losses import (
    compute_reverse_kl_loss,
    compute_forward_kl_loss,
    compute_jsd_loss,
    entropy_weighted_sample,
    _align_vocab,
)


def main():
    torch.manual_seed(0)
    N, V_S, V_T = 64, 151936, 152064  # Qwen2.5 1.5B / 7B

    s_logits = torch.randn(N, V_S, requires_grad=True)
    t_logits = torch.randn(N, V_T)

    # 1. _align_vocab returns common dim
    a_t, a_s = _align_vocab(t_logits, s_logits)
    assert a_t.shape == (N, V_S), a_t.shape
    assert a_s.shape == (N, V_S), a_s.shape

    # 2. Each loss runs on mismatched dims and produces finite scalar
    for name, fn in [
        ("reverse_kl", compute_reverse_kl_loss),
        ("forward_kl", compute_forward_kl_loss),
        ("jsd", lambda t, s, chunk_size=64: compute_jsd_loss(t, s, beta=0.5, chunk_size=chunk_size)),
    ]:
        loss, n = fn(t_logits.clone(), s_logits, chunk_size=32)
        assert loss.shape == (), (name, loss.shape)
        assert torch.isfinite(loss), (name, loss)
        assert n == N, (name, n)
        # backprop sanity
        s_grad = torch.autograd.grad(loss, s_logits, retain_graph=True)[0]
        assert s_grad.shape == s_logits.shape
        assert torch.isfinite(s_grad).all()
        print(f"[ok] {name:11s}  loss={loss.item():.6f}  grad_finite=True")

    # 3. entropy_weighted_sample works on mismatched dims
    s2, t2, info = entropy_weighted_sample(
        s_logits.detach(), t_logits, sample_ratio=0.5, entropy_alpha=1.0,
        chunk_size=32,
    )
    assert s2.shape[1] == V_S
    assert t2.shape[1] == V_S  # truncated
    print(f"[ok] entropy_weighted_sample  kept={s2.shape[0]}/{N}")

    # 4. Equal-vocab path stays a no-op (back compat)
    eq_t = torch.randn(N, V_S)
    a_t2, a_s2 = _align_vocab(eq_t, s_logits.detach())
    assert a_t2.data_ptr() == eq_t.data_ptr(), "equal-vocab path must not slice"
    print("[ok] equal-vocab path is no-op (no slice)")

    print("\nAll vocab-align tests passed.")


if __name__ == "__main__":
    main()
