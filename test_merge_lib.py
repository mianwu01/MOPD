"""CPU unit tests for mopd_merge. Run: python3 test_merge_lib.py"""

import math

import torch

from mopd_merge.linalg import (
    polar_factor, polar_decompose, newton_schulz, cayley_log, cayley_exp,
)
from mopd_merge.decompose import ord_decompose, ord_reconstruct
from mopd_merge.operators import OP_REGISTRY, merge_models
from mopd_merge.diagnostics import (
    delta_cosine_report, split_half_reliability, disattenuated_alignment,
)

torch.manual_seed(0)
PASS = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"  [{status}] {name}")


def random_rotation(n, scale=0.05):
    """Near-identity rotation via Cayley of a random small skew matrix."""
    a = torch.randn(n, n) * scale
    return cayley_exp(0.5 * (a - a.T))


print("== linalg ==")
a = torch.randn(6, 6)
q, info = polar_factor(a)
check("polar orthogonality", torch.allclose(q.T @ q, torch.eye(6), atol=1e-5))
q2, _ = polar_factor(3.0 * a)
check("polar scale invariance", torch.allclose(q, q2, atol=1e-5))
# Muon's NS coefficients trade precision for speed: singular values land near 1
# (roughly [0.7, 1.3]), not exactly 1 — assert direction agreement, not identity.
q_ns = newton_schulz(a, iters=12)
sv = torch.linalg.svdvals(q_ns)
check("newton_schulz singular values near 1", bool((sv > 0.5).all() and (sv < 1.5).all()))
rel = torch.linalg.matrix_norm(q_ns - q) / torch.linalg.matrix_norm(q)
check("newton_schulz aligned with SVD polar", rel < 0.3)
ar = torch.randn(4, 9)
qr, _ = polar_factor(ar)
check("rectangular polar semi-orthogonal", torch.allclose(qr @ qr.T, torch.eye(4), atol=1e-5))

r = random_rotation(8)
qlog, ok = cayley_log(r)
check("cayley domain ok for near-identity", ok)
check("cayley log skew", torch.allclose(qlog, -qlog.T, atol=1e-5))
check("cayley round trip", torch.allclose(cayley_exp(qlog), r, atol=1e-5))
flip = torch.eye(8)
flip[0, 0] = -1.0  # det = -1: outside SO(n)
_, ok_flip = cayley_log(flip)
check("cayley rejects det=-1", not ok_flip)

d = torch.randn(10, 6)
qd, hd, side = polar_decompose(d)
check("polar_decompose reconstructs", torch.allclose(qd @ hd, d, atol=1e-4))
check("H symmetric PSD", torch.allclose(hd, hd.T, atol=1e-5)
      and torch.linalg.eigvalsh(hd).min() > -1e-4)

print("== ORD ==")
w0 = torch.randn(8, 12)
r_true = random_rotation(8)
e_true = torch.randn(8, 12) * 0.01
w_d = r_true @ w0 + e_true
r_est, e_est, side, info = ord_decompose(w_d, w0)
check("ORD picks left side for wide matrix", side == "left")
check("ORD reconstruction exact",
      torch.allclose(ord_reconstruct(r_est, e_est, w0, side), w_d, atol=1e-4))
check("ORD recovers rotation", torch.linalg.matrix_norm(r_est - r_true) < 0.15)
check("ORD det +1", abs(info["det"] - 1.0) < 1e-3)
w0_tall = torch.randn(12, 8)
_, _, side_tall, _ = ord_decompose(r_true.T[:0].new_zeros(0, 0).new_ones(12, 8) * 0
                                   + w0_tall + 0.01 * torch.randn(12, 8), w0_tall)
check("ORD picks right side for tall matrix", side_tall == "right")

print("== operators ==")
base = {"layer.weight": torch.randn(8, 8), "layer.bias": torch.randn(8)}
b1 = {k: v + 0.02 * torch.randn_like(v) for k, v in base.items()}
b2 = {k: v + 0.02 * torch.randn_like(v) for k, v in base.items()}
names = list(base)


def gets(d):
    return lambda n: d[n]


# Lossless ops must reproduce a single branch exactly; ties needs density=1.0
# (trimming is lossy by design) and the dir/mag ablations are intentionally lossy.
for op in ("plain_avg", "ta", "v2_ord", "v2_ord_noxc", "v3_polar"):
    merged = merge_models(gets(base), [gets(b1)], names, op)
    exact = all(torch.allclose(merged[n], b1[n], atol=2e-3) for n in names)
    check(f"{op}: single branch reproduces branch", exact)
merged = merge_models(gets(base), [gets(b1)], names, "ties", density=1.0)
check("ties(density=1): single branch reproduces branch",
      all(torch.allclose(merged[n], b1[n], atol=2e-3) for n in names))
full1 = merge_models(gets(base), [gets(b1)], names, "v2_ord")["layer.weight"]
dir1 = merge_models(gets(base), [gets(b1)], names, "v2_dir_only")["layer.weight"]
mag1 = merge_models(gets(base), [gets(b1)], names, "v2_mag_only")["layer.weight"]
check("v2 channel additivity: dir + mag - base = full (single branch)",
      torch.allclose(dir1 + mag1 - base["layer.weight"], full1, atol=1e-4))

merged_ta = merge_models(gets(base), [gets(b1), gets(b2)], names, "ta")
expect = base["layer.weight"] + 0.5 * ((b1["layer.weight"] - base["layer.weight"])
                                       + (b2["layer.weight"] - base["layer.weight"]))
check("ta = weighted delta mean", torch.allclose(merged_ta["layer.weight"], expect, atol=1e-5))

merged_v2 = merge_models(gets(base), [gets(b1), gets(b2)], names, "v2_ord")
check("v2_ord runs and stays finite", all(torch.isfinite(merged_v2[n]).all() for n in names))
merged_dir = merge_models(gets(base), [gets(b1), gets(b2)], names, "v2_dir_only")
merged_mag = merge_models(gets(base), [gets(b1), gets(b2)], names, "v2_mag_only")
check("dir/mag ablations differ from full v2",
      not torch.allclose(merged_dir["layer.weight"], merged_v2["layer.weight"], atol=1e-6)
      and not torch.allclose(merged_mag["layer.weight"], merged_v2["layer.weight"], atol=1e-6))
merged_v3 = merge_models(gets(base), [gets(b1), gets(b2)], names, "v3_polar")
check("v3_polar runs and stays finite", all(torch.isfinite(merged_v3[n]).all() for n in names))

# xc restores norm on partially-cancelling rotations (pure rotation branches)
w0_sq = torch.randn(8, 8)
skew = torch.randn(8, 8) * 0.1
skew = 0.5 * (skew - skew.T)
r_a, r_b = cayley_exp(skew), cayley_exp(-0.6 * skew)  # conflicting rotations
base_r = {"w.weight": w0_sq}
ba = {"w.weight": r_a @ w0_sq}
bb = {"w.weight": r_b @ w0_sq}
m_xc = merge_models(gets(base_r), [gets(ba), gets(bb)], ["w.weight"], "v2_ord")
m_noxc = merge_models(gets(base_r), [gets(bb), gets(ba)], ["w.weight"], "v2_ord_noxc")
q_xc, _ = cayley_log(polar_factor(m_xc["w.weight"] @ w0_sq.T)[0])
q_noxc, _ = cayley_log(polar_factor(m_noxc["w.weight"] @ w0_sq.T)[0])
avg_norm = 0.5 * (torch.linalg.matrix_norm(skew) + torch.linalg.matrix_norm(0.6 * skew))
check("xc restores rotation norm to weighted mean",
      abs(torch.linalg.matrix_norm(q_xc) - avg_norm) < 0.05 * avg_norm)
check("no-xc rotation norm is cancelled (smaller)",
      torch.linalg.matrix_norm(q_noxc) < 0.5 * avg_norm)

# TIES: sign conflict resolution
zero = torch.zeros(4, 4)
d_pos, d_neg = zero.clone(), zero.clone()
d_pos[0, 0], d_neg[0, 0] = 1.0, -0.4    # conflict: elected sign = +
d_pos[1, 1], d_neg[1, 1] = 0.5, 0.7     # agreement
base_t = {"w.weight": zero}
m_ties = merge_models(gets(base_t), [gets({"w.weight": d_pos}), gets({"w.weight": d_neg})],
                      ["w.weight"], "ties", density=0.5)
check("ties keeps elected-sign winner", m_ties["w.weight"][0, 0] > 0)
check("ties averages agreeing entries", abs(m_ties["w.weight"][1, 1] - 0.3) < 1e-5)

print("== diagnostics ==")
da = {"w.weight": torch.randn(16, 16)}
db_same = {"w.weight": base["layer.weight"].new_tensor(da["w.weight"])}
db_opp = {"w.weight": -da["w.weight"]}
zero16 = {"w.weight": torch.zeros(16, 16)}
rep = delta_cosine_report(gets(zero16), {"a": gets(da), "b": gets(db_same)}, ["w.weight"])
check("identical deltas: cos = 1",
      abs(rep["groups"]["_global"]["pairwise_cos"]["a|b"] - 1.0) < 1e-5)
check("identical deltas: rho = 1", abs(rep["groups"]["_global"]["rho"] - 1.0) < 1e-5)
rep_o = delta_cosine_report(gets(zero16), {"a": gets(da), "b": gets(db_opp)}, ["w.weight"])
check("opposite deltas: cos = -1",
      abs(rep_o["groups"]["_global"]["pairwise_cos"]["a|b"] + 1.0) < 1e-5)
check("opposite deltas: rho = 0", rep_o["groups"]["_global"]["rho"] < 1e-5)
check("rho null is 1/sqrt(N)",
      abs(rep["groups"]["_global"]["rho_null_orthogonal"] - 1 / math.sqrt(2)) < 1e-9)
r_split = split_half_reliability(gets(zero16), gets(da), gets(db_same), ["w.weight"])
check("split-half reliability of identical halves = 1", abs(r_split - 1.0) < 1e-5)
check("disattenuation", abs(disattenuated_alignment(0.09, 0.3, 0.3) - 0.3) < 1e-9)

n_pass, n_total = sum(PASS), len(PASS)
print(f"\n{n_pass}/{n_total} checks passed")
raise SystemExit(0 if n_pass == n_total else 1)
