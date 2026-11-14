"""反馈方向离真实雅可比有多远——直接量。

对 `sensor_proj` 取 32 个随机方向 u_k：
- 真实方向导数：`(L(θ+εu) − L(θ−εu)) / 2ε`（有限差分，不看公式）；
- DFA 方向导数：`⟨d_DFA, u⟩`，其中 `d_DFA = c ⊗ (z ⊙ hormone)`。

两者的余弦相似度就是"反馈与真实雅可比的对齐度"。1 = 完全对齐（等于反向传播），0 = 无关。
再对比三种反馈：随机 B、B = Hᵀ（真读出）、B 学出来的（此处未做）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from research.sel.lm.serve_snn import load_model

CK = Path("D:/Projects/Semprea-4.5B/dist/semprea-4.5b-v1.0/fast/semprea_v1_ckpt.pt")
QW = Path("D:/Projects/Semprea-4.5B/dist/semprea-4.5b-v1.0/fast")
rng = np.random.default_rng(0)
model, tok, step = load_model(CK, Path("bpe"), torch.device("cpu"), 16, QW)
prefix = tok.encode("当前温度").ids
DIG = [str(d) for d in range(10)]
digit_ids = torch.tensor([tok.encode(d).ids[0] for d in DIG])
H = model.tree.weight[digit_ids].T.contiguous().float()


def feats(c):
    model.sensors = c
    trace = model.new_trace(1)
    with torch.no_grad():
        for i in range(len(prefix) - 1):
            model.step(torch.tensor([prefix[i]]), torch.tensor([prefix[i + 1]]),
                       trace, update=False, all_valid=True)
        return model._features(trace)[0].clone(), trace.z[0].clone()


def loss_and_err(c, y):
    f, z = feats(c)
    p = torch.softmax(f @ H, dim=0)
    err = p.clone(); err[int(y)] -= 1.0
    return float(-torch.log(p[int(y)] + 1e-9)), f, z, err


c0 = torch.zeros(1, 16); c0[0, 0] = 0.7
y0 = 6
L0, f0, z0, err0 = loss_and_err(c0, y0)

eps = 1e-3
K = 32
true_dirs, dfa_rand, dfa_true = [], [], []
B_rand = torch.randn(10, model.readout_width, generator=torch.Generator().manual_seed(11)) / 3.0
for _ in range(K):
    u = torch.randn_like(model.sensor_proj)
    u /= u.norm()
    with torch.no_grad():
        saved = model.sensor_proj.clone()
        model.sensor_proj = saved + eps * u
        Lp = loss_and_err(c0, y0)[0]
        model.sensor_proj = saved - eps * u
        Lm = loss_and_err(c0, y0)[0]
        model.sensor_proj = saved
    true_dirs.append((Lp - Lm) / (2 * eps))

    def dfa(B):
        mod = B.T @ err0
        half = mod.numel() // 2
        return float((torch.outer(c0[0], z0 * (mod[:half] + mod[half:])) * u).sum())

    dfa_rand.append(dfa(B_rand))
    dfa_true.append(dfa(H.T if H.shape[0] == model.readout_width else B_rand))

t = np.array(true_dirs); r = np.array(dfa_rand); tt = np.array(dfa_true)
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
out = {
    "K": K,
    "cos_random_feedback": round(cos(t, r), 4),
    "cos_true_readout_feedback": round(cos(t, tt), 4),
    "loss0": round(L0, 4),
    "note": "余弦 = 1 才等价于反向传播；0 表示无关。",
}
print(json.dumps(out, ensure_ascii=False, indent=1))
Path("sweep_results/synthetic_channel/jacobian.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
