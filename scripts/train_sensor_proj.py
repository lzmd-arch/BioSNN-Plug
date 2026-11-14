"""用**纯局部规则**训 `sensor_proj`：三因子（前 · 后 · 奖励）。

- 前：传感器向量 c 的第 i 维；
- 后：神经元 j 在这一段里的平均发放；
- 奖励：这次预测对不对（±1，标量广播）。

`Δproj[i,j] = η · reward · c[i] · mean_z[j]` —— 没有反向传播、没有全局梯度。

数据是合成的：词表 64，上下文 c∈R^8 决定标签，前缀 token 只贡献 30%。
判据：带传感器 vs 不带，留出 top-1 差 ≥ 20pp。
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
V, D, PREFIX = 64, 16, 2
rng = np.random.default_rng(0)

model, tok, step = load_model(CK, Path("bpe"), torch.device("cpu"), 16, QW)
model.sensor_proj = torch.zeros_like(model.sensor_proj)
# **读出必须是真的训过的**：用树上已训练的前 64 行当 64 个类的原型。
# 上一版用随机头，模型没有能力表达映射，测出来的是实验设计的错，不是通道的错。
H = model.tree.weight[:V].T.contiguous().float()
Wc = torch.tensor(rng.normal(size=(D, V)) / np.sqrt(D), dtype=torch.float32)


def sample(n):
    c = torch.tensor(rng.normal(size=(n, D)), dtype=torch.float32)
    prev = torch.tensor(rng.integers(0, V, size=n))
    logits = c @ Wc
    return c, prev, logits.argmax(1)


def forward(prefix_ids, c=None, want_z=False):
    model.sensors = c if c is not None else None
    trace = model.new_trace(1)
    zs = torch.zeros(model.n_rec) if want_z else None
    with torch.no_grad():
        for i in range(len(prefix_ids) - 1):
            model.step(torch.tensor([prefix_ids[i]]), torch.tensor([prefix_ids[i + 1]]),
                       trace, update=False, all_valid=True)
            if want_z:
                zs += trace.z[0]
        feats = model._features(trace)[0].clone()
    return feats, (zs / max(len(prefix_ids) - 1, 1) if want_z else None)


def predict(prefix_ids, c):
    feats, _ = forward(prefix_ids, c)
    return int((feats @ H).argmax())


def evaluate(n=60):
    c, prev, y = sample(n)
    hit_s = hit_n = 0
    for i in range(n):
        ids = [int(prev[i]), int(prev[(i + 1) % len(prev)])]
        hit_s += predict(ids, c[i:i+1]) == int(y[i])
        hit_n += predict(ids, None) == int(y[i])
    return hit_s / n, hit_n / n


print("训练前：带传感器 %.3f / 不带 %.3f" % evaluate(), flush=True)
# **模拟激素**：输出误差 e（64 维）经一个**固定随机**的反馈矩阵 B 广播到全部神经元。
# 这就是随机反馈对齐（DFA）：不穿过网络反传，只有一条固定广播通路。
# **反馈矩阵必须和读出同维**：64 → 15624（特征空间），再折回 7812 个神经元
B = torch.randn(V, model.readout_width, generator=torch.Generator().manual_seed(11)) / (V ** 0.5)
eta = 0.05
for epoch in range(10):
    c, prev, y = sample(200)
    for i in range(200):
        ids = [int(prev[i]), int(prev[(i + 1) % len(prev)])]
        feats, zs = forward(ids, c[i:i+1], want_z=True)
        # **加密奖励（模拟激素）**：不是"对/错"的 ±1，而是正确类的概率减去均匀基线。
        # 连续、每步都有值 ⇒ 即使一次都没答对，也有可用的调制信号。
        logits = feats @ H
        probs = torch.softmax(logits, dim=0)
        err = probs.clone()
        err[int(y[i])] -= 1.0
        mod_f = B.T @ err                        # (readout_width,) 特征空间的调制
        half = mod_f.numel() // 2
        # 每个神经元贡献 2 个特征维（两个时间尺度）⇒ 折回神经元级
        hormone = mod_f[:half] + mod_f[half:]
        # 三因子：前 = c[i]，后 = 该段平均发放，调制 = **激素向量**（不是标量）
        model.sensor_proj += eta * torch.outer(c[i], zs * hormone)
        # 范数上限：不加就爆炸，爆炸会掩盖真实趋势
        with torch.no_grad():
            n = model.sensor_proj.norm()
            if n > 20.0:
                model.sensor_proj *= 20.0 / n
    print(f"  epoch {epoch}: 带传感器 %.3f / 不带 %.3f  norm %.2f"
          % (*evaluate(), float(model.sensor_proj.norm())), flush=True)

a, b = evaluate(120)
out = {"with_sensor_top1": round(a, 4), "without_sensor_top1": round(b, 4),
       "gain_pp": round((a - b) * 100, 2),
       "proj_norm": round(float(model.sensor_proj.norm()), 4),
       "verdict": "局部规则训得出通道" if (a - b) >= 0.20 else "三个 epoch 内没训出来"}
print(json.dumps(out, ensure_ascii=False, indent=1))
p = Path("sweep_results/synthetic_channel/sensor_proj.json")
p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
