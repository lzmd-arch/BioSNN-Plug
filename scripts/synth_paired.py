"""**语义化合成配对数据**：传感器真的决定文本里那个数。

上一轮失败的原因是"任意绑定"——标签与 SNN 的语言表征无关。这一轮让 c 决定**温度读数**，
文本是自然的："温度{数}度，湿度{数}%，光线{数}%。"

前缀固定为"温度"，要预测的就是**那个数的第一个 token**（1–9 的开头数字）。
c 的前三维（温度/湿度/光线，归一化）真的决定它 ⇒ 独占信号存在且与语言相关。

两个臂：带 c / 不带 c。判据：top-1 差 ≥ 20pp。
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
model.sensor_proj = torch.zeros_like(model.sensor_proj)

PREFIX_TEXT = "当前温度"
prefix_ids = tok.encode(PREFIX_TEXT).ids
DIGITS = [str(d) for d in range(10)]
digit_ids = torch.tensor([tok.encode(d).ids[0] for d in DIGITS])


def sample(n):
    temp = rng.integers(0, 10, size=n)          # 温度的个位
    c = np.zeros((n, 16), dtype=np.float32)
    c[:, 0] = temp / 9.0                        # 第一维就是温度
    c[:, 1:] = rng.normal(size=(n, 15)) * 0.1   # 其余是噪声（不该有用）
    return torch.tensor(c), torch.tensor(temp)


def feats_of(ids, c=None):
    model.sensors = c if c is not None else None
    trace = model.new_trace(1)
    with torch.no_grad():
        for i in range(len(ids) - 1):
            model.step(torch.tensor([ids[i]]), torch.tensor([ids[i + 1]]),
                       trace, update=False, all_valid=True)
        f = model._features(trace)[0].clone()
        z = trace.z[0].clone()
    return f, z


H = model.tree.weight[digit_ids].T.contiguous().float()   # 10 个数字 token 的读出
B = torch.randn(10, model.readout_width, generator=torch.Generator().manual_seed(11)) / 3.0


def evaluate(n=200):
    c, y = sample(n)
    hs = hn = 0
    for i in range(n):
        f, _ = feats_of(prefix_ids, c[i:i+1]); hs += int((f @ H).argmax()) == int(y[i])
        f, _ = feats_of(prefix_ids, None);     hn += int((f @ H).argmax()) == int(y[i])
    return hs / n, hn / n


print("训练前：带 c %.3f / 不带 %.3f" % evaluate(80), flush=True)
eta = 0.02
for epoch in range(40):
    c, y = sample(200)
    for i in range(200):
        f, z = feats_of(prefix_ids, c[i:i+1])
        p = torch.softmax(f @ H, dim=0)
        err = p.clone(); err[int(y[i])] -= 1.0
        mod_f = B.T @ err
        half = mod_f.numel() // 2
        hormone = mod_f[:half] + mod_f[half:]
        model.sensor_proj += eta * torch.outer(c[i], z * hormone)
        with torch.no_grad():
            nn_ = model.sensor_proj.norm()
            if nn_ > 20.0:
                model.sensor_proj *= 20.0 / nn_
    if epoch % 4 == 0:
        print(f"  epoch {epoch}: 带 c %.3f / 不带 %.3f  norm %.1f" % (*evaluate(150), float(model.sensor_proj.norm())), flush=True)

a, b = evaluate(200)
out = {"with_sensor": round(a, 4), "without_sensor": round(b, 4),
       "gain_pp": round((a - b) * 100, 2),
       "verdict": "语义化合成数据上通道学得会" if (a - b) >= 0.20 else "仍未学出"}
print(json.dumps(out, ensure_ascii=False, indent=1))
Path("sweep_results/synthetic_channel/paired.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
