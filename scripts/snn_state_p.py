"""把状态送进**真的 SNN 池**，取出池向量当 `p`。

之前 `p` 是我手写的 12 维（直接编码读者模型/条数/在线/轮次）。这一版换成
**脉冲网络算出来的**状态：状态先变成传感器波形，驱动几个独立池，再读出池特征。

## 状态 → 传感器 → 池

| 状态量 | 驱动哪个池 | 怎么驱动 |
| :--- | :--- | :--- |
| 记忆条数 | `presence` | 常数 `log1p(mem)/12`（张力性） |
| 对话轮次 | `time` | 先插 `turns` 个空步，再看池推进到哪 |
| 在线与否 | `persona` | 16 维向量，在线/离线各一个原型 |
| 读者模型 | `memory` | 4 个读者模型各一个 16 维原型 |

## 为什么这样映射

池要**独立**才不互相淹没（这是传感器那条线已经验证过的）。四个状态量分别驱动四个
不同的池，池与池之间零连接，读出的就是四段互不污染的编码。

池用**设计编码**（温度计码 / 正交投影），尺度由标定值给。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sel.lm.model import TokenEProp

POOLS = ["presence", "time", "persona", "memory"]
POOL = 16
TEXT_N = 156
BACKBONES = ["Qwen3-4B", "DeepSeek", "Qwen2.5-0.5B", "本地小模型"]
PROTOS = np.random.default_rng(11).normal(0, 1, (8, POOL)).astype(np.float32)
PROTOS /= np.linalg.norm(PROTOS, axis=1, keepdims=True)
P_DIM = 12


def _build() -> TokenEProp:
    from research.sel.lm.train_sensor_task import _pool_columns

    n_rec = TEXT_N + POOL * len(POOLS)
    counts = torch.from_numpy(np.load("sweep_results/lm_10m/counts_32000.npy")).float()
    model = TokenEProp(
        32000,
        n_rec,
        recurrence="none",
        readout_taus=(2.0, 128.0),
        tree=None,
        embed_rank=256,
        sensor_dim=POOL * len(POOLS),
        lr_in=1e-3,
        lr_out=0.5,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    import research.sel.lm.train_sensor_task as T

    T.POOL = POOL
    T.TEXT_N = TEXT_N
    T.POOL_SIZES = dict.fromkeys(POOLS, POOL)
    T.POOL_OFFSET = {p: i * POOL for i, p in enumerate(POOLS)}
    model.set_sensor_pools({n: _pool_columns(n, k) for k, n in enumerate(POOLS)})
    with torch.no_grad():
        if getattr(model, "proj", None) is not None:
            model.proj[:, TEXT_N:] = 0.0
        if getattr(model, "w_in", None) is not None:
            model.w_in[:, TEXT_N:] = 0.0
        model.sensor_proj = torch.zeros_like(model.sensor_proj)
        for n in POOLS:
            kind = "ramp" if n in ("presence", "time") else "ortho"
            model.design_sensor_encoder(n, kind=kind, scale=2.0, seed=0, steps=2)
    del counts
    return model


class SCNNState:
    """把状态编码成池向量。**一次构建，反复查询**（池是冻结的）。"""

    def __init__(self) -> None:
        self.model = _build()

    @torch.no_grad()
    def encode(self, states: list[dict]) -> np.ndarray:
        model = self.model
        nb = len(states)
        full = np.zeros((nb, POOL * len(POOLS)), dtype=np.float32)
        off = {p: i * POOL for i, p in enumerate(POOLS)}
        for i, s in enumerate(states):
            full[i, off["presence"]] = min(1.0, math.log1p(s["mem"]) / 12.0)
            # 轮次直接当驱动：空步方案会把整批拉到最大轮次，轮次信息就丢了
            full[i, off["time"]] = min(1.0, s["turns"] / 50.0)
            full[i, off["persona"] : off["persona"] + POOL] = (
                PROTOS[0] if s["online"] else PROTOS[1]
            )
            full[i, off["memory"] : off["memory"] + POOL] = PROTOS[2 + s["bb"]]
        tr = model.new_trace(nb)
        tok = torch.full((nb,), 100, dtype=torch.long)
        model.sensors = torch.from_numpy(full)
        model.step(tok, tok, tr, update=False, all_valid=True)
        model.sensors = None
        feats = model._features(tr)  # (B, 2·n_rec)
        idx = []
        for t in range(2):
            for p in POOLS:
                lo = TEXT_N + off[p]
                idx.extend(range(t * model.n_rec + lo, t * model.n_rec + lo + POOL))
        f = feats[:, idx].numpy()
        # 固定随机投影到 12 维（不学习：p 只做状态载体）
        g = np.random.default_rng(7)
        proj = g.normal(0, 1 / math.sqrt(f.shape[1]), (f.shape[1], P_DIM)).astype(np.float32)
        p = f @ proj
        return (p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-6)).astype(np.float32)


if __name__ == "__main__":
    enc = SCNNState()
    p = enc.encode(
        [
            {"mem": 10, "online": True, "bb": 0, "turns": 3},
            {"mem": 300, "online": False, "bb": 2, "turns": 8},
        ]
    )
    print("p 形状", p.shape)
    print("两条状态的 p 差异", float(np.abs(p[0] - p[1]).mean()))
    same = enc.encode([{"mem": 10, "online": True, "bb": 0, "turns": 3}] * 2)
    print("同状态重复编码，差", float(np.abs(same[0] - same[1]).max()))
