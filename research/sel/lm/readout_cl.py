"""冻结表示下的**读出层持续学习**——Route R 的 Step 0（AF 基线）。

ADR-0017 把 CL 主路线定为 **Route R：冻结强表示 + 抗遗忘**，并且写明
"门槛数字要从 Step 0 重测基线推导"。本文件就是那个 Step 0 的仪器。

**冻结的**：传感器池的编码（解析构造）、文本主干、全部池内权重——本文件一次都不更新它们。
**学习的**：一个**线性读出**，用**解析梯度**（softmax + 交叉熵，手写，无 autograd）。
理由：线性层上的局部梯度**就是**精确梯度（`grad_cosine` 已量过），所以这一层既能学，
又能干净地测遗忘——不会把"规则不精确"混进 AF 里。

三臂（对应 ADR-0017 的归因链）：

| 臂 | 做法 | 对应的历史结论 |
| :--- | :--- | :--- |
| `single` | 一个共享头吃全部任务，无任务 id、无回放 | 纯 CIL 口径（ADR-0029 已归档，这里只作**下界对照**） |
| `multi` | 每任务一个头 + oracle 路由 | ADR-0018 的任务感知口径（AF 结构性为 0，是**上界**） |
| `replay` | **同一个共享头** + 每类定额的 latent replay | ADR-0017 的归因链第一档 |

指标：`ACC`（全部任务都学完之后，在**所有**任务上的平均正确率）、
`AF`（每个任务"历史最高"减"最终"的平均）。

**任务序列**：五个任务共用**同一份冻结特征**，只换标签映射
（`label = (value + 3k) mod 10 + 10k`）。这样遗忘是**真会发生**的；
若每个任务用不同的池，线性头天然互不干扰，测出来的 AF 恒为 0，什么也证明不了。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from research.sel.lm.hierarchical import build_huffman_paths
from research.sel.lm.model import TokenEProp
from research.sel.lm.train_sensor_task import (
    POOL,
    POOLS,
    PREFIX,
    TEXT_N,
    _pool_columns,
)

N_TASKS = 5
N_CLASS = 10


def build_model(pools: list[str], scales: dict[str, float], seed: int, design: bool) -> TokenEProp:
    n_rec = TEXT_N + POOL * len(pools)
    counts = torch.from_numpy(np.load("sweep_results/lm_10m/counts_32000.npy")).float()
    paths = build_huffman_paths(counts)
    model = TokenEProp(
        32000,
        n_rec,
        recurrence="none",
        readout_taus=(2.0, 128.0),
        tree=paths,
        tree_signal_scale=3.0,
        tree_packed=True,
        tree_init_weight=False,
        embed_rank=256,
        sensor_dim=POOL * len(pools),
        lr_in=1e-3,
        lr_out=0.5,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    model.set_sensor_pools({name: _pool_columns(name, k) for k, name in enumerate(pools)})
    with torch.no_grad():
        if getattr(model, "proj", None) is not None:
            model.proj[:, TEXT_N:] = 0.0
        if getattr(model, "w_in", None) is not None:
            model.w_in[:, TEXT_N:] = 0.0
        model.sensor_proj = torch.zeros_like(model.sensor_proj)
        for name in pools:
            sc = scales.get(name, 1.0)
            if design:
                kind = "ramp" if name in ("temp", "time", "presence") else "ortho"
                model.design_sensor_encoder(name, kind=kind, scale=sc, seed=seed, steps=2)
            else:
                i_sl, _c_sl = model.sensor_pools[name]
                gb = torch.Generator().manual_seed(1000 + 17 * POOLS.index(name) + 7919 * seed)
                ref = math.sqrt(TEXT_N + POOL)
                model.sensor_proj[i_sl] = torch.randn(
                    (i_sl.stop - i_sl.start, model.n_rec), generator=gb
                ) * (sc / ref)
        model.sensor_proj *= model._pool_mask
    return model


def extract(model, pool: str, slot: int, values: np.ndarray, prefix_ids, digit_ids) -> torch.Tensor:
    """冻结特征：跑一遍固定前缀，取该池的读出特征（两个时间尺度 × 16）。"""
    batch = len(values)
    sensors = np.zeros((POOL, batch, 1), dtype=np.float32)
    sensors[0, :, 0] = values / 9.0
    full = np.zeros((batch, POOL * len(model.sensor_pools)), dtype=np.float32)
    full[:, slot * POOL : (slot + 1) * POOL] = sensors[:, :, 0].T
    model.sensors = torch.from_numpy(full)
    n_rec = model.n_rec
    idx = []
    for t in range(2):
        lo = TEXT_N + slot * POOL
        idx.extend(range(t * n_rec + lo, t * n_rec + lo + POOL))
    tr = model.new_trace(batch)
    seq = [*prefix_ids, digit_ids[0]]
    for i in range(len(seq) - 1):
        tok = torch.full((batch,), seq[i], dtype=torch.long)
        tgt = torch.full((batch,), seq[i + 1], dtype=torch.long)
        model.step(tok, tgt, tr, update=False, all_valid=True)
    model.sensors = None
    with torch.no_grad():
        return model._features(tr)[:, idx].clone()


class LinearHead:
    """softmax + 交叉熵的**解析梯度**（手写，无 autograd）——线性层上它等于精确梯度。"""

    def __init__(self, n_in: int, n_out: int, lr: float = 0.5, wd: float = 1e-4):
        self.w = torch.zeros(n_in, n_out)
        self.b = torch.zeros(n_out)
        self.lr = lr
        self.wd = wd

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b

    def update(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        p = torch.softmax(self.logits(x), dim=-1)
        p[torch.arange(len(y)), y] -= 1.0
        if mask is not None:
            p = p * mask.unsqueeze(1)
        p = p / max(len(y), 1)
        self.w -= self.lr * (x.T @ p + self.wd * self.w)
        self.b -= self.lr * p.sum(0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x).argmax(1)


def make_task(rng: np.random.Generator, k: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """任务 k：同一份特征，**局部标签** `(t + 3k) mod 10`。

    多头的每个头只认 0..9；单头把它加上 `10k` 得到互不相交的全局标签空间。
    """
    t = rng.integers(0, 10, size=batch)
    return t, (t + 3 * k) % 10


def run(
    arm: str,
    tasks: int = N_TASKS,
    steps_per_task: int = 1500,
    batch: int = 64,
    replay_rho: float = 0.1,
    lr: float = 0.5,
    pool: str = "temp",
    scale: float = 2.0,
    seed: int = 0,
    design: bool = True,
) -> dict:
    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    tok = SubsetTokenizer(Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2"))
    digit_ids = [tok.encode(str(d)).ids[0] for d in range(10)]
    prefix_ids = tok.encode(PREFIX[0]).ids

    model = build_model([pool], {pool: scale}, seed, design)
    slot = 0
    rng = np.random.default_rng(seed)
    heads = (
        {k: LinearHead(2 * POOL, N_CLASS, lr=lr) for k in range(tasks)} if arm == "multi" else {}
    )
    shared = LinearHead(2 * POOL, N_CLASS * tasks, lr=lr) if arm != "multi" else None
    buffers: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {k: [] for k in range(tasks)}
    hist: list[list[float]] = [
        [0.0] * tasks for _ in range(tasks)
    ]  # hist[k][j] = acc(task j) after task k

    t0 = time.time()
    for k in range(tasks):
        for _ in range(steps_per_task):
            vals, lab = make_task(rng, k, batch)
            x = extract(model, pool, slot, vals, prefix_ids, digit_ids)
            if arm == "multi":
                heads[k].update(x, torch.from_numpy(lab))
            else:
                y = torch.from_numpy(lab + 10 * k)
                shared.update(x, y)
                buffers[k].append((x, y))
                if len(buffers[k]) > 40:
                    buffers[k].pop(0)
                if arm == "replay" and k > 0:
                    # latent replay：从每个旧任务按 ρ 抽一小批，混进**同一个**头
                    take = max(1, math.ceil(replay_rho * batch))
                    for j in range(k):
                        if not buffers[j]:
                            continue
                        xs = torch.cat([b[0] for b in buffers[j]])
                        ys = torch.cat([b[1] for b in buffers[j]])
                        sel = torch.randint(0, len(xs), (take,))
                        shared.update(xs[sel], ys[sel])
        # 每学完一个任务，测**所有**任务（含没见过的，记 0）
        for j in range(tasks):
            vals, lab = make_task(np.random.default_rng(10_000 + j), j, 512)
            xj = extract(model, pool, slot, vals, prefix_ids, digit_ids)
            # multi 是 oracle 路由：用**该任务自己的头**；另外两臂共用一个头
            pred = heads[j].predict(xj) if arm == "multi" else shared.predict(xj) % 10
            hist[k][j] = float((pred == torch.from_numpy(lab)).float().mean())
        print(
            f"  [{arm}] 学完任务 {k}: 逐步 ACC {np.mean([hist[k][j] for j in range(k + 1)]):.3f}"
            f"  ({(time.time() - t0):.0f}s)",
            flush=True,
        )

    final = hist[-1]
    acc = float(np.mean(final))
    af = float(np.mean([max(hist[k][j] for k in range(tasks)) - final[j] for j in range(tasks)]))
    return {
        "arm": arm,
        "tasks": tasks,
        "steps_per_task": steps_per_task,
        "rho": replay_rho,
        "pool": pool,
        "scale": scale,
        "seed": seed,
        "design": design,
        "acc": acc,
        "af": af,
        "matrix": hist,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("single", "multi", "replay"), required=True)
    ap.add_argument("--tasks", type=int, default=N_TASKS)
    ap.add_argument("--steps-per-task", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument(
        "--pool", default="temp", choices=("temp", "time", "memory", "persona", "presence")
    )
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-encoder", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("sweep_results/readout_cl"))
    args = ap.parse_args(argv)
    out = run(
        args.arm,
        args.tasks,
        args.steps_per_task,
        args.batch,
        args.rho,
        args.lr,
        args.pool,
        args.scale,
        args.seed,
        not args.random_encoder,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    key = f"{args.arm}_{args.pool}_s{args.seed}" + (f"_r{args.rho}" if args.arm == "replay" else "")
    (args.out / f"{key}.json").write_text(
        json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({k: v for k, v in out.items() if k != "matrix"}, ensure_ascii=False), flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
