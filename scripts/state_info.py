r"""**无标签**的状态评估：不再要求网络输出我编的类别。

## 用户 2026-11-16 的批评

> 生物脑的神经元本身就不做任何具体的任务，只是一个状态；
> 我们一直追求它输出可供人类语言解释的信息。

今天所有实验都用**我编的标签**（隐状态类别、LCA 节点）当判据。那测的是
"它能不能说我的话"，不是"它自己认识到了什么"。

## 三个无标签量

| 量 | 定义 | 说明 |
| :--- | :--- | :--- |
| **预测 R²** | 从 `s_t` 线性预测 `x_{t+k}`（k 步后的 token）的方差解释率 | 线性探测是**测量工具**，不是架构；它给的是"状态里有多少关于未来的信息"的**下界** |
| **有效维度 PR** | `(Σλ)²/Σλ²` 的状态协方差 | 用了几个方向，不需要标签 |
| **时间相干** | `\|Δt\|` 与状态距离的秩相关 | 相近经历是否落在相近状态 |

## 先验证这套指标自己

拿今天**已知差异极大**的两个配置：

| 配置 | 度 | 分类准确率（旧指标） |
| :--- | ---: | ---: |
| 环（局部） | 4 | **0.545** |
| 随机稀疏 | 32 | **0.126（= 随机）** |

**判据 V-1：无标签指标也必须把这两个分开**（预测 R² 的差 ≥0.05）。
分不开 ⇒ 指标本身有问题，先修指标，不拿它去判几何。
"""

from __future__ import annotations

# ruff: noqa: N803, N806  -- 矩阵推导里 X/W/A/T/K 是惯例
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.instrument import calibrate, linear_probe_acc, validity

OUT = _pathlib.Path("sweep_results/state_info")
TAUS = (2.0, 16.0)
V = 64


def make_stream(n_seq: int, T: int, seed: int, dev: str):
    """一条长流：状态按固定节律轮转，观测由状态生成。**没有'类别'这回事，只有时间序列。**"""
    g = torch.Generator().manual_seed(seed)
    K = 16
    pref = torch.distributions.Dirichlet(torch.full((K,), 3.0)).sample((K,)).to(dev)
    seg = max(1, T // K)
    phase = torch.randint(0, K, (n_seq,), generator=g).to(dev)
    toks = torch.zeros(n_seq, T, dtype=torch.long, device=dev)
    for t in range(T):
        s = (phase + t // seg) % K
        toks[:, t] = torch.multinomial(pref[s].cpu(), 1, generator=g).squeeze(-1).to(dev)
    return toks


def build(kind: str, n: int, seed: int, dev: str, deg: int = 32):
    g = torch.Generator().manual_seed(seed)
    w_in = (torch.randn(V, n, generator=g) / n**0.5).to(dev)
    if kind == "ring":
        off = torch.tensor([-1, -2, 1, 2]).to(dev)
        pre = (torch.arange(n).to(dev).unsqueeze(1) + off.unsqueeze(0)) % n
        wrec = torch.rand(n, 4, generator=g).to(dev)
    else:
        pre = torch.randint(0, n, (n, deg), generator=g).to(dev)
        wrec = torch.rand(n, deg, generator=g).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    return w_in, pre, wrec


@torch.no_grad()
def run(w_in, pre, wrec, toks, dev, stride=8, alpha=0.95, theta=0.5):
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, idx, rate = [], [], 0.0
    for t in range(T):
        rec = torch.zeros_like(z)
        for k in range(pre.shape[1]):
            rec = rec + z[:, pre[:, k]] * wrec[:, k].unsqueeze(0)
        nv = alpha * v + w_in[toks[:, t]] + rec - z * theta
        fresh = ((nv - theta) / theta > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
            idx.append(t)
    return torch.stack(feats, 1), torch.tensor(idx, device=dev), rate / T


def pred_r2(feats, toks, idx, k: int, cut: int, lam: float | None = None):
    """从 s_t 线性预测 x_{t+k} 的方差解释率（下界）。"""
    n_steps = feats.shape[1]
    keep = [i for i in range(n_steps) if i + k < n_steps]
    if not keep:
        return 0.0
    X = torch.stack([feats[:, i] for i in keep], 1).reshape(-1, feats.shape[-1])
    tidx = torch.stack([idx[i] + k for i in keep])
    Y = toks[:, tidx].reshape(-1)
    cut_row = cut * len(keep)
    Xtr, Xte, Ytr, Yte = X[:cut_row], X[cut_row:], Y[:cut_row], Y[cut_row:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Yh = torch.nn.functional.one_hot(Ytr, V).float()
    # **λ 必须相对 XᵀX 的对角量级**：标准化后对角 ≈ n_samples，λ=1 等于无正则 ⇒ 发散
    if lam is None:
        lam = 1e-2 * Xtr.shape[0]
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=X.device)
    W = torch.linalg.solve(A, Xtr.T @ Yh)
    pred = Xte @ W
    Yte_h = torch.nn.functional.one_hot(Yte, V).float()
    mse = ((pred - Yte_h) ** 2).mean()
    var = Yte_h.var(dim=0, unbiased=False).mean()
    return float(1 - mse / (var + 1e-9))


def participation(feats):
    X = feats.reshape(-1, feats.shape[-1]).float()
    X = X - X.mean(0, keepdim=True)
    if X.shape[0] < X.shape[1]:
        ev = torch.linalg.eigvalsh(X @ X.T / X.shape[0]).clamp(min=0)
    else:
        ev = torch.linalg.eigvalsh(X.T @ X / X.shape[0]).clamp(min=0)
    s1, s2 = float(ev.sum()), float((ev**2).sum())
    return (s1 * s1) / max(s2, 1e-12)


def temporal_coherence(feats, n_pair: int = 2000, seed: int = 0):
    """把状态摊平成 (seq*step, F)，看"时间上近"是否"状态上近"。"""
    X = feats.reshape(-1, feats.shape[-1])
    seq_len = feats.shape[1]
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    i = torch.randint(0, n, (n_pair,), generator=g).to(X.device)
    j = torch.randint(0, n, (n_pair,), generator=g).to(X.device)
    dt = ((i // seq_len) - (j // seq_len)).abs().float()  # 同序列内的步差
    d = (X[i] - X[j]).norm(dim=1)
    okay = dt > 0
    if not bool(okay.any()):
        return 0.0
    a, b = dt[okay], d[okay]
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--train-seq", type=int, default=384)
    ap.add_argument("--test-seq", type=int, default=96)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"cells": {}}

    for kind in ("ring", "random"):
        rows = []
        for sd in seeds:
            toks = make_stream(args.train_seq + args.test_seq, args.seq, 10 + sd, dev)
            w_in, pre, wrec = build(kind, args.n, sd, dev)
            # **标定走仪器**：等比细网格 + 窗口判定，失败就放弃该臂
            ok, sc, _ = calibrate(lambda k: run(w_in * k, pre, wrec, toks, dev)[2])
            if not ok:
                print(f"  {kind} seed {sd} — 标定失败，该臂无效", flush=True)
                continue
            feat, idx, fr = run(w_in * sc, pre, wrec, toks, dev)
            n_keep = feat.shape[1]
            probe = {}
            for k in (1, 2, 4):
                if k >= n_keep:
                    probe[k] = float("nan")
                    continue
                Xk = torch.stack([feat[:, i] for i in range(n_keep - k)], 1).reshape(
                    -1, feat.shape[-1]
                )
                Yk = toks[:, torch.stack([idx[i] + k for i in range(n_keep - k)])].reshape(-1)
                probe[k] = linear_probe_acc(Xk, Yk, args.train_seq * (n_keep - k), V)
            r2 = probe
            rows.append(
                {
                    "scale": sc,
                    "firing": fr,
                    "pr": participation(feat),
                    "coh": temporal_coherence(feat),
                    "r2": r2,
                }
            )
            del feat
            torch.cuda.empty_cache()
        if not rows:
            print(f"  {kind:6s} ✗ 所有种子都出窗，该臂无效", flush=True)
            continue
        agg = {
            "firing": sum(r["firing"] for r in rows) / len(rows),
            "pr": sum(r["pr"] for r in rows) / len(rows),
            "coh": sum(r["coh"] for r in rows) / len(rows),
            "r2_1": sum(r["r2"][1] for r in rows) / len(rows),
            "r2_4": sum(r["r2"][4] for r in rows) / len(rows),
        }
        res["cells"][kind] = agg
        print(
            f"  {kind:6s} 发放 {agg['firing']:.4f}  PR {agg['pr']:7.1f}  "
            f"时间相干 {agg['coh']:+.3f}  预测R²(k=1) {agg['r2_1']:+.3f}  (k=4) {agg['r2_4']:+.3f}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    if "ring" not in res["cells"] or "random" not in res["cells"]:
        print("")
        print("✗ 至少一个臂无效，V-1 无法判定")
        return 1
    r, m = res["cells"]["ring"], res["cells"]["random"]
    print("\n判（V-1：无标签指标必须把环与随机分开）：")
    for key, label in (
        ("r2_1", "预测R²(k=1)"),
        ("r2_4", "预测R²(k=4)"),
        ("pr", "有效维度PR"),
        ("coh", "时间相干"),
    ):
        print(f"  {label:14s} 环 {r[key]:+.3f}  随机 {m[key]:+.3f}  差 {r[key] - m[key]:+.3f}")
    ok = (r["r2_1"] - m["r2_1"]) >= 0.05 or (r["r2_4"] - m["r2_4"]) >= 0.05
    print(f"  V-1 {'过（指标可用，可替代分类器）' if ok else '不过（指标分不开，先修指标）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
