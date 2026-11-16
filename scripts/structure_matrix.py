"""结构 × 读出 的 2×2×3 矩阵：**把「三维行不行」拆成三个可分的变量**。

## 为什么上一轮的结论不能要

上一轮说"三维比一维低 8pp"。但三件事没控制住：

| 混淆 | 实测 | 后果 |
| :--- | :--- | :--- |
| **递归驱动** | 一维 1.20 / 三维 1.81 / 四维 2.41 | 高维天然多 2 倍递归输入；本仓历史：2.16× 掉 4.9pp |
| **读出非线性** | 只有线性（块平均 + 岭回归） | 若三维把状态编码在非线性码里，线性读出看不见 |
| **读出范围** | 只有局部块 | 局部是我自己加的假设，没验证过 |

## 本矩阵

- 三种拓扑：一维环 / 三维网格 / 四维网格；
- **递归驱动全部归一到 1.20**；
- 输入映射一律**打乱**（上一轮已证地形映射在三维有害，这里不再引入第三个变量）；
- 读出：`{全局, 局部} × {线性, MLP}` 四种。

判据：

| 编号 | 判据 |
| :--- | :--- |
| **M-0** | 窗口 32-token 对照显著高于随机（任务可解） |
| **M-1** | 在**任一**读出配置下，三维或四维 > 一维 **≥5pp** —— 证明"维度有用，只是要配对的读出" |
| **M-2** | 同一拓扑下，MLP 比线性高 ≥5pp —— 证明"线性读出低估了结构" |

M-1 不过 ⇒ **三维这条路收口**，理由不再是"测出来差"，而是"三个变量都控住之后仍然不赢"。
"""

from __future__ import annotations

# ruff: noqa: N806  -- 矩阵推导里 X/W/A/T/K 是惯例
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.structure_ladder import K, Res, make_pref, make_task, window_features

OUT = _pathlib.Path("sweep_results/structure_matrix")
TOPOS = [("D", "1D环"), ("3h", "3D网格"), ("4h", "4D网格")]


def ridge(feats, st, n_train, k=K):
    X = feats.reshape(-1, feats.shape[-1])
    Y = st.reshape(-1)
    cut = n_train * st.shape[1]
    Xtr, Xte = X[:cut], X[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Yh = torch.nn.functional.one_hot(Y[:cut], k).float()
    A = Xtr.T @ Xtr + 1e-2 * torch.eye(Xtr.shape[1], device=X.device)
    W = torch.linalg.solve(A, Xtr.T @ Yh)
    return float(((Xte @ W).argmax(-1) == Y[cut:]).float().mean())


def mlp(feats, st, n_train, k=K, steps=400, hidden=128):
    X = feats.reshape(-1, feats.shape[-1])
    Y = st.reshape(-1)
    cut = n_train * st.shape[1]
    Xtr, Xte = X[:cut], X[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ytr, Yte = Y[:cut], Y[cut:]
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, k),
    ).to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        perm = torch.randperm(Xtr.shape[0], device=X.device)[:4096]
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(net(Xtr[perm]), Ytr[perm])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((net(Xte).argmax(-1) == Yte).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=384)
    ap.add_argument("--test-seq", type=int, default=96)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    pref = make_pref(0, args.alpha).to(dev)
    tr_toks, tr_st, _ = make_task(args.train_seq, args.seq, 1, dev, pref)
    te_toks, te_st, _ = make_task(args.test_seq, args.seq, 2, dev, pref)
    toks = torch.cat([tr_toks, te_toks])
    st = torch.cat([tr_st, te_st])[:, :: args.stride]

    res: dict = {"n": args.n, "gain": args.gain, "cells": {}}
    # 对照
    wf = window_features(toks, L=32, stride=args.stride)
    ctl = ridge(wf, st, args.train_seq)
    res["control"] = {"window32": ctl, "random": 1.0 / K}
    print(f"  窗口32 对照 {ctl:.3f}   随机 {1 / K:.3f}", flush=True)

    for arm, label in TOPOS:
        pool = Res(arm, args.n, dev, shuffle_map=True, matched_gain=args.gain)
        for scope in ("global", "local"):
            feat = pool.run(toks, args.stride, scope=scope)
            for dec_name, dec in (("linear", ridge), ("mlp", mlp)):
                acc = dec(feat, st, args.train_seq)
                key = f"{arm}/{scope}/{dec_name}"
                res["cells"][key] = {"label": label, "scope": scope, "decode": dec_name, "acc": acc}
                print(f"  {label:6s} {scope:6s} {dec_name:6s} acc {acc:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    def best(arm):
        return max(v["acc"] for k, v in res["cells"].items() if k.startswith(arm + "/"))

    b1, b3, b4 = best("D"), best("3h"), best("4h")
    print("\n判：")
    print(f"  M-0 窗口32 {ctl:.3f} ≥ {1 / K + 0.2:.3f} ⇒ {'过' if ctl >= 1 / K + 0.2 else '不过'}")
    gain = max(b3, b4) - b1
    print(
        f"  M-1 最佳高维 {max(b3, b4):.3f} − 一维最佳 {b1:.3f} = {gain:+.3f} ≥0.05 ⇒ "
        f"{'过（维度有用，要配对的读出）' if gain >= 0.05 else '不过（三维收口）'}"
    )
    for arm, label in TOPOS:
        lin = res["cells"][f"{arm}/local/linear"]["acc"]
        ml = res["cells"][f"{arm}/local/mlp"]["acc"]
        print(f"  M-2 {label} 局部 MLP−线性 = {ml - lin:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
