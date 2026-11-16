r"""**无标签**的状态评估：不再要求网络输出我编的类别。

## 用户 2026-11-16 的批评

> 生物脑的神经元本身就不做任何具体的任务，只是一个状态；
> 我们一直追求它输出可供人类语言解释的信息。

今天所有实验都用**我编的标签**（隐状态类别、LCA 节点）当判据。那测的是
"它能不能说我的话"，不是"它自己认识到了什么"。

## 三个量

| 量 | 定义 | 说明 |
| :--- | :--- | :--- |
| **隐状态探针** | 从 `s_t` 线性预测**流本身生成的隐变量** `s` | 探针是**测量工具**，不是架构；天花板接近 1，所以"0.3 算好还是差"答得出来 |
| **有效维度 PR** | `(Σλ)²/Σλ²` 的状态协方差 | 用了几个方向，不需要标签 |
| **时间相干** | `|Δt|` 与状态距离的秩相关 | 相近经历是否落在相近状态 |

next-token 探针**照报但不作判据**：观测是从 `pref[s]` 抽出来的随机 token，
即使完美知道状态也猜不准，天花板低且未知（见 `prereg_amend1.md`）。

## 先验证这套指标自己

拿今天**已知差异极大**的两个配置：

| 配置 | 度 | 分类准确率（旧指标） |
| :--- | ---: | ---: |
| 环（局部） | 4 | **0.545** |
| 随机稀疏 | 32 | **0.126（= 随机）** |

**判据 V-1′：隐状态探针必须把这两个分开（环 − 随机 ≥ 0.05）**（见 `prereg_amend1.md`）。
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

from scripts.instrument import (
    calibrate,
    linear_probe_acc,
    majority_baseline,
    mlp_probe,
    probe,
    probe_gain,
    seeded_dirichlet,
    validity,
)
from scripts.structure_ladder import window_features

OUT = _pathlib.Path("sweep_results/state_info")
TAUS = (2.0, 16.0)
V, K_LATENT = 64, 16


def make_stream(n_seq: int, T: int, seed: int, dev: str):
    """一条长流：状态按固定节律轮转，观测由状态生成。

    返回 `(toks, st)`。`st` 是**流自己的隐变量**，不是人编的标签 ——
    `toks` 就是从它生成的，留着它不引入任何新假设。
    """
    g = torch.Generator().manual_seed(seed)
    pref = seeded_dirichlet(3.0, K_LATENT).to(dev)
    seg = max(1, T // K_LATENT)
    phase = torch.randint(0, K_LATENT, (n_seq,), generator=g).to(dev)
    toks = torch.zeros(n_seq, T, dtype=torch.long, device=dev)
    st = torch.zeros(n_seq, T, dtype=torch.long, device=dev)
    for t in range(T):
        s = (phase + t // seg) % K_LATENT
        st[:, t] = s
        toks[:, t] = torch.multinomial(pref[s].cpu(), 1, generator=g).squeeze(-1).to(dev)
    return toks, st


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
            toks, st = make_stream(args.train_seq + args.test_seq, args.seq, 10 + sd, dev)
            w_in, pre, wrec = build(kind, args.n, sd, dev)
            # **标定走仪器**：等比细网格 + 窗口判定，失败就放弃该臂
            ok, sc, _ = calibrate(
                lambda k, _wi=w_in, _p=pre, _w=wrec, _t=toks: run(_wi * k, _p, _w, _t, dev)[2]
            )
            if not ok:
                print(f"  {kind} seed {sd} — 标定失败，该臂无效", flush=True)
                continue
            feat, idx, fr = run(w_in * sc, pre, wrec, toks, dev)
            n_keep = feat.shape[1]

            # ---- 主观测量：隐变量探针（天花板接近 1）----
            X = feat.reshape(-1, feat.shape[-1])
            Y = st[:, idx].reshape(-1)
            cut = args.train_seq * n_keep
            acc_lat, lam_lat = probe(X, Y, cut, K_LATENT)
            floor_lat = majority_baseline(Y[:cut], Y[cut:], K_LATENT)
            n_test_lat = int(Y.numel() - cut)
            # 非线性探针：超参与 dim_ladder.mlp 逐字相同（此前"环 ≫ 随机"的证据来源）
            acc_mlp, _mlp_runs = mlp_probe(X, Y, cut, K_LATENT)
            # 任务自身的天花板：最近 8 个 token 的 one-hot，完全不经过网络
            wf = window_features(toks, L=8, stride=8)
            acc_win = linear_probe_acc(wf.reshape(-1, wf.shape[-1]), Y, cut, K_LATENT)
            del wf

            # ---- 次要：next-token 探针（天花板低且未知，不作判据）----
            tok: dict = {}
            for k in (1, 4):
                if k >= n_keep:
                    tok[k] = float("nan")
                    continue
                Xk = feat[:, : n_keep - k].reshape(-1, feat.shape[-1])
                Yk = toks[:, torch.stack([idx[i] + k for i in range(n_keep - k)])].reshape(-1)
                cutk = args.train_seq * (n_keep - k)
                tok[k], tok[f"lam{k}"] = probe(Xk, Yk, cutk, V)
                tok[f"floor{k}"] = majority_baseline(Yk[:cutk], Yk[cutk:], V)
                del Xk, Yk

            ok_v, why = validity(acc_lat, fr, floor_lat, n_test_lat)
            if not ok_v:
                print(f"  {kind} seed {sd} — 仪器无效：{why}", flush=True)
                continue
            rows.append(
                {
                    "scale": sc,
                    "firing": fr,
                    "pr": participation(feat),
                    "coh": temporal_coherence(feat),
                    "acc_latent": acc_lat,
                    "lam_latent": lam_lat,
                    "floor_latent": floor_lat,
                    "gain_latent": probe_gain(acc_lat, floor_lat),
                    "acc_mlp": acc_mlp,
                    "acc_win": acc_win,
                    "tok": tok,
                }
            )
            g_lat = probe_gain(acc_lat, floor_lat)
            print(
                f"  {kind} seed {sd}  scale {sc:5.2f}  发放 {fr:.4f}  "
                f"隐状态 线性 {acc_lat:.4f} / MLP {acc_mlp:.4f} / 窗口天花板 {acc_win:.4f}"
                f"（底板 {floor_lat:.4f}，增益 {g_lat:+.3f}，λ {lam_lat:g}）",
                flush=True,
            )
            del feat, X, Y
            torch.cuda.empty_cache()
        if not rows:
            print(f"  {kind:6s} ✗ 所有种子均被仪器闸拒绝，该臂无效", flush=True)
            continue
        m = len(rows)
        res["cells"][kind] = {
            "n": m,
            "firing": sum(r["firing"] for r in rows) / m,
            "pr": sum(r["pr"] for r in rows) / m,
            "coh": sum(r["coh"] for r in rows) / m,
            "acc_latent": sum(r["acc_latent"] for r in rows) / m,
            "lam_latent": sum(r["lam_latent"] for r in rows) / m,
            "floor_latent": sum(r["floor_latent"] for r in rows) / m,
            "gain_latent": sum(r["gain_latent"] for r in rows) / m,
            "acc_mlp": sum(r["acc_mlp"] for r in rows) / m,
            "acc_win": sum(r["acc_win"] for r in rows) / m,
            "tok1": sum(r["tok"][1] for r in rows) / m,
            "tok1_floor": sum(r["tok"]["floor1"] for r in rows) / m,
            "tok4": sum(r["tok"][4] for r in rows) / m,
            "tok4_floor": sum(r["tok"]["floor4"] for r in rows) / m,
            "runs": [r["acc_latent"] for r in rows],
            "scales": [r["scale"] for r in rows],
        }
        c = res["cells"][kind]
        print(
            f"  {kind:6s} n={m}  发放 {c['firing']:.4f}  PR {c['pr']:7.1f}  相干 {c['coh']:+.3f}  "
            f"隐状态 {c['acc_latent']:.4f}（底板 {c['floor_latent']:.4f}）  "
            f"next-token {c['tok1']:.4f}（底板 {c['tok1_floor']:.4f}）",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )
    if "ring" not in res["cells"] or "random" not in res["cells"]:
        print("")
        print("✗ 至少一个臂无效，V-1′ 无法判定")
        return 1
    r, rnd = res["cells"]["ring"], res["cells"]["random"]
    print("")
    print("判（V-1′：隐状态探针必须把环与随机分开，差 ≥0.05）：")
    for key, label in (
        ("acc_latent", "隐状态 线性探针"),
        ("acc_mlp", "隐状态 MLP探针"),
        ("acc_win", "窗口天花板(不经过网络)"),
        ("gain_latent", "隐状态归一化增益"),
        ("pr", "有效维度PR"),
        ("coh", "时间相干"),
        ("tok1", "next-token准确率"),
    ):
        print(f"  {label:18s} 环 {r[key]:+.4f}  随机 {rnd[key]:+.4f}  差 {r[key] - rnd[key]:+.4f}")
    # **两族探针都报，不拿通过的那一个顶替**。V-1′ 的原文没写探针族，
    # 这是判据的缺口，照实记在这里，不事后改。
    g_lin = r["acc_latent"] - rnd["acc_latent"]
    g_mlp = r["acc_mlp"] - rnd["acc_mlp"]
    print("")
    print(f"  V-1′（线性探针）：差 {g_lin:+.4f} ⇒ {'过' if g_lin >= 0.05 else '不过'}")
    print(f"  V-1′（MLP 探针）：差 {g_mlp:+.4f} ⇒ {'过' if g_mlp >= 0.05 else '不过'}")
    if (g_lin >= 0.05) != (g_mlp >= 0.05):
        print("  **两族探针结论相反**：无标签指标没有摆脱读出瓶颈，它只是把读出换了个名字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
