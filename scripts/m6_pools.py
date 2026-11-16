"""M6p：多池竞争 —— 给"图案"一个能存在的地方。

预注册：`research/sel/prereg-m6-pools-v1.md`

## 改什么（都不是调参，是加结构）

1. **多池**：32×32 环面切成 K 个**连续**域，同池候选的接线分数 ×(1+β)；
2. **交叉抑制**：`cur_i += g_inh·(φ_{k(i)} − φ)` —— 经典 WTA；
3. **慢增益**：内增益更新率 `eta_gain` 可调（-1 = 自发期不施加，= M6 原样）。

`K=1, β=0, g_inh=0, eta_gain=-1` 时逐项退化成 M6。
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import make_task
from scripts.m7_history import A_HIGH, A_LOW, T_HIST

OUT = _pathlib.Path("sweep_results/m6_pools")
L, RAD, V, DEG = 32, 8, 64, 8
N = L * L
T_SPONT = 256
T_LATE = 64
I_PACE = 0.5
G_FIELD = 1.0
B_BATCH = 32


def pool_id(dev: str, K: int):
    """连续空间分区。K=1/2/4/8，池大小均分。"""
    idx = torch.arange(N)
    if K <= 1:
        q = torch.zeros(N, dtype=torch.long)
    elif K == 2:
        q = (idx % L >= L // 2).long()
    elif K == 4:
        q = (idx % L >= L // 2).long() + 2 * (idx // L >= L // 2).long()
    else:
        q = (idx % L // (L // 4)).long() + 4 * (idx // L >= L // 2).long()
    return q.to(dev)


def build_pools(seed: int, dev: str, K: int = 1, beta: float = 0.0):
    """按环面距离建接线；同池候选分数 ×(1+β)。

    β=0 时 RNG 消耗与 `turnover.build_net` 逐项相同（可复现 M6 的图）。
    """
    gen = torch.Generator().manual_seed(seed)
    idx = torch.arange(N)
    x = (idx % L).float().to(dev)
    y = (idx // L).float().to(dev)
    ax = torch.arange(-RAD, RAD + 1).float()
    ox, oy = torch.meshgrid(ax, ax, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev), oy.reshape(-1).to(dev)
    C = ox.numel()
    cx = (x.unsqueeze(1) + ox.unsqueeze(0)) % L
    cy = (y.unsqueeze(1) + oy.unsqueeze(0)) % L
    d = torch.sqrt((ox.unsqueeze(0) ** 2 + oy.unsqueeze(0) ** 2).expand(N, C))
    w = torch.exp(-d / 4.0)
    w[:, (ox == 0) & (oy == 0)] = 0.0
    score = w * (1.0 + 0.5 * torch.rand(N, C, generator=gen).to(dev))
    lin = (cy * L + cx).long()
    pid = pool_id(dev, K)
    if beta > 0.0 and K > 1:
        same = pid.unsqueeze(1) == pid[lin]
        score = score * torch.where(same, 1.0 + beta, 1.0)
    idx_top = score.topk(DEG, dim=1).indices
    pre = lin.gather(1, idx_top)
    wrec = torch.rand(N, DEG, generator=gen).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    w_in = (torch.randn(V, N, generator=gen) / N**0.5).to(dev)
    return w_in, pre, wrec, pid


def within_frac(pre, pid) -> float:
    return float((pid[pre] == pid.unsqueeze(1)).float().mean())


@torch.no_grad()
def run_spont(
    seed: int,
    alpha: float,
    hist_seed: int,
    scale: float,
    p: float,
    dev: str,
    K: int = 1,
    beta: float = 0.0,
    g_inh: float = 0.0,
    eta_gain: float = -1.0,
    t_hist: int = T_HIST,
    gain_route: str = "field",
    t_spont: int = T_SPONT,
    v0: torch.Tensor | None = None,
    hist_in: torch.Tensor | None = None,
    net: tuple | None = None,
    stdp: float = 0.0,
    alive: torch.Tensor | None = None,
    record_len: int = T_LATE,
    v_add: torch.Tensor | None = None,
    g_field: float = G_FIELD,
):
    """历史 → 零输入自发。返回 `(每个体后期发射率 (B,N), 后期快照 (B,N,T_LATE))`。"""
    if net is None:
        w_in, pre, wrec, pid = build_pools(seed, dev, K, beta)
    else:
        w_in, pre, wrec, pid = net
    w_in = w_in * scale
    alive = torch.ones(N, device=dev) if alive is None else alive
    gen = torch.Generator().manual_seed(seed + 31337)
    pace = (torch.rand(N, generator=gen) < p).float().to(dev) * I_PACE

    counts = torch.bincount(pid, minlength=K).float().clamp(min=1.0)
    onehot = torch.zeros(N, K, device=dev)
    onehot[torch.arange(N, device=dev), pid] = 1.0 / counts[pid]

    # 历史期的增益更新率固定 2.0（= M6/M7 的 `run_seg`），
    # `eta_gain` 只支配**自发期**；-1 = 自发期不施加增益（M6 原样）
    eta_hist = 2.0 if eta_gain < 0.0 else eta_gain
    hist = hist_in
    if hist is None and t_hist > 0:
        hist, _ = make_task(B_BATCH, t_hist, 8, hist_seed, dev, alpha)
    if hist is not None:
        t_hist = hist.shape[1]  # **外部历史的长度说了算**
    # **batch 必须跟着历史走** —— 写死 32 时外部给 8 条历史会形状不匹配
    B = hist.shape[0] if hist is not None else B_BATCH
    v = torch.zeros((B, N), device=dev)
    z = torch.zeros((B, N), device=dev)
    ref = torch.zeros((B, N), device=dev)
    gain = torch.ones(N, device=dev)
    f_i = torch.full((N,), 0.08, device=dev)
    f_run = 0.08

    for t in range(t_hist):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phk = z @ onehot
        phi_g = phk.mean()  # 全局场（与 run_seg 的 float(z.mean()) 一致）
        if eta_hist >= 0.0 and gain_route == "all":
            rec = rec * gain.unsqueeze(0)  # 增益乘**总**输入，不止 w_in
        cur = w_in[hist[:, t]] * gain.unsqueeze(0) - g_field * (f_run / 0.08) * phi_g
        if g_inh != 0.0:
            cur = cur + g_inh * (phk[:, pid] - phk.mean(1, keepdim=True))
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        if stdp > 0.0:
            # STDP 原地改写 wrec —— 调用方传入同一张量即可跨阶段累积
            from scripts.m7_history import a_minus, a_plus

            e = a_plus * (nz.unsqueeze(2) * z[:, pre]).mean(0) - a_minus * (
                z.unsqueeze(2) * nz[:, pre]
            ).mean(0)
            upd = (wrec + stdp * e).clamp(min=0.0)
            wrec.copy_(upd * (1.2 / upd.sum(1, keepdim=True).clamp(min=1e-9)))
        if eta_hist >= 0.0:
            f_i = 0.9 * f_i + 0.1 * z.mean(0)
            if t % 4 == 3:
                gain = torch.clamp(gain * torch.exp(eta_hist * (0.08 - f_i)), 0.2, 5.0)
        f_run = 0.9 * f_run + 0.1 * float(z.mean())

    if v0 is not None:
        v = v0.clone()  # 吸引子计数用：绕过历史，直接给初值
    if v_add is not None:
        v = v + v_add  # 扰动（M10 用）：在输入停止的那一刻加

    rates = torch.zeros((B, N), device=dev)
    early = torch.zeros((B, N), device=dev)  # 自发期**前** T_LATE 步（Q-4 的"停没停"）
    snaps = []
    for step in range(t_spont):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phk = z @ onehot
        phi_g = phk.mean()
        cur = pace.unsqueeze(0) - g_field * phi_g
        if eta_gain >= 0.0:
            cur = cur * gain.unsqueeze(0)
            if gain_route == "all":
                rec = rec * gain.unsqueeze(0)
        if g_inh != 0.0:
            cur = cur + g_inh * (phk[:, pid] - phk.mean(1, keepdim=True))
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        if eta_gain >= 0.0:
            f_i = 0.9 * f_i + 0.1 * z.mean(0)
            if step % 4 == 3:
                gain = torch.clamp(gain * torch.exp(eta_gain * (0.08 - f_i)), 0.2, 5.0)
        if step < record_len:
            early += z
        if step >= t_spont - T_LATE:
            rates += z
        if step >= t_spont - record_len:
            snaps.append(z)  # 默认 T_LATE；M9–M12 要更长的记录就调 `record_len`
    del w_in, pre, wrec, hist
    torch.cuda.empty_cache()
    # 第三个返回是自发期**前** T_LATE 步的率，用于 Q-4（窗口内停没停）
    return (
        rates / T_LATE,
        torch.stack(snaps, 2),
        early / max(record_len, 1),
    )  # (B,N,T_LATE)，dim=1 会变 (B,T,N)


def pool_rates(r: torch.Tensor, pid, K: int) -> torch.Tensor:
    """(B,N) -> (B,K)"""
    idx = pid.unsqueeze(0).expand(r.shape[0], -1)
    out = torch.zeros(r.shape[0], K, device=r.device)
    out.scatter_add_(1, idx, r)
    counts = torch.bincount(pid, minlength=K).float().clamp(min=1.0)
    return out / counts


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (0.5 * (a.norm() + b.norm()) + 1e-12))


def pr_of(cov: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(cov.double().cpu()).clamp(min=0)
    return float((ev.sum() ** 2) / max(float((ev**2).sum()), 1e-24))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ks", default="1,4,8")
    ap.add_argument("--betas", default="0")
    ap.add_argument("--ginhs", default="0,2,4")
    ap.add_argument("--etas", default="-1,0.2,2.0")
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    Ks = [int(x) for x in args.ks.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    ginhs = [float(x) for x in args.ginhs.split(",")]
    etas = [float(x) for x in args.etas.split(",")]

    arms = []
    for K in Ks:
        for beta in betas:
            if K == 1 and beta != 0.0:
                continue
            for g in ginhs:
                if K == 1 and g != 0.0:
                    continue  # K=1 时 g_inh 是恒等零算子
                for e in etas:
                    arms.append((K, beta, g, e))

    res: dict = {"arms": {}, "n_seeds": args.seeds, "p": args.p, "scale": args.scale}
    for K, beta, g, e in arms:
        key = f"K={K}/b={beta:g}/g={g:g}/eta={e:g}"
        d_hist, d_null, wins, winners, dom, frs, sds, agree = [], [], [], [], [], [], [], []
        cov_b, cov_i, n_b, n_i = None, None, 0, 0
        pid = pool_id(dev, K)
        for sd in range(args.seeds):
            ra, za, _ = run_spont(sd, A_LOW, 100 + sd, args.scale, args.p, dev, K, beta, g, e)
            rb, zb, _ = run_spont(sd, A_HIGH, 100 + sd, args.scale, args.p, dev, K, beta, g, e)
            rn, zn, _ = run_spont(sd, A_LOW, 900 + sd, args.scale, args.p, dev, K, beta, g, e)
            frs.append(float(ra.mean()))
            sds.append(float((ra - ra.mean(1, keepdim=True)).std()))
            if K > 1:
                # **个体** = 一个网络；batch 的 32 条只是同一个体的 32 个样本 ⇒
                # 状态取 batch 均值，个体内的分散度另记（若胜者随样本乱跳，那是噪声不是状态）
                Ra, Rb, Rn = (
                    pool_rates(ra.mean(0, keepdim=True), pid, K)[0],
                    pool_rates(rb.mean(0, keepdim=True), pid, K)[0],
                    pool_rates(rn.mean(0, keepdim=True), pid, K)[0],
                )
                d_hist.append(rel(Ra - Ra.mean(), Rb - Rb.mean()))
                d_null.append(rel(Ra - Ra.mean(), Rn - Rn.mean()))
                wins.append(1 if d_hist[-1] > d_null[-1] else 0)
                win = int(Ra.argmax())
                winners.append(win)
                others = float((Ra.sum() - Ra[win]) / max(K - 1, 1))
                dom.append(float(Ra[win]) / max(others, 1e-12))
                per_b = pool_rates(ra, pid, K).argmax(1)
                agree.append(float((per_b == win).float().mean()))
            # 线性谱：时间轴残余参与比。**按神经元中心化**（M6 的 `za.mean(0)`）
            zmb = za.mean(0)
            for zz, which in ((zmb, "b"), (za, "i")):
                zc = (zz - zz.mean(-2, keepdim=True)).reshape(-1, T_LATE).double()
                c = zc.T @ zc
                if which == "b":
                    cov_b = c if cov_b is None else cov_b + c
                    n_b += zc.shape[0]
                else:
                    cov_i = c if cov_i is None else cov_i + c
                    n_i += zc.shape[0]
            del ra, rb, rn, za, zb, zn
        mh, mn = sum(d_hist) / max(len(d_hist), 1), sum(d_null) / max(len(d_null), 1)
        ratio = mh / max(mn, 1e-12) if K > 1 else float("nan")
        w = sum(wins)
        res["arms"][key] = {
            "K": K,
            "beta": beta,
            "g_inh": g,
            "eta_gain": e,
            "firing": sum(frs) / len(frs),
            "spatial_sd": sum(sds) / len(sds),
            "d_hist": mh,
            "d_null": mn,
            "ratio_pattern": ratio,
            "wins_pattern": w,
            "n": args.seeds,
            "winners": winners,
            "distinct_winners": len(set(winners)),
            "dominance": sum(dom) / max(len(dom), 1),
            "winner_agreement": sum(agree) / max(len(agree), 1),
            "pr_time_batch": pr_of(cov_b / max(n_b, 1)) if cov_b is not None else None,
            "pr_time_indiv": pr_of(cov_i / max(n_i, 1)) if cov_i is not None else None,
        }
        v0 = res["arms"][key]
        print(
            f"  {key:24s} 发放 {v0['firing']:.4f}  SD {v0['spatial_sd']:.4f}  "
            f"图案比 {ratio:.2f}x({w}/{args.seeds})  胜者 {winners}  "
            f"支配 {v0['dominance']:.1f}x  PR(b) {v0['pr_time_batch']:.2f}  "
            f"PR(i) {v0['pr_time_indiv']:.2f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    cc = res["arms"]
    base = cc.get("K=1/b=0/g=0/eta=-1")
    if base is None:
        print("  P-0 缺基线 => 作废")
        return 0
    p0 = abs(base["firing"] - 0.1420) <= 0.010
    print(
        f"  P-0 仪器复现 M6 (0.1420±0.010)：{base['firing']:.4f} => {'过' if p0 else '不过 => 作废'}"
    )
    if not p0:
        return 0
    ok1 = [k for k, v in cc.items() if v["K"] > 1 and 0.01 <= v["firing"] <= 0.20]
    print(f"  P-1 存在自发活动（K>1）：{len(ok1)} 档 => {'过' if ok1 else '不过 => 作废'}")
    if not ok1:
        return 0
    kb = max(ok1, key=lambda k: cc[k]["dominance"])
    v = cc[kb]
    p2 = v["dominance"] >= 2.0
    print(
        f"  P-2 竞争发生（支配 ≥2×）：{kb} 支配 {v['dominance']:.2f}x => {'过' if p2 else '不过'}"
    )
    n_ok3 = sum(1 for k2 in ok1 if cc[k2]["distinct_winners"] > 1)
    p3 = v["distinct_winners"] > 1 and n_ok3 >= 4 * len(ok1) / 6
    print(
        f"  P-3 胜者不恒定：{kb} 胜者 {v['winners']}（不同 {v['distinct_winners']} 个）；"
        f"过线档 {n_ok3}/{len(ok1)} => {'过' if p3 else '不过'}"
    )
    kb4 = max(ok1, key=lambda k: cc[k]["ratio_pattern"])
    v4 = cc[kb4]
    p4 = v4["ratio_pattern"] > 1.5 and v4["wins_pattern"] >= 5
    print(
        f"  P-4 图案带历史（>1.5, 5/6）：{kb4} {v4['ratio_pattern']:.2f}x "
        f"{v4['wins_pattern']}/{v4['n']} => {'过' if p4 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
