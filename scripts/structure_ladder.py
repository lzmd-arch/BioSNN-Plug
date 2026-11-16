"""结构阶梯：**SNN 的内容没用，是因为它太简单吗？**

## 要打破的循环

U-1 测出「SNN 的 `p` 与随机向量等价」。但若结构本身就无法编码状态，
那就等于**用一个注定测不出东西的仪器去证明东西不存在**。
本实验把**结构**当自变量。

## 六臂

| 臂 | 结构 | 与 A 的差别 |
| :--- | :--- | :--- |
| **A** | 平坦单层、无递归、稠密随机输入 | 基线（≈ 现状） |
| **B** | A + **低秩递归**（`w_rec` 打开） | 只加时间整合 |
| **C** | **环拓扑连接** + **地形输入**（token 映射到高斯包） | 只加空间结构 |
| **D** | C + **局部区块读出**（分块平均，不是全局均值） | 空间 + 局部读出 |
| **E** | 同参数量 **GRU** | **正向对照：证明任务可解** |
| **F** | 真状态（oracle） | 上界 |

**网络权重全部冻结**（蓄水池设定），只训练**线性解码器**。
这样测的是「结构里有多少状态信息」，与学习规则无关——避免把两件事混在一起。

## 任务：隐状态生成序列

- `K=8` 个隐状态，每步以 `p_stay` 保持，否则跳到下一个；
- 观测 token 由当前隐状态的多项分布生成（`V=64`）；
- **真值就是隐状态本身**，所以"状态解码准确率"有严格定义。

**平坦无递归的网络在原理上做不了这个任务**（没有记忆），
所以它天生能区分「结构不够」与「结构够」。

## 判据（先落盘）

| 编号 | 判据 | 通过线 |
| :--- | :--- | :--- |
| **S-0** | GRU（E 臂）的状态解码 | **比随机（1/K=12.5%）高 ≥20pp** —— 任务可解 |
| **S-1** | 至少一个结构臂（B/C/D）> 平坦臂 A | **≥5pp** |
| **S-2** | C 臂的**地形映射打乱**（连接不变、映射随机）后增益消失 | 归因成立 |
| **S-3** | 各臂参数量差 | **≤10%** |

**S-0 不过 ⇒ 整个实验作废**，不能拿 GRU 都解不了的任务去判 SNN。
"""

from __future__ import annotations

# ruff: noqa: N803, N806  —— 矩阵推导里用 X/W/A/T 是惯例，改名反而难读
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

OUT = _pathlib.Path("sweep_results/structure_ladder")
V, K = 64, 8
P_STAY = 0.97
TAUS = (2.0, 16.0)


def make_pref(seed: int = 0, alpha: float = 3.0) -> torch.Tensor:
    """**状态→观测的生成分布。训练集与测试集必须共用**，否则两边不是同一个任务。

    `alpha` 越大分布越平 ⇒ 单个 token 携带的状态证据越弱 ⇒ 必须**跨多步积分**
    才能解码。这正是让"结构有没有用"可测的前提。
    """
    torch.manual_seed(seed)
    return torch.distributions.Dirichlet(torch.full((K,), alpha)).sample((K,))


def make_task(n_seq: int, T: int, seed: int, dev: str, pref: torch.Tensor):
    """返回 (tokens (N,T), states (N,T), pref)。"""
    g = torch.Generator().manual_seed(seed)  # multinomial 只收 CPU 生成器
    s = torch.zeros(n_seq, dtype=torch.long)
    states = torch.zeros(n_seq, T, dtype=torch.long)
    toks = torch.zeros(n_seq, T, dtype=torch.long)
    for t in range(T):
        move = torch.rand(n_seq) > P_STAY
        s = torch.where(move, (s + 1) % K, s)
        states[:, t] = s
        probs = pref[s]
        toks[:, t] = torch.multinomial(probs.cpu(), 1, generator=g).squeeze(-1).to(dev)
    return toks.to(dev), states.to(dev), pref.to(dev)


#: 网格臂的维度与拓扑类型
GRID = {
    "3C": ((16, 16, 8), False),  # 三维网格 + 软地形（高斯球）
    "3D": ((16, 16, 8), False),
    "3h": ((16, 16, 8), True),  # 三维网格 + **硬分区**（token 独占一块）
    "4h": ((8, 8, 8, 4), True),  # **四维网格 + 硬分区**
}


def _coords(n: int, dims: tuple[int, ...]):
    """线性下标 → 各维坐标。"""
    idx = torch.arange(n)
    out, stride = [], 1
    for d in reversed(dims):
        out.append((idx // stride) % d)
        stride *= d
    return torch.stack(out[::-1], 1)


def _lin(c, dims) -> torch.Tensor:
    """各维坐标 → 线性下标。"""
    out, stride = torch.zeros(c.shape[0], dtype=torch.long), 1
    for k in range(len(dims) - 1, -1, -1):
        out = out + c[:, k] * stride
        stride *= dims[k]
    return out


class Res:
    """蓄水池：四种结构共用一个前向骨架，差别只在连接与输入。"""

    def __init__(
        self,
        arm: str,
        n: int,
        dev: str,
        seed: int = 0,
        shuffle_map: bool = False,
        matched_gain: float | None = None,
    ):
        g = torch.Generator().manual_seed(seed)  # multinomial 只收 CPU 生成器
        self.arm, self.n, self.dev = arm, n, dev
        self.alpha = 0.95
        self.theta = 0.5
        # 输入：A/B 稠密随机；C/D 一维地形；3C/3D 三维地形（token → 网格里的一个高斯球）
        if arm in GRID:
            dims, hard = GRID[arm]
            self.dims, self.hard = dims, hard
            coords = _coords(n, dims).float()
            if hard:
                # **硬分区**：每个 token 独占一个互不重叠的细胞块（对应关系锐利，不摊薄）
                # 把 V 个 token 均摊到各轴上：每轴取能同时整除 dims[axis] 与剩余 V 的最大因子
                rem, parts = V, []
                for d in dims:
                    best = 1
                    for k in range(min(d, rem), 1, -1):
                        if rem % k == 0 and d % k == 0:
                            best = k
                            break
                    parts.append(best)
                    rem //= best
                assert rem == 1, f"V={V} 无法均摊到 {dims}：parts={parts}"
                cell = torch.zeros(n, dtype=torch.long)
                stride = 1
                ci = coords.long()
                for k in range(len(dims) - 1, -1, -1):
                    cell = cell + (ci[:, k] // (dims[k] // parts[k])) * stride
                    stride *= parts[k]
                w_in = torch.zeros(V, n)
                w_in[cell.clamp(max=V - 1), torch.arange(n)] = 1.0
                # 归一化到 ~每 token 支撑集大小
                cnt = w_in.sum(1, keepdim=True).clamp(min=1)
                self.w_in = (w_in / cnt * 8).to(dev)
            else:
                anchors = torch.tensor(
                    [
                        [i * (dims[0] - 1) / 3, j * (dims[1] - 1) / 3, k * (dims[2] - 1) / 3]
                        for i in range(4)
                        for j in range(4)
                        for k in range(4)
                    ]
                )[:V].float()
                d2 = ((coords.unsqueeze(0) - anchors.unsqueeze(1)) ** 2).sum(-1)
                sigma = 1.5
                w_in = torch.exp(-d2 / (2 * sigma**2))
                self.w_in = (w_in / w_in.sum(1, keepdim=True)).to(dev)
            if shuffle_map:
                self.w_in = self.w_in[torch.randperm(V, generator=g)]
        elif arm in ("C", "D"):
            centers = torch.linspace(0, n - 1, V)
            idx = torch.arange(n).float().unsqueeze(0)
            w_in = torch.exp(-((idx - centers.unsqueeze(1)) ** 2) / (2 * (n / V) ** 2))
            if shuffle_map:  # S-2：连接不变、映射打乱
                perm = torch.randperm(V, generator=g)
                w_in = w_in[perm]
            self.w_in = (w_in * (1.0 / w_in.sum(1, keepdim=True)) * 8).to(dev)
        else:
            self.w_in = (torch.randn(V, n, generator=g) / n**0.5).to(dev)
        # 递归
        self.pre = self.wrec = None
        if arm == "B":  # 低秩全连接
            self.rank = 32
            self.p = (torch.randn(n, self.rank, generator=g) / n**0.5).to(dev)
            self.q = (torch.randn(self.rank, n, generator=g) / self.rank**0.5).to(dev)
        elif arm in GRID:  # 网格：每个轴 ±1 的面邻居（环绕边界）
            dims = GRID[arm][0]
            ci = _coords(n, dims)
            nb = []
            for ax in range(len(dims)):
                for delta in (1, -1):
                    c2 = ci.clone()
                    c2[:, ax] = (c2[:, ax] + delta) % dims[ax]
                    nb.append(_lin(c2, dims))
            self.pre = torch.stack(nb, 1)  # (n, 2*ndim)
            self.wrec = (torch.rand(n, self.pre.shape[1], generator=g) * 0.6).to(dev)
            # 8 块平均矩阵（每轴对半分），与一维臂的 8 块严格对齐
            bid = torch.zeros(n, dtype=torch.long)
            for ax in range(len(dims)):
                bid = bid * 2 + (ci[:, ax] // (dims[ax] // 2))
            m = torch.zeros(n, 8)
            m[torch.arange(n), bid.clamp(max=7)] = 1.0
            self.block_avg = (m / m.sum(0, keepdim=True)).to(dev)
        elif arm in ("C", "D"):  # 环：每个神经元接前后两个邻居
            off = torch.tensor([-1, -2, 1, 2])
            self.pre = (torch.arange(n).unsqueeze(1) + off.unsqueeze(0)) % n
            self.wrec = (torch.rand(n, 4, generator=g) * 0.6).to(dev)
        if self.wrec is not None and matched_gain is not None:
            # **把每个神经元的递归权重和归一到同一个数**：否则高维拓扑天然拿到更多递归驱动，
            # 「三维更差」里就混着「递归更强」。本仓历史：递推/输入 2.16× 掉 4.9pp。
            self.wrec = self.wrec * (matched_gain / self.wrec.sum(1, keepdim=True).clamp(min=1e-6))

    @torch.no_grad()
    def run(self, toks: torch.Tensor, stride: int = 8, scope: str = "local") -> torch.Tensor:
        b, T = toks.shape
        n = self.n
        v = torch.zeros((b, n), device=self.dev)
        z = torch.zeros((b, n), device=self.dev)
        ref = torch.zeros((b, n), device=self.dev)
        decays = torch.tensor([1.0 / t for t in TAUS], device=self.dev)
        read = torch.zeros((b, len(TAUS), n), device=self.dev)
        feats = []
        for t in range(T):
            cur = self.w_in[toks[:, t]]
            if self.arm == "B":
                cur = cur + (z @ self.p) @ self.q
            elif self.arm in ("C", "D", "3C", "3D"):
                rec = torch.zeros_like(z)
                for s in range(self.pre.shape[1]):
                    rec = rec + z[:, self.pre[:, s]] * self.wrec[:, s].unsqueeze(0)
                cur = cur + rec
            nv = self.alpha * v + cur - z * self.theta
            fresh = ((nv - self.theta) / self.theta > 0).float()
            nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
            ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
            v, z = nv, nz
            read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
            if t % stride == 0:  # 只保留采样时刻，否则特征张量会按序列长度线性膨胀
                if scope == "global":
                    feats.append(read.reshape(b, -1))
                elif self.arm in ("3D", "3h", "4h"):
                    # 三维子块池化：**显式平均矩阵**，正好 2x2x2 = 8 块
                    feats.append((read @ self.block_avg).reshape(b, -1))
                else:  # 一维局部区块读出
                    feats.append(read.reshape(b, len(TAUS) * 8, n // 8).mean(-1))
        return torch.stack(feats, 1)  # (b, T, F)


def decode(feats: torch.Tensor, states: torch.Tensor, n_train: int, dev: str) -> float:
    """线性解码器（岭回归闭式解），返回测试集准确率。"""
    X = feats.reshape(-1, feats.shape[-1])
    Y = states.reshape(-1)
    Xtr, Ytr, Xte, Yte = (
        X[: n_train * states.shape[1]],
        Y[: n_train * states.shape[1]],
        X[n_train * states.shape[1] :],
        Y[n_train * states.shape[1] :],
    )
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Yh = torch.nn.functional.one_hot(Ytr, K).float()
    A = Xtr.T @ Xtr + 1e-2 * torch.eye(Xtr.shape[1], device=dev)
    W = torch.linalg.solve(A, Xtr.T @ Yh)
    pred = (Xte @ W).argmax(-1)
    return float((pred == Yte).float().mean())


def window_features(toks: torch.Tensor, L: int = 8, stride: int = 8) -> torch.Tensor:
    """**正向对照**：最近 L 个 token 的 one-hot 拼接 → 线性解码。

    这才是「任务可解」的证明：它不依赖任何循环结构，只看观测窗口里有没有状态信息。
    原来的 GRU 臂是**随机初始化且冻结**的——那只是另一个蓄水池，证明不了任何事。
    """
    b, T = toks.shape
    pad = torch.zeros(b, L - 1, dtype=torch.long, device=toks.device)
    full = torch.cat([pad, toks], dim=1)  # (b, T+L-1)
    feats = []
    for t in range(0, T, stride):
        win = full[:, t : t + L]  # (b, L)
        feat = torch.zeros(b, L * V, device=toks.device)
        for j in range(L):
            feat[torch.arange(b, device=toks.device), j * V + win[:, j]] = 1.0
        feats.append(feat)
    return torch.stack(feats, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048, help="每个 SNN 臂的神经元数")
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=384)
    ap.add_argument("--test-seq", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--alpha", type=float, default=3.0, help="Dirichlet 浓度，越大单步证据越弱")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    pref = make_pref(0, args.alpha).to(dev)
    tr_toks, tr_st, _ = make_task(args.train_seq, args.seq, 1, dev, pref)
    te_toks, te_st, _ = make_task(args.test_seq, args.seq, 2, dev, pref)
    toks = torch.cat([tr_toks, te_toks])
    st = torch.cat([tr_st, te_st])
    # 两段拼接后按序列切分：前 train_seq 条训练、后 test_seq 条测试
    res: dict = {"n": args.n, "seq": args.seq, "train_seq": args.train_seq, "arms": {}}

    arms = [
        ("A", "平坦"),
        ("B", "低秩递归"),
        ("C", "环+地形"),
        ("D", "环+地形+局部读出"),
        ("3C", "3D网格+软地形"),
        ("3D", "3D网格+软地形+局部读出"),
        ("3h", "3D网格+**硬分区**+局部读出"),
        ("4h", "**4D网格+硬分区+局部读出**"),
    ]
    for arm, label in arms:
        pool = Res(arm, args.n, dev)
        feat = pool.run(toks, args.stride)
        acc = decode(feat, st[:, :: args.stride], args.train_seq, dev)
        parts = [pool.w_in]
        for extra in ("wrec", "p", "q"):
            t = getattr(pool, extra, None)
            if t is not None:
                parts.append(t)
        nparam = sum(int(x.numel()) for x in parts)
        res["arms"][arm] = {
            "label": label,
            "acc": acc,
            "params": int(nparam),
            "feat_dim": int(feat.shape[-1]),
        }
        print(
            f"  {arm} {label:14s} acc {acc:.3f}  feat {feat.shape[-1]:5d}  params {nparam}",
            flush=True,
        )

    # S-2：地形映射打乱（一维与三维各做一次）
    for tag, arm, label in (
        ("C_shuffled", "C", "环+地形（映射打乱）"),
        ("3h_shuffled", "3h", "3D硬分区（映射打乱）"),
        ("4h_shuffled", "4h", "4D硬分区（映射打乱）"),
    ):
        pool = Res(arm, args.n, dev, shuffle_map=True)
        acc = decode(pool.run(toks, args.stride), st[:, :: args.stride], args.train_seq, dev)
        res["arms"][tag] = {"label": label, "acc": acc}

    st_s = st[:, :: args.stride]
    for L in (8, 32):
        wf = window_features(toks, L=L, stride=args.stride)
        res["arms"][f"E{L}"] = {
            "label": f"窗口{L}-token 对照",
            "acc": decode(wf, st_s, args.train_seq, dev),
            "params": L * V * K,
        }
    res["random_baseline"] = 1.0 / K

    print(f"  C' 地形打乱       acc {res['arms']['C_shuffled']['acc']:.3f}")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    base = res["arms"]["A"]["acc"]
    best = max(res["arms"][a]["acc"] for a in ("B", "C", "D", "3C", "3D", "3h", "4h"))
    print("\n判：")
    best_ctl = max(res["arms"]["E8"]["acc"], res["arms"]["E32"]["acc"])
    floor = res["random_baseline"] + 0.2
    print(
        f"  S-0 最佳窗口对照 {best_ctl:.3f} ≥ {floor:.3f} ⇒ "
        f"{'过' if best_ctl >= floor else '不过（任务不可解，实验作废）'}"
    )
    print(
        f"  S-1 最佳结构臂 {best:.3f} − 平坦 {base:.3f} = {best - base:+.3f} ≥0.05 ⇒ "
        f"{'过' if best - base >= 0.05 else '不过'}"
    )
    d1 = res["arms"]["C"]["acc"] - res["arms"]["C_shuffled"]["acc"]
    print(f"  S-2a 一维 C − C'(打乱映射) = {d1:+.3f}")
    for tag in ("3h", "4h"):
        if f"{tag}_shuffled" not in res["arms"]:
            continue
        dd = res["arms"][tag]["acc"] - res["arms"][f"{tag}_shuffled"]["acc"]
        print(f"  S-2 {tag} − 打乱映射 = {dd:+.3f}")
    print(f"  参考 3D硬 − 一维D = {res['arms']['3h']['acc'] - res['arms']['D']['acc']:+.3f}")
    print(f"  参考 4D硬 − 一维D = {res['arms']['4h']['acc'] - res['arms']['D']['acc']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
