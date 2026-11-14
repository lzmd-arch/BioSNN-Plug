"""在**带传感器的合成任务**上训练小模型——"局部学习 + 独占输入"的最小闭环。

任务骨架：文本前缀恒定（`当前温度`），**答案完全由该池决定**，
所以 `zero` 臂只能到 0.109 附近（常数预测 + 噪声）。

| 池 | 驱动量 | 答案 |
| :--- | :--- | :--- |
| `temp` | 温度 t ∈ [0,9] → 标量 t/9，整段恒定 | t |
| `time` | 张力性常数驱动 1.0，**空步数** g ∈ [0,9] | g |
| `memory` | 16 维检索命中向量，**只在一次**（第 0 步）出现 | 原型号 |
| `persona` | 16 维状态向量，**每一步都给**（持续语境） | 簇号 |
| `presence` | 每步变化的标量（输入速率），答案取**最后一步** | 末值档 |

五个池的区别是**时间结构**，不是"换了组随机数"：恒定标量 / 积分器 / 一次注入 /
持续语境 / 时变跟踪。这是它们值得分开做的理由（预注册 §四）。

**训练与推理全程无 autograd**（与主网络同一条纪律）。
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

PREFIX = ["当前温度"]
TEXT_N = 156
POOL = 16
#: 逐池神经元数（`--pool-sizes`）；空时全部用 `--pool-size`
POOL_SIZES: dict[str, int] = {}
#: 逐池在传感器向量里的列偏移（由 `POOL_SIZES` 累加得到）
POOL_OFFSET: dict[str, int] = {}
POOLS = ("temp", "time", "memory", "persona", "presence")
_PROTO_CACHE: dict[int, np.ndarray] = {}


def _protos(n: int) -> np.ndarray:
    """10 个长度为 ``n`` 的单位原型。**按池大小缓存**——原先写死成 POOL=16，
    池 64 时形状对不上、直接崩（`memory`/`persona` 的池 64 臂就是这样丢的）。"""
    if n not in _PROTO_CACHE:
        r = np.random.default_rng(11).normal(0, 1, (10, n)).astype(np.float32)
        _PROTO_CACHE[n] = (r / np.linalg.norm(r, axis=1, keepdims=True)).astype(np.float32)
    return _PROTO_CACHE[n]


#: 一个回合里模型真的跑几步（= 前缀 token 数）。`presence` 的随机游走长度必须等于它，
#: 否则"答案取末值"的那个末值**从来没被喂进去**——这正是它单跑 0.24 的直接原因。
_N_STEPS = 2


def _pool_columns(name: str, pool_index: int) -> tuple[slice, slice]:
    """该池的 (输入列, 神经元列)。输入列在 `sensors` 向量里，神经元列在 `n_rec` 里。

    **逐池大小**：`POOL_SIZES` 给了就用它，否则用全局 `POOL`。
    """
    n = POOL_SIZES.get(name, POOL)
    lo = POOL_OFFSET.get(name, pool_index * POOL)
    return slice(lo, lo + n), slice(TEXT_N + lo, TEXT_N + lo + n)


def _make_episode(name: str, batch: int, rng: np.random.Generator, persona_cluster: int):
    """返回 ``(答案, 传感器, 空步数)``。

    传感器形状 ``(POOL, batch, T)``——第 ``t`` 个时间片；恒定池 ``T=1`` 时广播。
    ``gaps`` 是**整批相同**的空步数（None = 不插）。
    """
    if name == "temp":
        t = rng.integers(0, 10, size=batch)
        n = POOL_SIZES.get(name, POOL)
        s = np.zeros((n, batch, 1), dtype=np.float32)
        s[0, :, 0] = t / 9.0  # 只有第 0 个输入列承载温度（与历史结果同口径）
        return t, s, None
    if name == "time":
        g = int(rng.integers(0, 10))
        # 张力性驱动：常数 1.0 一直给 ⇒ 池是一条随空步推进的斜坡（间隔计时的标准模型）
        s = np.ones((POOL_SIZES.get(name, POOL), batch, 1), dtype=np.float32)
        return np.full(batch, g), s, g
    if name in ("memory", "persona"):
        if name == "memory":
            cl = rng.integers(0, 10, size=batch)
        else:
            cl = np.full(batch, persona_cluster)
        n = POOL_SIZES.get(name, POOL)
        v = _protos(n)[cl] + rng.normal(0, 0.3, (batch, n)).astype(np.float32)
        s = np.zeros((n, batch, 1), dtype=np.float32)
        s[:, :, 0] = v.T.astype(np.float32)
        return cl, s, None
    if name == "presence":
        # 输入速率：从随机起点做有界随机游走，答案 = **最后一步**所在的档。
        # 游走长度必须 = 回合步数，否则末值喂不进去（修之前写死 5 步，
        # 而回合只有 2 步 ⇒ 只喂了 walk[0]、walk[1]，答案 walk[4] 从未出现过）。
        walk = np.zeros((_N_STEPS, batch), dtype=np.int64)
        walk[0] = rng.integers(0, 10, size=batch)
        for k in range(1, _N_STEPS):
            walk[k] = np.clip(walk[k - 1] + rng.integers(-2, 3, size=batch), 0, 9)
        s = np.zeros((POOL_SIZES.get(name, POOL), batch, _N_STEPS), dtype=np.float32)
        # 只喂第 0 个输入列：16 列同值是 16 倍冗余，ramp 编码需要单一标量通道
        s[0] = (walk / 9.0).T
        return walk[-1], s, None
    raise ValueError(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("zero", "add", "gate"), required=True)
    ap.add_argument("--pools", default="temp", help="逗号分隔：temp,time,memory,persona,presence")
    ap.add_argument(
        "--subset", type=Path, default=Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2/subset_32k.json")
    )
    ap.add_argument("--counts", type=Path, default=Path("sweep_results/lm_10m/counts_32000.npy"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-sensor", type=float, default=0.05)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument(
        "--pool-scales",
        default="",
        help="逐池覆盖 init_scale，如 temp:32,time:2（空 = 用 --init-scale）",
    )
    ap.add_argument("--desensitize", type=float, default=0.05, help="受体脱敏 λ；0 = 关")
    ap.add_argument("--clip", type=float, default=10.0, help="‖sensor_proj‖ 上限；0 = 不限")
    ap.add_argument(
        "--encoder",
        choices=("random", "design"),
        default="random",
        help="random = 随机投影（旧口径）；design = 解析构造（ramp / ortho）",
    )
    ap.add_argument(
        "--encoder-levels",
        type=int,
        default=0,
        help="温度计码的阈值级数；0 = 等于池神经元数（满分辨率）",
    )
    ap.add_argument("--pool-size", type=int, default=16, help="每个池的默认神经元数")
    ap.add_argument(
        "--pool-sizes", default="", help="逐池覆盖，如 temp:32,time:64（池与池可以不同）"
    )
    ap.add_argument("--text-neurons", type=int, default=156, help="文本神经元数")
    ap.add_argument("--tau-text", type=float, default=20.0, help="文本神经元的膜时间常数")
    ap.add_argument(
        "--tau-pool", type=float, default=0.0, help="池神经元的膜时间常数；0 = 与文本同值（同质）"
    )
    ap.add_argument("--route", type=int, default=0, help="激素路由的掩码条数；0 = 关闭")
    ap.add_argument("--readout-taus", default="2,128", help="逗号分隔的读出时间常数")
    ap.add_argument("--debug", action="store_true", help="打印每池的读出诊断")
    ap.add_argument("--seed", type=int, default=0, help="池初值的随机种子（配对比较用）")
    ap.add_argument("--inhibition", type=float, default=0.0, help="侧抑制增益；0 = 关")
    ap.add_argument("--hormone", choices=("scalar", "tonic"), default="tonic")
    ap.add_argument("--receptors", choices=("global", "perpool"), default="perpool")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/sensor_pools"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    global POOL, TEXT_N, POOL_SIZES, POOL_OFFSET
    POOL = int(args.pool_size)
    TEXT_N = int(args.text_neurons)
    pools = [p for p in args.pools.split(",") if p]
    POOL_SIZES = {}
    for item in filter(None, args.pool_sizes.split(",")):
        _k, _v = item.split(":")
        POOL_SIZES[_k] = int(_v)
    for p in pools:
        POOL_SIZES.setdefault(p, POOL)
    POOL_OFFSET, _acc = {}, 0
    for p in pools:
        POOL_OFFSET[p] = _acc
        _acc += POOL_SIZES[p]
    _sensor_dim = _acc
    for p in pools:
        if p not in POOLS:
            raise SystemExit(f"未知池 {p}")
    n_rec = TEXT_N + _sensor_dim
    tok = SubsetTokenizer(args.subset.parent)
    digit_ids = [tok.encode(str(d)).ids[0] for d in range(10)]
    prefix_ids = tok.encode(PREFIX[0]).ids
    n_steps = len(prefix_ids)  # 前缀 + 数字位
    global _N_STEPS
    _N_STEPS = n_steps
    counts = torch.from_numpy(np.load(args.counts)).float()
    paths = build_huffman_paths(counts)
    model = TokenEProp(
        32000,
        n_rec,
        recurrence="none",
        # !! 这里曾经写死 (2.0, 128.0)，于是 `--readout-taus` 被静默忽略：
        # B-3（读出维度 2→4）与基因搜索里的 `readout` 基因全部是同一条臂。
        readout_taus=tuple(float(x) for x in args.readout_taus.split(",")),
        tree=paths,
        tree_signal_scale=3.0,
        tree_packed=True,
        tree_init_weight=False,
        embed_rank=256,
        sensor_dim=_sensor_dim,
        lr_in=1e-3,
        lr_out=0.5,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    # **功能分离由模型强制**：`set_sensor_pools` 把 sensor_proj 锁成块对角、每次更新后重施。
    # 不再依赖训练脚本里的掩码——掩码忘了写就静默退回混合，今天已经吃过一次。
    model.set_sensor_pools({name: _pool_columns(name, k) for k, name in enumerate(pools)})
    with torch.no_grad():
        # 文本→神经元的路径可能是 proj（低秩）、也可能是 w_in（稠密查找），两者都要挡住
        if getattr(model, "proj", None) is not None:
            model.proj[:, TEXT_N:] = 0.0
        if getattr(model, "w_in", None) is not None:
            model.w_in[:, TEXT_N:] = 0.0
        # **每个池的初值只由它自己决定**：生成器按池名固定、分母用固定的参考规模。
        # 不这样做的话，单池与五池共存拿到的是两次不同的随机抽样、且尺度跟着 n_rec 变——
        # 于是 S-4 的"单跑 vs 共存"里混进了两个与"共存"无关的差异。
        scales = {}
        for item in filter(None, args.pool_scales.split(",")):
            k, v = item.split(":")
            scales[k] = float(v)
        ref = math.sqrt(TEXT_N + POOL)
        model.sensor_proj = torch.zeros_like(model.sensor_proj)
        if args.encoder == "design":
            # 1 维标量池 → 温度计码；向量池 → 正交投影。尺度仍由 --init-scale 标定。
            for name in pools:
                kind = "ramp" if name in ("temp", "time", "presence") else "ortho"
                model.design_sensor_encoder(
                    name,
                    kind=kind,
                    scale=scales.get(name, args.init_scale),
                    seed=args.seed,
                    steps=n_steps,
                    levels=args.encoder_levels,
                )
            pools_done = True
        else:
            pools_done = False
        for name in [] if pools_done else pools:
            sc = scales.get(name, args.init_scale)
            i_sl, _c_sl = model.sensor_pools[name]
            gb = torch.Generator().manual_seed(1000 + 17 * POOLS.index(name) + 7919 * args.seed)
            model.sensor_proj[i_sl] = torch.randn(
                (i_sl.stop - i_sl.start, model.n_rec), generator=gb
            ) * (sc / ref)
        model.sensor_proj *= model._pool_mask
    if args.debug:
        for _n in pools:
            _i, _c = model.sensor_pools[_n]
            print(
                f"    [dbg] 编码 {_n}: rows {_i.start}:{_i.stop} cols {_c.start}:{_c.stop} "
                f"块范数 {float(model.sensor_proj[_i].norm()):.3f} "
                f"第0行非零 {int((model.sensor_proj[_i][0].abs() > 0).sum())}",
                flush=True,
            )
    model.sensor_mode = "gate" if args.arm == "gate" else "add"
    model.lr_sensor = args.lr_sensor
    if args.tau_text != 20.0 or args.tau_pool > 0:
        # **τ 异质**：各池自己的膜时间常数（B-1）
        pt = args.tau_pool if args.tau_pool > 0 else args.tau_text
        model.alpha = math.exp(-1.0 / args.tau_text)
        model.set_pool_taus(dict.fromkeys(pools, pt), default_tau=args.tau_text)
    if args.route > 0:
        model.set_sensor_routes(k=args.route, seed=args.seed)
    model.receptor_desensitize = args.desensitize
    model.sensor_clip = args.clip
    model.set_cross_inhibition(args.inhibition if args.inhibition > 0 else None)
    if args.hormone == "scalar":
        # 对照臂：紧张性时间常数 = 1 步 ⇒ 退化成"本步值"，与旧的单一标量逐位一致
        model.hormone_tonic_tau = 1.0
        model.hormone_phasic_share = 0.0
    if args.receptors == "perpool" and len(pools) > 1:
        rho = torch.tensor([1.0, 0.5, 0.5])
        model.sensor_receptor_matrix = rho.repeat(len(pools), 1) * torch.linspace(
            1.0, 0.6, len(pools)
        ).unsqueeze(1)

    n_taus = len(model.readout_decays) if model.readout_decays is not None else 1
    probe_idx = {}
    rate_idx = {}
    for name in pools:
        lo = TEXT_N + POOL_OFFSET[name]
        idx = []
        for t in range(n_taus):
            idx.extend(range(t * n_rec + lo, t * n_rec + lo + POOL_SIZES[name]))
        probe_idx[name] = idx
        rate_idx[name] = list(range(lo, lo + POOL_SIZES[name]))

    rng = np.random.default_rng(0)
    dev = model.device
    # **每个池自己的缓冲**：多池共存时混用一个缓冲会让原型被"别的池里同名的类"污染
    # （实测 memory 0.931 → 0.459，全都不是模型的错）。
    buf_x: dict[str, list[torch.Tensor]] = {p: [] for p in pools}
    buf_y: dict[str, list[torch.Tensor]] = {p: [] for p in pools}
    t0 = time.time()

    def run_episode(name, batch, rng_, persona_cluster, update, digit=None, slot=None):
        ans, sensors, gaps = _make_episode(name, batch, rng_, persona_cluster)
        if digit is not None:
            # **标签必须跟着激励走**：原先只覆盖 ans 而不覆盖 gaps，
            # 于是"答案"和"实际插了几个空步"对不上——节奏池因此被跑成随机。
            ans = np.full(batch, int(gaps))
        slot = pools.index(name) if slot is None else slot
        tr = model.new_trace(batch)
        seq = [*prefix_ids, digit_ids[0]]
        for i in range(n_steps):
            tok_t = torch.full((batch,), seq[i], dtype=torch.long, device=dev)
            if i == n_steps - 1:
                tgt = torch.tensor([digit_ids[int(a)] for a in ans], device=dev)
            else:
                tgt = torch.full((batch,), seq[i + 1], dtype=torch.long, device=dev)
            if args.arm != "zero":
                blk = sensors[:, :, min(i, sensors.shape[2] - 1)].T
                _off = POOL_OFFSET[name]
                full = np.zeros((batch, _sensor_dim), dtype=np.float32)
                full[:, _off : _off + POOL_SIZES[name]] = blk
                model.sensors = torch.from_numpy(full).to(dev)
            model.step(tok_t, tgt, tr, update=update, all_valid=True)
            # 空步插在第一个 token 之后：张力性驱动的池在这段里推进，
            # 而文本神经元已在静息，所以差别只来自池。空步不产生梯度（target=-100）。
            if gaps is not None and i == 0:
                for _ in range(gaps):
                    model.step(
                        tok_t,
                        torch.full_like(tgt, -100),
                        tr,
                        update=False,
                        all_valid=True,
                        blank=True,
                    )
        if args.arm != "zero":
            model.sensors = None
        return ans, tr

    last = {"hit": {}, "rate": {}, "norm": model.sensor_norm}
    for step in range(1, args.steps + 1):
        name = pools[(step - 1) % len(pools)]
        # `time` 池每批一个 g，遍历 0..9 —— 与评测口径一致
        digit = (step // len(pools)) % 10 if name == "time" else None
        # **人格池的簇必须每回合随机**：原先用 8 个簇心循环，位置可被训练步数预测
        # ⇒ `zero` 臂也能到 1.000（S-3 的泄露守卫正是为此写的）。
        pcl = int(rng.integers(0, 10)) if name == "persona" else 0
        ans, tr = run_episode(name, args.batch, rng, pcl, True, digit)
        feats = model._features(tr)
        bx, by = buf_x[name], buf_y[name]
        bx.append(feats.detach())
        by.append(torch.tensor(ans))
        if len(bx) > 200:
            bx.pop(0)
            by.pop(0)

        if step % args.eval_every == 0:
            hit = {}
            hit_cos = {}
            hit_lin = {}
            hit_knn = {}
            hit_maha = {}
            rates = {}
            cv = {}
            pool_means: dict[str, torch.Tensor] = {}
            pool_act: dict[str, list[float]] = {p: [] for p in pools}
            with torch.no_grad():
                for p in pools:
                    if not buf_x[p]:
                        continue
                    x_all = torch.stack(buf_x[p][-100:]).reshape(-1, model.readout_width)
                    y_all = torch.stack(buf_y[p][-100:]).reshape(-1)
                    # **探针必须保留幅度**：余弦相似度对整体缩放不变，而标量驱动的池
                    # （温度、节奏）把答案编码在**发放率**上——归一化会把唯一可分的那个
                    # 维度抹掉。这里用"标准化 + 欧氏最近原型"，仍然无参数、无训练。
                    z_pool = x_all[:, probe_idx[p]]
                    # **两个探针都算**：余弦只看方向、欧氏看方向+幅度。哪个都不占优，
                    # 所以 S-1 要求**两个探针同时过**——把"仪器决定的结论"挡在门外。
                    protos = []
                    for c in range(10):
                        sel = y_all == c
                        protos.append(
                            z_pool[sel].mean(0)
                            if sel.any().item()
                            else torch.zeros(z_pool.shape[1])
                        )
                    proto = torch.stack(protos)
                    # **学习型读出**：岭回归闭式解。原型探针对"维度加权"敏感
                    # （复制一维就等于给那一维加倍权重），所以必须另有一个自己学权重的
                    # 读出，才能把"信息"和"度量"分开。
                    cut = int(0.7 * len(buf_x[p]))
                    x_tr = torch.stack(buf_x[p][:cut]).reshape(-1, model.readout_width)
                    x_tr = x_tr[:, probe_idx[p]]  # 与原型探针看同一批维度
                    y_tr = torch.stack(buf_y[p][:cut]).reshape(-1)
                    y_oh = torch.nn.functional.one_hot(y_tr, 10).float()
                    # **标准化再岭回归**：两个时间尺度的 EMA 范数差几个数量级，
                    # 不标准化时 λ 在不同维度上的作用完全不同，解会病态
                    # （`time` 池曾因此掉到 0.200 = 常数预测）。
                    mu_ = x_tr.mean(0)
                    sd_ = x_tr.std(0) + 1e-6
                    xs_tr = (x_tr - mu_) / sd_
                    a_ = xs_tr.T @ xs_tr + 1.0 * torch.eye(x_tr.shape[1])
                    w_lin = torch.linalg.solve(a_, xs_tr.T @ y_oh)
                    # **kNN**：非参数、非线性（Voronoi 型），且**对"把一维复制成两维"不敏感**
                    # ——欧氏距离只在原空间里量一次，标准化后更不会。
                    # 用它当主判据，线性读出只作参照。
                    knn_x = xs_tr
                    knn_y = y_tr
                    # **马氏最近类心**：非参数、非线性（按类内协方差白化），
                    # 且**对"复制一维"不敏感**——白化后复制的那一维落在零空间里，
                    # 距离不变。这比线性读出和 kNN 都更适合当主判据：
                    # kNN 在维数升高时会因距离集中而崩（实测 512 维只有 0.31）。
                    _cm = torch.stack([xs_tr[y_tr == c].mean(0) for c in range(10)])
                    _res = xs_tr - _cm[y_tr]
                    _cov = _res.T @ _res / max(len(y_tr) - 1, 1)
                    _tr = torch.diagonal(_cov).mean()
                    _cov = 0.9 * _cov + 0.1 * _tr * torch.eye(_cov.shape[0])
                    _evals, _evecs = torch.linalg.eigh(_cov)
                    _whiten = _evecs @ torch.diag(1.0 / _evals.clamp(min=1e-6).sqrt()) @ _evecs.T
                    _cm_w = _cm @ _whiten
                    proto_cos = proto / (proto.norm(dim=1, keepdim=True) + 1e-9)
                    acc = acc_cos = acc_lin = acc_knn = acc_maha = n = 0
                    r_acc = 0.0
                    per_class: dict[int, list[float]] = {}
                    mean_feat = torch.zeros(len(probe_idx[p]))
                    # **评测用独立的、按池固定的采样流**：不能吃训练用的 `rng`。
                    # 否则同一配置在不同池组合下会拿到不同的评测样本，
                    # "多池更差"就可能是采样差而不是模型差（实测温度池 0.16 vs 0.10）。
                    rng_eval = np.random.default_rng(9000 + POOLS.index(p))
                    for j in range(args.eval_batches):
                        dig = j % 10 if p == "time" else None
                        a2, tr2 = run_episode(
                            p, args.batch, rng_eval, int(rng_eval.integers(0, 10)), False, dig
                        )
                        if args.arm != "zero":
                            zz = tr2.z[:, rate_idx[p]].mean(1)  # 逐样本
                            r_acc += float(zz.mean())
                            for bb in range(len(a2)):
                                per_class.setdefault(int(a2[bb]), []).append(float(zz[bb]))
                        q = model._features(tr2)[:, probe_idx[p]]
                        mean_feat += q.mean(0)
                        # 逐样本的池活动：这是"两个池是否在重复同一件事"的**冗余**口径
                        pool_act[p].extend(q.mean(1).tolist())
                        d2 = ((q.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(-1)
                        acc += int((d2.argmin(1).cpu().numpy() == a2).sum())
                        q_cos = q / (q.norm(dim=1, keepdim=True) + 1e-9)
                        acc_cos += int(((q_cos @ proto_cos.T).argmax(1).cpu().numpy() == a2).sum())
                        q_std = (q - mu_) / sd_
                        acc_lin += int(((q_std @ w_lin).argmax(1).cpu().numpy() == a2).sum())
                        _q_w = q_std @ _whiten
                        acc_maha += int(
                            (torch.cdist(_q_w, _cm_w).argmin(1).cpu().numpy() == a2).sum()
                        )
                        _d = torch.cdist(q_std, knn_x)
                        _idx = _d.topk(5, largest=False).indices
                        _votes = knn_y[_idx]
                        _pred = torch.mode(_votes, dim=1).values
                        acc_knn += int((_pred.cpu().numpy() == a2).sum())
                        n += args.batch
                    hit[p] = acc / max(n, 1)
                    hit_cos[p] = acc_cos / max(n, 1)
                    hit_lin[p] = acc_lin / max(n, 1)
                    hit_knn[p] = acc_knn / max(n, 1)
                    hit_maha[p] = acc_maha / max(n, 1)
                    if args.debug:
                        tr_acc = float(((xs_tr @ w_lin).argmax(1) == y_tr).float().mean())
                        print(
                            f"    [dbg] {p}: n_train={len(y_tr)} dim={xs_tr.shape[1]} "
                            f"train_acc={tr_acc:.3f} lin={hit_lin[p]:.3f} "
                            f"proto={hit[p]:.3f}",
                            flush=True,
                        )
                    rates[p] = r_acc / max(args.eval_batches, 1)
                    # **饱和检测**：各答案档之间发放率的变异系数。
                    # 标量平均发放率看不出饱和——饱和的特征是"档与档之间没有差别"。
                    cls = [sum(v) / len(v) for v in per_class.values() if v]
                    if args.debug:
                        print(
                            f"    [dbg] {p}: 逐类发放率 {[round(float(x), 2) for x in cls]}",
                            flush=True,
                        )
                    if len(cls) > 1:
                        mu = sum(cls) / len(cls)
                        var = sum((c - mu) ** 2 for c in cls) / len(cls)
                        cv[p] = (var**0.5) / mu if mu > 0 else 0.0
                    else:
                        cv[p] = 0.0
                    pool_means[p] = mean_feat / max(args.eval_batches, 1)
            cos = {}
            corr = {}
            names = list(pool_means)
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    if pool_means[names[a]].numel() != pool_means[names[b]].numel():
                        # 池与池大小不同时没有逐维对照，余弦无定义
                        cos[f"{names[a]}-{names[b]}"] = float("nan")
                        continue
                    u = pool_means[names[a]] / (pool_means[names[a]].norm() + 1e-9)
                    v = pool_means[names[b]] / (pool_means[names[b]].norm() + 1e-9)
                    cos[f"{names[a]}-{names[b]}"] = float(u @ v)
                    x = torch.tensor(pool_act[names[a]])
                    y = torch.tensor(pool_act[names[b]])
                    x = x - x.mean()
                    y = y - y.mean()
                    den = float(x.norm() * y.norm())
                    corr[f"{names[a]}-{names[b]}"] = float((x @ y) / den) if den > 0 else 0.0
            last = {
                "hit": hit,
                "hit_cos": hit_cos,
                "rate": rates,
                "rate_cv": cv,
                "norm": model.sensor_norm,
                "cos": cos,
                "corr": corr,
            }
            detail = "  ".join(
                f"{p} e{hit[p]:.3f}/m{hit_maha[p]:.3f}/l{hit_lin[p]:.3f}" for p in pools
            )
            cv_s = "  ".join(f"{k} {v:.2f}" for k, v in cv.items())
            rate_s = (
                "  " + "  ".join(f"{p} {rates[p]:.2f}" for p in pools) if args.arm != "zero" else ""
            )
            print(
                f"  [{args.arm}|{'+'.join(pools)}] {step} 步  top-1 {detail}"
                f"  发放率{rate_s}  CV {cv_s}"
                f"  ‖w‖ {model.sensor_norm:.2f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    out = {
        "arm": args.arm,
        "pools": pools,
        "steps": args.steps,
        "batch": args.batch,
        "init_scale": args.init_scale,
        "inhibition": args.inhibition,
        "hormone": args.hormone,
        "receptors": args.receptors,
        "tag": args.tag,
        **last,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    key = f"{args.arm}_{'+'.join(pools)}" + (f"_{args.tag}" if args.tag else "")
    (args.out / f"{key}.json").write_text(
        json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
