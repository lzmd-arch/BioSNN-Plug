"""TokenEProp：**逐时刻**的局部学习语言模型（无反向传播）。

三种递推来源，除此之外代码完全同一条路径（判据见 `research/sel/prereg-lm-eprop.md` 与
`research/sel/prereg-lm-s0-ssm.md`）：

- ``recurrence="random"``：冻结的随机稀疏 ``w_rec``（M0/M0b 的默认臂）；
- ``recurrence="none"``：没有递推，纯前馈 LIF；
- ``recurrence="ssm"``：**对角 SSM**（S4D-real 思路）。极点 ``a_s = exp(-1/tau_s)`` 冻结，
  ``h_t = a ⊙ h_{t-1} + E[x_t]``，电流 ``C·h_t + D[x_t]``，``E / C / D`` 用同一套局部规则学习。

**没有任何 autograd**：所有更新都在 ``torch.no_grad()`` 下手写。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median as statistics_median

import torch

from research.sel.lm.hierarchical import HuffmanPaths, TreeReadout, path_probabilities

__all__ = ["LMTrace", "TokenEProp"]


@dataclass
class LMTrace:
    """跨块结转的全部状态。只在会话边界调用 :meth:`reset`。"""

    v: torch.Tensor  # (B, N) 膜电位
    z: torch.Tensor  # (B, N) 上一步脉冲
    refractory: torch.Tensor  # (B, N) 不应期计数
    eps_in: torch.Tensor  # (B, V) 输入痕迹（one-hot 的衰减计数）
    eps_rec: torch.Tensor  # (B, N) 循环痕迹
    readout: torch.Tensor  # (B, N) 读出 EMA
    readouts: torch.Tensor | None = None  # (B, K, N) 多时间尺度 EMA（仅 readout_taus）
    h: torch.Tensor | None = None  # (B, S) SSM 状态（仅 recurrence="ssm"）
    eps_e: torch.Tensor | None = None  # (B, V, S) SSM 输入投影的资格痕迹
    eps_in_groups: torch.Tensor | None = None  # (K, B, V) 异质 τ 的分组输入痕迹
    a: torch.Tensor | None = None  # (B, N) ALIF 适应变量（仅 beta>0）

    @classmethod
    def zeros(
        cls,
        batch: int,
        n_rec: int,
        n_vocab: int,
        *,
        n_state: int = 0,
        n_taus: int = 0,
        n_groups: int = 1,
        use_adapt: bool = False,
        device=None,
        dtype=torch.float32,
    ):
        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=device, dtype=dtype)

        return cls(
            v=zeros(batch, n_rec),
            z=zeros(batch, n_rec),
            refractory=zeros(batch, n_rec),
            eps_in=zeros(batch, n_vocab),
            eps_rec=zeros(batch, n_rec),
            readout=zeros(batch, n_rec),
            readouts=zeros(batch, n_taus, n_rec) if n_taus else None,
            h=zeros(batch, n_state) if n_state else None,
            eps_e=zeros(batch, n_vocab, n_state) if n_state else None,
            eps_in_groups=zeros(n_groups, batch, n_vocab) if n_groups > 1 else None,
            a=zeros(batch, n_rec) if use_adapt else None,
        )

    def reset(self, rows: torch.Tensor) -> None:
        """把指定行清零（会话边界）。``rows`` 是 bool 掩码或下标张量。"""
        names = ["v", "z", "refractory", "eps_in", "eps_rec", "readout"]
        if self.readouts is not None:
            names += ["readouts"]
        if self.h is not None:
            names += ["h", "eps_e"]
        if self.eps_in_groups is not None:
            names += ["eps_in_groups"]
        if self.a is not None:
            names += ["a"]
        for name in names:
            getattr(self, name)[rows] = 0.0


class TokenEProp:
    """LIF + 逐时刻 e-prop + EMA 读出；递推可以是随机稀疏、无、或对角 SSM。"""

    def __init__(
        self,
        n_vocab: int,
        n_rec: int = 1024,
        *,
        fan_in: int = 32,
        tau: float = 20.0,
        threshold: float = 0.62,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        tau_read: float = 8.0,
        input_std: float = 0.12,
        lr_in: float = 2e-4,
        lr_rec: float = 1e-4,
        lr_out: float = 0.5,
        recurrence: str = "random",
        readout_width: int | None = None,
        readout_taus: tuple[float, ...] | None = None,
        candidates: torch.Tensor | None = None,
        tree: HuffmanPaths | None = None,
        tree_signal_scale: float = 1.0,
        tree_packed: bool = False,
        tree_init_weight: bool = True,
        sensor_dim: int = 0,
        pad_vocab: bool = True,
        bf16_weights: bool = False,
        weight_dtype: str = "fp32",
        mirror_dtype: str = "",
        norm_groups: int = 1,
        trace_decay: float = 0.0,
        bf16_input: bool = False,
        readout_rank: int | None = None,
        readout_basis: torch.Tensor | None = None,
        embed_rank: int | None = None,
        embed_proj: torch.Tensor | None = None,
        embed_init: torch.Tensor | None = None,
        tau_groups: int = 1,
        beta: float = 0.0,
        rho: float = 0.99,
        amp: bool = False,
        n_state: int = 32,
        tau_state: tuple[float, float] = (1.0, 128.0),
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if recurrence not in ("random", "none", "ssm"):
            raise ValueError(f"recurrence 只能是 random / none / ssm，收到 {recurrence!r}")
        self.n_vocab = int(n_vocab)
        #: S1：词表维补到 8 的倍数。cuBLAS 在 V=3001 这类"非 tile 倍数"的形状上掉出张量核
        #: （实测 bf16 16.3 → 33.6 TFLOPS）。补出来的列不参与 softmax，**输出逐位不变**。
        self.n_pad = (self.n_vocab + 7) // 8 * 8 if pad_vocab else self.n_vocab
        #: S2：读出权重存 bf16（cuBLAS 内部 fp32 累加）。显存减半、走张量核。
        #: `weight_dtype` 优先；`bf16_weights=True` 是它的旧写法。
        name = "bf16" if bf16_weights else weight_dtype
        self.w_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]
        self.weight_dtype_name = name
        #: S2'：**主权重保持 fp32**，只把一份镜像降到低精度供两次矩阵乘读；更新写回 fp32 主权重，
        #: 然后刷新镜像。低精度存储（S2）会把舍入直接打进累加器，实测 top-1 −11.1pp；镜像不会。
        self.mirror_dtype = (
            {"bf16": torch.bfloat16, "fp16": torch.float16}[mirror_dtype] if mirror_dtype else None
        )
        self.mirror_dtype_name = mirror_dtype
        #: S5：读出的 NLMS 归一化从**一个全局标量**改成**每组一个**。零算力成本。
        #: 脑区局部抑制的计算对应物：一个组活动大不该把别的组的步长一起压小。
        #: 1 = 历史行为（逐位不变）。
        self.norm_groups = max(1, int(norm_groups))
        #: S2b：输入权重 `w_in` 也存 bf16。它在 1B 下是 `V·N` = 200M MAC/token，
        #: 是低秩读出之后**最大的一项**，而且此刻完全没进 bf16。
        self.w_dtype_in = torch.bfloat16 if bf16_input else torch.float32
        self.n_rec = int(n_rec)
        self.fan_in = int(fan_in)
        self.threshold = float(threshold)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)
        self.alpha = float(math.exp(-1.0 / tau))
        #: S7：资格痕迹自己的衰减 β，**与膜常数解耦**。e-prop 的推导里两者相同（都等于 α），
        #: 但 Traces Propagation（arXiv:2509.13053，NCE 2026）把 β 当作每层独立的超参在调
        #: （SHD 递归臂 0.85、前馈 0.96、DVS-CIFAR10 0.18）——也就是说"因果窗口"本身是个自由度。
        #: 我们的 α=e^(−1/20) ⇒ 窗口只有 ~20 步，而序列是 192 步。``0`` = 跟 `alpha` 相同（历史行为，逐位不变）。
        self.trace_decay = float(trace_decay) if trace_decay > 0 else self.alpha
        #: G2 异质膜 τ：神经元分成 ``tau_groups`` 组，各组一个 τ（log 均匀 [5, 500]，**冻结**）。
        #: ``tau_groups=1`` 时走原路径、逐位一致。
        #: G1 ALIF：阈值 ``θ + beta·a``，``a = rho·a + z``（用**上一步**的脉冲）。
        #: ``beta=0`` 时与 LIF 逐位一致。
        self.beta = float(beta)
        self.rho = float(rho)
        #: T3：把**大矩阵乘**降到 bf16（状态与权重仍是 fp32，所以不影响阈值判据）。
        self.amp = bool(amp)
        self.n_groups = max(1, int(tau_groups))
        if self.n_groups > 1:
            # 注意：self.device 在这一段之后才设，这里直接用传入的 device
            target = torch.device(device)
            taus = torch.logspace(math.log10(5.0), math.log10(500.0), self.n_groups).to(target)
            # **分组必须与 tau_vec 同一口径**：`tau_vec` 是连续块（repeat_interleave），
            # 而原先的 `group_ids` 用的是交错下标（index % n_groups），两者对不上 ⇒
            # 输入痕迹按错误的 τ 衰减。2026-11-10 修：统一切成连续块。
            bounds = [round(g * n_rec / self.n_groups) for g in range(self.n_groups + 1)]
            self.group_slices = [(bounds[g], bounds[g + 1]) for g in range(self.n_groups)]
            counts = [hi - lo for lo, hi in self.group_slices]
            self.tau_vec = torch.repeat_interleave(taus, torch.tensor(counts, device=target))
            self.alpha_vec = torch.exp(-1.0 / self.tau_vec)
            self.group_ids = [torch.arange(lo, hi, device=target) for lo, hi in self.group_slices]
            self.group_alphas = torch.exp(-1.0 / taus)  # 与 `alpha` 同口径：e^(−1/τ)
        else:
            self.tau_vec = None
            self.alpha_vec = None
            self.group_ids = None
            self.group_slices = None
            self.group_alphas = None
        self.decay_read = float(math.exp(-1.0 / tau_read))
        self.lr_in = float(lr_in)
        self.lr_rec = float(lr_rec)
        self.lr_out = float(lr_out)
        self.recurrence = recurrence
        self.n_state = int(n_state) if recurrence == "ssm" else 0
        self.device = torch.device(device)
        #: A1 候选剪枝：``(n_vocab, C)`` 的 long 张量，每行是"上一个字 → 最可能的 C 个下一个字"。
        #: ``None`` = 全量 softmax（历史行为，逐位不变）。
        self.candidates = None if candidates is None else candidates.to(self.device)

        gen = generator or torch.Generator()
        self.w_in = (torch.randn(n_vocab, n_rec, generator=gen).to(self.device) * input_std).to(
            self.w_dtype_in
        )
        # 读出宽度：None 或等于 n_rec 时读全部；否则初始化时固定抽一个子集（R3 的 (b) 臂）
        if readout_taus:
            if readout_width is not None:
                raise ValueError(
                    "readout_taus 与 readout_width 互斥（R4 与 R3 的变量不能同时开）。"
                )
            self.readout_decays = torch.exp(
                -1.0 / torch.tensor(readout_taus, device=self.device, dtype=torch.float32)
            )
            # y_t = d·y_{t-1} + z_t 的稳态方差是 var(z)/(1-d²)（越慢越大），
            # 乘 sqrt(1-d²) 后各抽头方差相同；不归一化的话慢抽头大 6–8 倍，会盖过快抽头
            self.readout_scales = torch.sqrt(1.0 - self.readout_decays**2).to(self.device)
            self.readout_index = None
            width = len(readout_taus) * n_rec
        elif readout_width is None or int(readout_width) >= n_rec:
            self.readout_decays = None
            self.readout_scales = None
            self.readout_index: torch.Tensor | None = None
            width = n_rec
        else:
            self.readout_decays = None
            self.readout_scales = None
            width = int(readout_width)
            self.readout_index = torch.randperm(n_rec, generator=gen)[:width].to(self.device)
        #: 层次 softmax（H-1）：读出换成「树上每个内部节点一个 M 维向量」。
        #: 参数量 (V-1)·M 与稠密版 M·V 同量级，省的是**算力**不是显存。
        self.tree: TreeReadout | None = None
        if tree is not None:
            if candidates is not None:
                raise ValueError("树读出与候选剪枝互斥（两条都改输出层的归一化范围）")
            if readout_rank is not None:
                raise ValueError("树读出与低秩读出互斥（低秩改的正是读出矩阵的形状）")
            if norm_groups > 1:
                raise ValueError("树读出的归一化口径未定义，norm_groups 只能为 1")
            self.tree = TreeReadout(
                tree.to(self.device),
                width,
                lr_out=lr_out,
                signal_scale=tree_signal_scale,
                packed=tree_packed,
                init_weight=tree_init_weight,
                generator=gen,
                device=self.device,
                dtype=self.w_dtype,
            )
            self.w_out = None
            self.w_out_mirror = None
        else:
            self.w_out = (
                torch.randn(width, self.n_pad, generator=gen).to(self.device) / math.sqrt(width)
            ).to(self.w_dtype)
            self.w_out_mirror: torch.Tensor | None = (
                self.w_out.to(self.mirror_dtype) if self.mirror_dtype is not None else None
            )
        bounds = [round(g * width / self.norm_groups) for g in range(self.norm_groups + 1)]
        self.norm_bounds = [(bounds[g], bounds[g + 1]) for g in range(self.norm_groups)]
        self.norm_sizes = torch.tensor([hi - lo for lo, hi in self.norm_bounds], device=self.device)
        #: A3 低秩读出：``w_out ≈ w_a @ w_b``。``w_b`` 固定（随机基），只学 ``w_a``。
        #: ``readout_rank=None``（或 ≥ width）= 稠密读出，逐位不变。
        #: A2 低秩输入：``w_in ≈ emb(V×d) @ proj(d×N)``。给了 ``embed_rank`` 时 ``w_in`` 不用。
        #: ``embed_proj`` 给定即冻结（不学），否则也学。
        self.embed_rank: int | None = None
        self.emb: torch.Tensor | None = None
        self.proj: torch.Tensor | None = None
        self.proj_frozen = False
        if embed_rank is not None and 0 < int(embed_rank) <= n_rec:
            self.embed_rank = int(embed_rank)
            if embed_init is not None:
                # 词嵌入**几何**迁移：从千问的 token_embd 投影而来（见 scripts/transfer_embedding.py）
                if tuple(embed_init.shape) != (n_vocab, self.embed_rank):
                    raise ValueError(
                        f"embed_init 形状 {tuple(embed_init.shape)} 与 "
                        f"{(n_vocab, self.embed_rank)} 不符"
                    )
                self.emb = embed_init.to(device=self.device, dtype=torch.float32)
            else:
                self.emb = (
                    torch.randn(n_vocab, self.embed_rank, generator=gen).to(self.device) * input_std
                )
            if embed_proj is not None:
                self.proj = embed_proj.to(self.device)
                self.proj_frozen = True
            else:
                self.proj = torch.randn(self.embed_rank, n_rec, generator=gen).to(
                    self.device
                ) / math.sqrt(self.embed_rank)
            del self.w_in
            self.w_in = None
        self.readout_rank: int | None = None
        self.w_a: torch.Tensor | None = None
        self.w_b: torch.Tensor | None = None
        if readout_rank is not None and 0 < int(readout_rank) < width:
            self.readout_rank = int(readout_rank)
            self.w_a = (
                torch.randn(width, self.readout_rank, generator=gen).to(self.device)
                / math.sqrt(width)
            ).to(self.w_dtype)
            if readout_basis is not None:
                self.w_b = readout_basis.to(self.device)
                if self.w_b.shape[1] < self.n_pad:  # 补零列，保持与 w_out 同形
                    self.w_b = torch.nn.functional.pad(
                        self.w_b, (0, self.n_pad - self.w_b.shape[1])
                    )
                self.w_b = self.w_b.to(self.w_dtype)
            else:
                self.w_b = (
                    torch.randn(self.readout_rank, self.n_pad, generator=gen).to(self.device)
                    / math.sqrt(self.readout_rank)
                ).to(self.w_dtype)
            del self.w_out
            self.w_out = None

        #: **传感器通道**（Semprea-4.5B 的独有输入）：环境/多模态向量 → 神经元电流。
        #: `sensor_dim=0` 时完全不建，逐位不变；建了也**零初始化** ⇒ 与没有它逐位相同。
        #: 这是"信息不对称"的落点：这些数据不出设备，云端永远拿不到。
        self.sensor_proj: torch.Tensor | None = None
        #: **传感器通道的学习率**。这条通路的更新规则是本类**唯一**的非 e-prop 更新：
        #: 激素式三因子 `Δproj[i,j] = η · h · c[i] · post[j]`，h 是树给出的调制信号。
        #: 之前 `sensor_proj` 只被读、从不被写，所以"传感器学不出来"的根因在这里。
        self.lr_sensor = 0.05
        #: **多种激素**：三个本地可得的调制信号各一路，每个传感器池有自己的受体权重。
        #: 生物学上多巴胺/乙酰胆碱/去甲肾上腺素对同一事件给出不同的可塑性调制，
        #: 单一标量无法表达"该学什么"与"该学多快"的区别。
        #: [0] 显著度 = 树的预测误差信号；[1] 唤醒 = 发放率偏离基线；[2] 新奇 = 特征变化。
        self.sensor_receptors = torch.tensor([1.0, 0.5, 0.5])
        self.rate_baseline = 0.1
        #: **受体脱敏**：受体被持续激活后自身变钝。`ρ ← ρ·(1 − λ·|激活|)`。
        #: 生理上这是受体下调；工程上它是唯一能防止调制线性放大到爆炸的机制——
        #: DFA 的范数从 0 炸到 155，就是没有这一条。
        self.receptor_desensitize = 0.05
        #: **突触缩放**（Turrigiano）：每个神经元按自己的发放率与目标之比，乘性地缩放
        #: 自己的全部输入突触。与"固定衰减"的区别是**它有设定点、是负反馈、双向**——
        #: 实测固定脱敏下范数照样从 14.9 涨到 37.4，因为单向衰减管不到量级。
        self.synaptic_scaling = 0.0  # 0 = 关闭（保持与之前逐位相同）
        #: 传感器权重的 L2 上限；0 = 不限。**这个数需要靠"范数–性能曲线"来定**，
        #: 不是拍脑袋——今天三次"有界"尝试（脱敏/缩放/clip）都因为缺这条曲线而失败。
        import os as _os

        #: **默认 0（关）**。曾经定过 10，依据是"clip 5/20/60 都是 0.306 ⇒ [5,60] 免费"。
        #: 那次测量是在范数自然落在 10–32 的配置上做的；**按发放率标定后的温度池
        #: 自然范数是 ~128**，此时 clip=10 实测把 top-1 从 0.306 削到 0.209（−9.7pp）。
        #: ⇒ "免费"只在"自然范数本来就在区间内"时成立，不是普适结论。默认关，要就显式给。
        self.sensor_clip = float(_os.environ.get("AXO_SENSOR_CLIP", "0") or 0.0)
        self.sensors: torch.Tensor | None = None
        #: **池结构**：`{池名: (输入列 slice, 神经元列 slice)}`。设了它之后，
        #: `sensor_proj` 被强制成**块对角**——每个池只能写自己的神经元，
        #: 别的模态写不进来。这是"功能分离"从训练脚本的约定升级成模型的结构约束：
        #: 今天五个混杂因素里有两个（共享神经元、均值外积）就是靠脚本里的掩码，
        #: 而掩码一旦忘了写就静默退回混合。
        self.sensor_pools: dict[str, tuple[slice, slice]] = {}
        self._pool_mask: torch.Tensor | None = None
        #: **区间侧抑制**：池 r 的电流减去 γ·(其它池上一步的平均发放)。`None` = 关闭（逐位不变）。
        #: 没有它，四个池会退化成彼此冗余的拷贝（设计文档 §二）。
        self.cross_inhib: torch.Tensor | None = None
        self.inhib_gain = 0.5
        #: **激素状态**（多路、分时程）。长度 d 与 `sensor_receptors` 的最后一维一致。
        #: `hormone_tonic_tau` = 紧张性（慢）成分的时间常数；相位性成分 = 当前值 − 紧张性。
        self.hormone_state: torch.Tensor | None = None
        self.hormone_tonic_tau = 20.0
        self.hormone_phasic_share = 0.5
        #: 每个池自己的受体向量（`(n_pools, d)`）；`None` 时退回全局单向量（逐位不变）。
        self.sensor_receptor_matrix: torch.Tensor | None = None
        #: 传感器注入方式。``add`` = 加性电流（现行，冻结权重下的行为）；
        #: ``gate`` = **门控**：乘一个 (1+调制)，符合"神经调制改变增益而不是加偏置"。
        #: 默认 ``add`` ⇒ 逐位不变。
        self.sensor_mode = "add"
        self.sensor_dim = int(sensor_dim)
        if self.sensor_dim > 0:
            self.sensor_proj = torch.zeros(self.sensor_dim, n_rec, device=self.device)

        post = torch.arange(n_rec, device=self.device).unsqueeze(1)
        candidates = torch.randint(0, n_rec - 1, (n_rec, fan_in), generator=gen).to(self.device)
        self.pre_index = candidates + (candidates >= post).long()
        self.w_rec = torch.randn(n_rec, fan_in, generator=gen).to(self.device) / math.sqrt(fan_in)

        if recurrence == "ssm":
            # 极点：时间常数在 [tau_min, tau_max] 上对数等距，冻结
            low, high = math.log(tau_state[0]), math.log(tau_state[1])
            taus = torch.exp(torch.linspace(low, high, self.n_state, device=self.device))
            self.poles = torch.exp(-1.0 / taus)  # (S,)
            self.state_taus = taus
            self.E = torch.randn(n_vocab, self.n_state, generator=gen).to(self.device) / math.sqrt(
                self.n_state
            )
            self.C = torch.randn(n_rec, self.n_state, generator=gen).to(self.device) / math.sqrt(
                self.n_state
            )
            self.D = torch.randn(n_vocab, n_rec, generator=gen).to(self.device) * input_std

    # -- 状态 -----------------------------------------------------------------

    def new_trace(self, batch: int) -> LMTrace:
        n_taus = 0 if self.readout_decays is None else int(self.readout_decays.numel())
        return LMTrace.zeros(
            batch,
            self.n_rec,
            self.n_vocab,
            n_state=self.n_state,
            n_taus=n_taus,
            n_groups=self.n_groups,
            use_adapt=self.beta > 0.0,
            device=self.device,
        )

    # -- 传感器池 / 激素 -------------------------------------------------------

    def set_sensor_pools(self, pools: dict[str, tuple[slice, slice]]) -> None:
        """登记池结构：``{池名: (输入列 slice, 神经元列 slice)}``。

        登记之后 ``sensor_proj`` 被强制成**块对角**，并且每次更新后重新施加。
        这把"功能分离"从训练脚本里的约定升级成**模型的结构约束**——
        今天五个混杂因素里有两个（共享神经元、均值外积）原本只靠脚本里的掩码，
        掩码一旦忘了写就静默退回混合。

        同时按池结构建好**区间侧抑制矩阵**（``cross_inhib``）。
        """
        if self.sensor_proj is None:
            raise ValueError("sensor_dim=0 时没有池可登记")
        self.sensor_pools = {name: (sl[0], sl[1]) for name, sl in pools.items()}
        mask = torch.zeros(self.sensor_proj.shape, dtype=torch.bool, device=self.device)
        for i_sl, c_sl in self.sensor_pools.values():
            mask[i_sl, c_sl] = True
        self._pool_mask = mask
        self.sensor_proj = self.sensor_proj * mask
        self._build_cross_inhibition()

    def _build_cross_inhibition(self) -> None:
        """池 r 的目标列只接收**其它池**的平均发放；文本神经元不参与。

        矩阵乘是 ``z @ cross_inhib``，``[i, j]`` 是"源神经元 i 对目标 j 的权重"。
        """
        if not self.sensor_pools or self.n_rec <= 0:
            self.cross_inhib = None
            return
        n = self.n_rec
        onehot = torch.zeros((len(self.sensor_pools), n), device=self.device)
        for k, (_i, c_sl) in enumerate(self.sensor_pools.values()):
            onehot[k, c_sl] = 1.0
        same = onehot.T @ onehot  # (n, n)：同一池为 1
        in_pool = (onehot.sum(0) > 0).to(onehot.dtype)  # (n,)
        diff = (1.0 - same) * in_pool.unsqueeze(0) * in_pool
        count = diff.sum(0, keepdim=True).clamp(min=1.0)
        self.cross_inhib = (diff / count).to(self.device)

    def design_sensor_encoder(
        self,
        name: str,
        kind: str = "ramp",
        lo: float = 0.05,
        hi: float = 0.95,
        scale: float = 1.0,
        seed: int = 0,
        steps: int = 2,
    ) -> None:
        """把某个池的输入投影**解析构造**出来，而不是随机抽。

        理由（2026-11-14 实测）：同一任务、同一尺度、同一训练步数，只换随机种子，
        1 维标量池的 top-1 在 0.097–0.223 之间跳；而训练曲线从 250 步到 6000 步是平的。
        ⇒ **是抽签决定成败，不是学习**。既然如此，就不该让编码去抽。

        ``kind="ramp"``：1 维标量 → **温度计码**。神经元 j 的输入权重取
        ``θ(1−α)/s*_j``，使它在标量值 ``s*_j`` 处刚好越过阈值，``s*`` 在 ``[lo, hi]``
        上等间距铺开。这是 LIF 稳态 ``v = I/(1−α)`` 的解析反解，不是拟合。

        ``kind="ortho"``：d 维向量 → **正交投影**（QR）。保证各列条件数一致，
        不会出现"随机抽到近共线 ⇒ 整池塌在一个方向上"。
        """
        if name not in self.sensor_pools:
            raise KeyError(f"未登记的池：{name}")
        i_sl, c_sl = self.sensor_pools[name]
        n_in = i_sl.stop - i_sl.start
        n_col = c_sl.stop - c_sl.start
        # **必须用暂态增益，不是稳态增益**：回合只有几步，膜电位到不了 ``I/(1−α)``。
        # 从静止起 n 步、恒定电流下 ``v_n = I·(1−αⁿ)/(1−α)``。
        # 用稳态值会把阈值放到 10 倍远处——实测整池静默（发放率 0.00）。
        a = float(self.alpha)
        gain = (1.0 - a ** max(int(steps), 1)) / (1.0 - a)
        thr = float(self.threshold) / gain
        block = torch.zeros((n_in, self.n_rec), device=self.device)
        if kind == "ramp":
            xs = torch.linspace(lo, hi, n_col, device=self.device)
            block[0, c_sl] = thr / xs
        elif kind == "ortho":
            # 单位正交行 × (thr/0.5)：单位范数输入下 |I| 最大到 2×阈值 ⇒ 既不塌也不饱和
            g = torch.Generator().manual_seed(1000 + 17 * seed)
            q, _ = torch.linalg.qr(torch.randn((n_col, n_col), generator=g))
            block[i_sl, c_sl] = q.to(self.device) * (thr / 0.5)
        else:
            raise ValueError(f"未知的编码类型：{kind}")
        self.sensor_proj[i_sl] = block * float(scale)

    def set_cross_inhibition(self, gain: float | None) -> None:
        """``gain=None`` 关闭侧抑制（``cross_inhib`` 置空，逐位不变）。"""
        if gain is None:
            self.cross_inhib = None
        else:
            self.inhib_gain = float(gain)
            self._build_cross_inhibition()

    @property
    def sensor_norm(self) -> float:
        return 0.0 if self.sensor_proj is None else float(self.sensor_proj.norm())

    def _hormone_raw(self, tree_signal: torch.Tensor, trace: LMTrace) -> torch.Tensor:
        """三路**本地可得**的量：显著度（树的预测误差）/ 唤醒（发放率偏离基线）/ 新奇（读出变化）。"""
        prev = getattr(self, "_prev_readout", None)
        if prev is None:
            prev = trace.readout.mean()
        raw = torch.stack(
            [
                torch.tensor(float(tree_signal.mean()), dtype=torch.float32),
                (trace.z.mean() - self.rate_baseline).abs(),
                (trace.readout.mean() - prev).abs(),
            ]
        )
        self._prev_readout = trace.readout.mean().detach().clone()
        return raw.detach().to(torch.float32)

    def _hormone_step(self, raw: torch.Tensor) -> torch.Tensor:
        """分时程：紧张性（慢 EMA，设定点在高频行为上稳定）+ 相位性（相对慢基线的瞬态）。

        ``hormone_tonic_tau`` 极大且 ``hormone_phasic_share=0`` 时退回"单一步进值"——
        这是 S-7 的对照臂，它与旧行为逐位等价。
        """
        if self.hormone_state is None or self.hormone_state.numel() != raw.numel():
            self.hormone_state = raw.clone()
            return raw
        rate = 1.0 / max(float(self.hormone_tonic_tau), 1.0)
        tonic = self.hormone_state + (raw - self.hormone_state) * rate
        self.hormone_state = tonic
        share = float(self.hormone_phasic_share)
        return tonic + share * (raw - tonic)

    def _pool_gain(self, h: torch.Tensor) -> torch.Tensor:
        """逐**输入列**的调制标量。分区受体时每个池用自己的 ρ_r；否则用全局向量。"""
        if self.sensor_proj is None:
            return h
        n_in = self.sensor_proj.shape[0]
        base = float(h @ self.sensor_receptors.to(h.dtype))
        if self.sensor_receptor_matrix is None or not self.sensor_pools:
            return torch.full((n_in,), base, dtype=torch.float32)
        rows = self.sensor_receptor_matrix.shape[0]
        gain = torch.full((n_in,), base, dtype=torch.float32)
        for k, (i_sl, _c_sl) in enumerate(self.sensor_pools.values()):
            rho = self.sensor_receptor_matrix[k % rows].to(h.dtype)
            gain[i_sl] = float(h @ rho)
        return gain

    def _sensor_post_update(self, trace: LMTrace, h: torch.Tensor) -> None:
        """更新后的四道整备，顺序有意义：脱敏 → 缩放 → 池结构 → 范数上限。

        池结构放在缩放之后：缩放只看本池的发放率，不该跨池泄漏；
        范数上限放最后，保证"任何一步之后都不超过 clip"。
        """
        lam = float(self.receptor_desensitize)
        if lam > 0.0:
            decay = (1.0 - lam * h.abs()).clamp(0.05, 1.0)
            self.sensor_receptors = (self.sensor_receptors * decay).clamp(0.05, 1.0)
            if self.sensor_receptor_matrix is not None:
                self.sensor_receptor_matrix = (
                    self.sensor_receptor_matrix * decay.unsqueeze(0)
                ).clamp(0.05, 1.0)
        alpha = float(self.synaptic_scaling)
        if alpha > 0.0 and self.sensor_pools:
            for _i_sl, c_sl in self.sensor_pools.values():
                rate = float(trace.z[:, c_sl].mean())
                ratio = (self.rate_baseline / max(rate, 1e-6)) ** alpha
                self.sensor_proj[:, c_sl] *= ratio
        if self._pool_mask is not None:
            self.sensor_proj *= self._pool_mask
        if self.sensor_clip > 0.0:
            norm = float(self.sensor_proj.norm())
            if norm > self.sensor_clip:
                self.sensor_proj *= self.sensor_clip / norm

    # -- 前向与更新 ------------------------------------------------------------

    def _candidate_forward(self, features, tokens, targets):
        """A1：只在候选列上算 logits。target 单独一列（并把它在候选里的副本屏蔽掉）。

        返回 ``(cols, logits, loss)``；``logits`` 是 ``(B, C+1)``，最后一列是 target。
        """
        cols = self.candidates[tokens]  # (B, C)
        w_c = self.w_out[:, cols]  # (M, B, C)
        logits_c = torch.einsum("bm,mbc->bc", features, w_c)
        tgt = targets.clamp(min=0)
        w_t = self.w_out[:, tgt]  # (M, B)
        logit_t = torch.einsum("bm,mb->b", features, w_t)
        collide = cols == tgt.unsqueeze(1)
        logits_c = logits_c.masked_fill(collide, float("-inf"))
        logits = torch.cat([logits_c, logit_t.unsqueeze(1)], dim=1)  # (B, C+1)
        index = torch.full(
            (logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device
        )
        loss = torch.nn.functional.cross_entropy(logits, index, reduction="none")
        return cols, logits, loss

    def _candidate_gradient(self, logits, cols, targets, active, batch):
        """候选版的梯度与学习信号。返回 ``(signal_sub, grad_cols, grad_target)``。"""
        probability = torch.softmax(logits, dim=-1)  # (B, C+1)
        grad_cols = probability[:, :-1] * active  # (B, C)
        grad_target = (probability[:, -1:] - 1.0) * active  # (B, 1)
        scale = 1.0 / max(batch, 1)
        w_c = self.w_out[:, cols]  # (M, B, C)
        signal_c = torch.einsum("bc,mbc->bm", grad_cols, w_c)
        w_t = self.w_out[:, targets.clamp(min=0)]  # (M, B)
        signal_t = torch.einsum("b,mb->bm", grad_target.squeeze(1), w_t)
        return (signal_c + signal_t) * scale, grad_cols * scale, grad_target * scale

    def _candidate_update(self, features, cols, grad_cols, grad_target, targets, norm) -> None:
        """只更新候选列与 target 列（A1 的全部算力收益在这里）。"""
        delta = torch.einsum("bm,bc->mbc", features, grad_cols)  # (M, B, C)
        tgt = targets.clamp(min=0)
        for row in range(features.shape[0]):
            block = self.w_out[:, cols[row]]
            block.add_(delta[:, row, :], alpha=-self.lr_out / norm)
            self.w_out[:, cols[row]] = block
            self.w_out[:, tgt[row]] -= self.lr_out * features[row] * grad_target[row, 0] / norm

    @property
    def readout_width(self) -> int:
        """读出特征宽度 M（与是否低秩无关）。"""
        if self.tree is not None:
            return int(self.tree.n_features)
        return int(self.w_out.shape[0]) if self.w_b is None else int(self.w_a.shape[0])

    def _mm(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """大矩阵乘：右操作数是什么精度就用什么精度（fp32 直乘；低精度走张量核，升回 fp32）。

        ``amp`` 是老开关，强制 bf16。**精度跟着权重走**，所以低精度存储与低精度镜像共用这一条路径。
        """
        dtype = torch.bfloat16 if self.amp else right.dtype
        if dtype == torch.float32:
            return left @ right
        return (left.to(dtype) @ right.to(dtype)).float()

    def _mm_in(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """输入权重的更新矩阵乘；精度跟随 ``w_dtype_in``（交给 ``add_`` 落回目标精度）。"""
        if self.w_dtype_in == torch.float32:
            return left @ right
        return left.to(torch.bfloat16) @ right.to(torch.bfloat16)

    @property
    def _w_read(self) -> torch.Tensor:
        """矩阵乘该读的读出权重：有镜像读镜像（低精度），否则读主权重。"""
        return self.w_out if self.w_out_mirror is None else self.w_out_mirror

    @property
    def _w_read_t(self) -> torch.Tensor:
        return self._w_read.T

    def _token_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """整词表上的概率 ``(B, V)``。树读出下走路径概率，稠密下就是 softmax。

        top-1 / top-k 只在概率上取 argmax / topk，所以**评测口径不变**——
        这是 H-1 能与稠密基线直接比的前提。
        """
        if self.tree is not None:
            return path_probabilities(features, self.tree.weight, self.tree.paths, self.tree.bias)
        return torch.softmax(self._logits(features)[:, : self.n_vocab], dim=-1)

    def _logits(self, features: torch.Tensor) -> torch.Tensor:
        """``features -> logits``。稠密版直乘；低秩版先投影到 r 再展开。"""
        if self.w_b is None:
            return self._mm(features, self._w_read)
        return self._mm(self._mm(features, self.w_a), self.w_b)

    def _features(self, trace: LMTrace) -> torch.Tensor:
        if self.readout_decays is None:
            return self._readout_features(trace.readout)
        scaled = trace.readouts * self.readout_scales.view(1, -1, 1)
        return scaled.reshape(scaled.shape[0], -1)

    def _readout_features(self, readout: torch.Tensor) -> torch.Tensor:
        """读出看到的特征：默认全部神经元；``readout_width`` 给定时只看固定子集。"""
        return readout if self.readout_index is None else readout[:, self.readout_index]

    def _recurrent_current(self, z_previous: torch.Tensor) -> torch.Tensor:
        if self.recurrence != "random":
            return torch.zeros_like(z_previous)
        current = torch.zeros_like(z_previous)
        for slot in range(self.fan_in):
            current += z_previous[:, self.pre_index[:, slot]] * self.w_rec[:, slot]
        return current

    @torch.no_grad()
    def step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        trace: LMTrace,
        *,
        update: bool = True,
        learn_hidden: bool = True,
        learn_recurrent: bool = True,
        all_valid: bool = False,
        learn_mask: torch.Tensor | None = None,
        blank: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """跑一个时刻。返回 (逐样本 CE, 本步发放数)。

        ``blank=True`` 是**空步**：没有文本电流，膜电位照样按 α 衰减、复发电流与
        传感器电流照样进。``tokens`` 的内容被忽略（但形状要对）。

        ``all_valid=True`` 时走**全有效快路径**：跳过三个 ``torch.where(active...)`` 与
        ``* active`` 的乘法（实测每步省 4–5 个内核）。它与慢路径**逐位等价**——``active`` 全 1 时
        ``where(active, x, y) == x``、``x*active == x``。
        """
        batch = tokens.shape[0]
        dtype = trace.v.dtype
        valid = targets != -100
        active = valid.to(dtype).unsqueeze(1)  # (B, 1)

        z_previous = trace.z
        h_new = None
        if self.recurrence == "ssm":
            # SSM 臂里 D 就是那条"直接通路"，不再另设 w_in（否则两者是同一个东西）
            h_new = self.poles * trace.h + self.E[tokens]
            current = self.D[tokens] + h_new @ self.C.T
            trace.h = torch.where(active > 0, h_new, trace.h)
        else:
            if blank:
                # **空步**：没有文本电流，只有复发电流与传感器电流。
                # 时间/节奏池就用它——"输入间隔"在离散时间里就是"插入多少个空步"。
                if self.recurrence == "ssm":
                    raise ValueError("blank 步不支持 ssm 递推（它没有独立的 w_in 通路）")
                direct = torch.zeros((batch, self.n_rec), device=trace.v.device, dtype=dtype)
            else:
                direct = self.w_in[tokens] if self.emb is None else self.emb[tokens] @ self.proj
                direct = direct.float()  # bf16 存储时升回 fp32，膜电位与阈值判据保持 fp32
            current = direct + self._recurrent_current(z_previous)
        if self.sensor_proj is not None and self.sensors is not None:
            # 传感器向量加进**电流**而不是特征：它作用于膜电位，与输入通路正交，
            # 因此不影响 e-prop 的资格痕迹（那些乘的是 w_in 的输入）。
            modulation = self.sensors.to(current.dtype) @ self.sensor_proj
            if getattr(self, "sensor_mode", "add") == "gate":
                current = current * (1.0 + modulation)
            else:
                current = current + modulation
        if self.cross_inhib is not None:
            # **区间侧抑制**：池 r 的电流减去 γ·(其它池上一步的平均发放)。
            # 用上一步的 z，与复发电流同口径，不引入当前步内的先后依赖。
            # 没有它，多池会退化成彼此冗余的拷贝（设计文档 §二）。
            other = z_previous @ self.cross_inhib
            current = current - self.inhib_gain * other

        alpha_used = self.alpha if self.n_groups == 1 else self.alpha_vec
        theta = self.threshold
        if trace.a is not None:
            # G1：适应变量用**上一步**的脉冲更新，阈值抬到 θ + β·a；复位仍用基阈值。
            adapt = self.rho * trace.a + z_previous
            trace.a = adapt if all_valid else torch.where(active > 0, adapt, trace.a)
            theta = self.threshold + self.beta * trace.a
        new_v = alpha_used * trace.v + current - z_previous * self.threshold
        scaled = (new_v - theta) / theta
        refractory = trace.refractory > 0
        fresh = (scaled > 0).to(dtype)  # 硬阈值；本类不做反传，不需要伪导数
        new_z = torch.where(refractory, torch.zeros_like(fresh), fresh)
        new_r = torch.clamp(
            trace.refractory + self.n_refractory * new_z - 1.0,
            min=0.0,
            max=float(self.n_refractory),
        )
        if all_valid:
            trace.v, trace.z, trace.refractory = new_v, new_z, new_r
        else:
            trace.v = torch.where(active > 0, new_v, trace.v)
            trace.z = torch.where(active > 0, new_z, torch.zeros_like(new_z))
            trace.refractory = torch.where(active > 0, new_r, trace.refractory)
        trace.readout = self.decay_read * trace.readout + trace.z

        # 资格痕迹：用**本步之前**的脉冲与本步的输入累积（与 EventDrivenEProp 同口径）
        # **推理不更新资格痕迹**：`update=False` 时这些量只服务于学习，
        # 而 eps_in 是 (B, 32000)，每步一次整表乘 + 一次 scatter_add_，
        # 剖析里占一步的 10%（scatter_add_ 单次 91 µs）。
        # 跳过它们不改变 v/z/refractory/readout —— 那四个才是推理用得上的状态。
        #: 推理时是否跳过学习痕迹（可 A/B 的开关；默认 True 与上一条改动一致）
        keep = update and getattr(self, "skip_learning_traces", False) is False
        if keep:
            trace.eps_rec = self.trace_decay * trace.eps_rec + z_previous
        if trace.eps_in_groups is None:
            if keep:
                trace.eps_in = self.trace_decay * trace.eps_in
                if not blank:
                    trace.eps_in.scatter_add_(
                        1, tokens.reshape(-1, 1), active.to(trace.eps_in.dtype)
                    )
        else:
            groups = trace.eps_in_groups
            groups.mul_(self.group_alphas.view(-1, 1, 1))
            index = tokens.view(1, -1, 1).expand(groups.shape[0], -1, 1)
            groups.scatter_add_(2, index, active.to(groups.dtype).view(1, -1, 1).expand_as(index))
        if self.recurrence == "ssm":
            trace.eps_e = trace.eps_e * self.poles.view(1, 1, -1)
            trace.eps_e.scatter_add_(
                1,
                tokens.view(-1, 1, 1).expand(-1, 1, self.n_state),
                active.view(-1, 1, 1).expand(-1, 1, self.n_state),
            )

        if self.readout_decays is not None:
            trace.readouts = trace.readouts * self.readout_decays.view(
                1, -1, 1
            ) + trace.z.unsqueeze(1)
        features = self._features(trace)
        cols = None
        tree_signal = None
        if self.tree is not None:
            # 目标为 -100 的行不进损失 / 信号 / 更新（与稠密版 active 同口径）
            tree_mask = targets != -100
            if learn_mask is not None:
                tree_mask = tree_mask & learn_mask
            loss, tree_signal = self.tree.step(
                features, targets.clamp(min=0), update=update, mask=tree_mask
            )
        elif self.candidates is not None:
            cols, logits, loss = self._candidate_forward(features, tokens, targets)
        else:
            logits = self._logits(features)[:, : self.n_vocab]  # (B, V)；补出来的列丢掉
            loss = torch.nn.functional.cross_entropy(
                logits, targets, ignore_index=-100, reduction="none"
            )
        if (
            update
            and self.sensor_proj is not None
            and self.sensors is not None
            and tree_signal is not None
        ):
            # **纯局部**：前 = 本批传感器均值，后 = 本步平均发放，调制 = 激素向量。
            # 没有 .backward()、没有跨层误差路径。
            raw = self._hormone_raw(tree_signal, trace)
            h = self._hormone_step(raw)
            gain = self._pool_gain(h)  # (sensor_dim,)：逐输入列的调制标量
            # **必须去均值**：不去均值时外积是一条近乎恒定的方向，
            # 学到的东西只会整体放大，不会把不同输入分开（历史记录：训练不改善、0.106）。
            # 去均值之后它是**协方差**——逐样本的输入差异才落进权重。
            c = self.sensors.to(torch.float32)
            if c.dim() == 1:  # 一条向量广播给整批（推理/测试里常见）
                c = c.unsqueeze(0).expand(trace.z.shape[0], -1)
            c = c - c.mean(0, keepdim=True)
            z = trace.z.to(torch.float32)
            z = z - z.mean(0, keepdim=True)
            self.sensor_proj += (
                self.lr_sensor
                * ((gain.to(torch.float32).unsqueeze(1) * c.T) @ z)
                / max(c.shape[0], 1)
            )
            self._sensor_post_update(trace, h)
        fired = trace.z.sum()

        if update:
            grad_cols = grad_target = None
            if self.tree is not None:
                # 树的更新已在 `tree.step` 里就地做完，这里只把信号接着往下传
                signal_sub = tree_signal
            elif cols is not None:
                signal_sub, grad_cols, grad_target = self._candidate_gradient(
                    logits, cols, targets, active, batch
                )
            else:
                probability = torch.softmax(logits, dim=-1)
                # 梯度只在前 n_vocab 列非零；补出来的列留 0，好让 addmm 与原矩阵同形
                gradient = torch.zeros(
                    probability.shape[0],
                    self.n_pad,
                    device=probability.device,
                    dtype=probability.dtype,
                )
                gradient[:, : self.n_vocab] = probability
                gradient.scatter_(1, targets.clamp(min=0).reshape(-1, 1), -1.0, reduce="add")
                if not all_valid:
                    gradient = gradient * active
                gradient = gradient / max(batch, 1)
                # **学习掩码与状态掩码是两件事**：`valid` 决定"这一步要不要更新膜电位"，
                # `learn_mask` 决定"这一步要不要产生梯度"。探针里填充位必须更新状态
                # （否则膜电位被冻结、网络不发放），但不该产生梯度（否则读出被"预测填充"
                # 淹没：259 个平凡位置对 1 个真正要学的位置）。
                if learn_mask is not None:
                    gradient = gradient * learn_mask.to(gradient.dtype).unsqueeze(1)
                if self.w_b is None:
                    signal_sub = self._mm(gradient, self._w_read_t)  # (B, M)
                else:
                    signal_sub = self._mm(self._mm(gradient, self.w_b.T), self.w_a.T)  # (B, M)
            if self.readout_decays is not None:
                # 多时间尺度：每个抽头的信号属于同一批神经元，按抽头求和
                signal = signal_sub.view(signal_sub.shape[0], -1, self.n_rec).sum(dim=1)
            elif self.readout_index is None:
                signal = signal_sub
            else:
                signal = torch.zeros(
                    signal_sub.shape[0],
                    self.n_rec,
                    device=signal_sub.device,
                    dtype=signal_sub.dtype,
                )
                signal.scatter_(
                    1, self.readout_index.unsqueeze(0).expand(signal_sub.shape[0], -1), signal_sub
                )
            psi = (self.dampening_factor / self.threshold) * torch.clamp(
                1.0 - scaled.abs(), min=0.0
            )
            # F1（2026-11-10）：ψ 的掩码必须用**上一步结束时**的计数器（即本步进入时的
            # `refractory` 局部量），不能用 `trace.refractory`——后者已含本步的脉冲，
            # 会把 ψ 在脉冲时刻置零。口径出处：`research/eprop/traces.py::refractory_mask`
            # 的 docstring（"判据是上一步之后的计数器，不是本步之后的"）。
            psi = torch.where(refractory, torch.zeros_like(psi), psi)
            post = signal * psi if all_valid else signal * psi * active

            if learn_hidden and self.recurrence != "ssm":
                if self.emb is None and trace.eps_in_groups is not None:
                    # G2：每个 τ 组有自己的输入痕迹，只更新本组那几列。
                    # 各组列宽之和 = N ⇒ **总成本与同质路径相同**（V·B·N），不是 K 倍。
                    for g, (lo, hi) in enumerate(self.group_slices):
                        self.w_in[:, lo:hi].add_(
                            self._mm_in(trace.eps_in_groups[g].T, post[:, lo:hi]),
                            alpha=-self.lr_in,
                        )
                elif self.emb is None:
                    # 一次 (V,B)@(B,N) 的 GEMM；原先的 einsum 会被降成 bmm（收缩维长度为 1），
                    # FLOP 相同但访存差。数学完全一样。
                    self.w_in.add_(self._mm_in(trace.eps_in.T, post), alpha=-self.lr_in)
                else:
                    # A2：dL/dE = eps_inᵀ (post·Pᵀ) = (V,d)；dL/dP = (eps_in·E)ᵀ post = (d,N)。
                    # **结合顺序是关键**：先右乘再左乘，避免物化 (V,N)。
                    signal_emb = post @ self.proj.T  # (B, d)
                    self.emb.add_(
                        torch.einsum("bv,bd->vd", trace.eps_in, signal_emb), alpha=-self.lr_in
                    )
                    if not self.proj_frozen:
                        hidden = trace.eps_in @ self.emb  # (B, d)
                        self.proj.add_(torch.einsum("bd,bn->dn", hidden, post), alpha=-self.lr_in)
            if learn_recurrent and self.recurrence == "random":
                gathered = trace.eps_rec[:, self.pre_index]  # (B, N, fan_in)
                self.w_rec.add_(torch.einsum("bnf,bn->nf", gathered, post), alpha=-self.lr_rec)
            if self.recurrence == "ssm":
                # E：极点决定各自的衰减；信号是该极点对输出的贡献
                pole_signal = post @ self.C  # (B, S)
                self.E.add_(torch.einsum("bs,bvs->vs", pole_signal, trace.eps_e), alpha=-self.lr_in)
                # C：状态 → 电流，本步状态是 h_new
                self.C.add_(post.T @ h_new, alpha=-self.lr_rec)
                # D：跳跃连接，与 w_in 同构
                self.D.index_add_(0, tokens, post, alpha=-self.lr_in)
            # 不取 float()：每一步都 `.item()` 会强制 GPU→CPU 同步，实测是主要开销之一
            norm = features.pow(2).sum(dim=1).mean().clamp(min=1e-3)
            # 分母：全局版就是 `norm`；分组版把 **全部**归一化搬进 features 的逐列缩放，
            # 这里必须是 1——否则会多出一个 1/norm 的因子，等于顺手把学习率乘了 G 倍。
            denom = norm
            if self.norm_groups > 1 and cols is None:
                # S5：每组自己的归一化。更新对 features 是线性的，所以只要把每组乘上 1/norm_g，
                # 后面完全不用改——FLOPs 与内核次数都不变，只多一次逐元素乘。
                per_group = torch.stack(
                    [
                        features[:, lo:hi].pow(2).sum(dim=1).mean().clamp(min=1e-3)
                        for lo, hi in self.norm_bounds
                    ]
                )
                # **必须再除一个 G**：每一步的总修正量是 Σ_g (f_g·Δw_g) = Σ_g g/G = g，
                # 才与全局归一化"一步把 g 投影掉"同量级。少了这个 1/G，等于每步过冲 G 倍
                # ——实测 CE 从 8.0 一路涨到 18.1（发散），不是"分组没用"，是**实现放大了步长**。
                scale = 1.0 / (self.norm_groups * per_group)
                features = features * torch.repeat_interleave(scale, self.norm_sizes)
                denom = 1.0
            if self.tree is not None:
                pass
            elif cols is not None:
                self._candidate_update(features, cols, grad_cols, grad_target, targets, norm)
            elif self.w_b is None:
                # 原地 addmm：省掉 (M,V) 的 fp32 临时与一次"读+写"整表（实测 8%）
                torch.addmm(
                    self.w_out,
                    features.to(self.w_dtype).T,
                    (gradient / denom).to(self.w_dtype),
                    alpha=-self.lr_out,
                    beta=1.0,
                    out=self.w_out,
                )
                if self.w_out_mirror is not None:
                    self.w_out_mirror.copy_(self.w_out)
            else:
                grad_a = self._mm(gradient, self.w_b.T)  # (B, r)
                self.w_a.add_(features.T @ grad_a / denom, alpha=-self.lr_out)
        return loss, fired

    # -- 标定与诊断 ------------------------------------------------------------

    @torch.no_grad()
    def measure_current_ratio(self, tokens: torch.Tensor, *, steps: int = 120) -> float:
        """初始化后前 ``steps`` 步的 ``median(|I_rec|) / median(|I_in|)``。"""
        if self.recurrence != "random":
            return 0.0
        trace = self.new_trace(tokens.shape[0])
        direct: list[float] = []
        recurrent: list[float] = []
        for t in range(min(steps, tokens.shape[1] - 1)):
            direct.append(float(self.w_in[tokens[:, t]].abs().mean()))
            recurrent.append(float(self._recurrent_current(trace.z).abs().mean()))
            self.step(tokens[:, t], tokens[:, t + 1], trace, update=False)
        if not direct or statistics_median(direct) == 0:
            return 0.0
        return statistics_median(recurrent) / statistics_median(direct)

    @torch.no_grad()
    def calibrate_recurrent(
        self, tokens: torch.Tensor, *, target: float, steps: int = 120, tolerance: float = 0.05
    ) -> dict:
        """把 ``w_rec`` 整体缩放，使递推/输入电流比达到 ``target``。

        **不是一次乘到位**：递推电流是 ``w_rec·z`` 的乘积，而 ``z`` 本身随递推电流变，
        实测是**超线性**的（缩放 0.5 倍，电流比掉 6.6 倍）。固定步长会来回震荡，
        所以先用两点估出幂指数 ``k``（``r ∝ s^k``），再解出所需缩放，最后用局部指数精修一次。
        """
        if self.recurrence != "random":
            return {"target": target, "achieved": 0.0, "total_scale": 1.0, "rounds": 0}
        probe = 0.8
        base = self.measure_current_ratio(tokens, steps=steps)
        self.w_rec.mul_(probe)
        probed = self.measure_current_ratio(tokens, steps=steps)
        total_scale = probe
        exponent = 1.0
        if base > 0 and probed > 0 and abs(probed - base) > 1e-9:
            exponent = math.log(probed / base) / math.log(probe)
            exponent = min(max(exponent, 1.0), 6.0)
        achieved = probed
        rounds = 0
        for _ in range(4):
            if abs(achieved - target) <= tolerance:
                break
            factor = 1.2 if achieved <= 1e-6 else (target / achieved) ** (1.0 / exponent)
            factor = min(max(factor, 0.7), 1.4)
            self.w_rec.mul_(factor)
            total_scale *= factor
            achieved = self.measure_current_ratio(tokens, steps=steps)
            rounds += 1
        return {
            "target": target,
            "initial": round(base, 4),
            "exponent": round(exponent, 3),
            "total_scale": round(total_scale, 4),
            "achieved": round(achieved, 4),
            "rounds": rounds,
        }

    @torch.no_grad()
    def silent_rate(self, batch: int, *, steps: int = 50) -> float:
        """**零输入**下跑若干步的发放率——测网络会不会自持振荡。"""
        trace = self.new_trace(batch)
        fired = 0
        for _ in range(steps):
            current = self._recurrent_current(trace.z)
            new_v = self.alpha * trace.v + current - trace.z * self.threshold
            scaled = (new_v - self.threshold) / self.threshold
            fresh = (scaled > 0).to(trace.v.dtype)
            new_z = torch.where(trace.refractory > 0, torch.zeros_like(fresh), fresh)
            trace.refractory = torch.clamp(
                trace.refractory + self.n_refractory * new_z - 1.0,
                min=0.0,
                max=float(self.n_refractory),
            )
            trace.v, trace.z = new_v, new_z
            fired += int(new_z.sum())
        return fired / (steps * batch * self.n_rec)

    # -- 整块 -----------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        tokens: torch.Tensor,
        trace: LMTrace,
        *,
        valid: torch.Tensor | None = None,
        update: bool = True,
        learn_hidden: bool = True,
        learn_recurrent: bool = True,
    ) -> dict:
        """跑一整块 ``(B, L)``：内部按位移一位做 next-token。"""
        # 整块判定一次：全有效时走快路径（每步省 4–5 个内核），与慢路径逐位等价
        all_valid = valid is None or bool(valid.all())
        # 全部用张量累加，**循环里不做任何 `.item()`**——每一步同步一次是这条路径最大的隐藏开销
        total_loss = torch.zeros((), device=tokens.device)
        counted = torch.zeros((), device=tokens.device)
        fired = torch.zeros((), device=tokens.device)
        for t in range(tokens.shape[1] - 1):
            target = tokens[:, t + 1]
            if valid is not None:
                target = target.masked_fill(~valid[:, t + 1], -100)
            loss, shots = self.step(
                tokens[:, t],
                target,
                trace,
                update=update,
                learn_hidden=learn_hidden,
                learn_recurrent=learn_recurrent,
                all_valid=all_valid,
            )
            mask = (target != -100).to(loss.dtype)
            total_loss = total_loss + (loss * mask).sum()
            counted = counted + mask.sum()
            fired = fired + shots
        return {
            "ce": float(total_loss / counted.clamp(min=1)),
            "tokens": int(counted),
            "firing_rate": float(fired) / max(int(counted) * self.n_rec, 1),
        }

    @torch.no_grad()
    def evaluate(self, tokens: torch.Tensor, trace: LMTrace, *, valid=None) -> dict:
        """教师强制下的逐步 CE 与 top-1（评测用，不更新任何权重）。"""
        total_loss = 0.0
        hits = 0
        counted = 0
        fired = 0
        #: **top-k 才是"能不能当补全候选条"的指标。** top-1 只回答"第一个对不对"，
        #: 而产品里用户看的是前几个里有没有。这一项之前一直没量。
        topk_hits = {5: 0, 10: 0}
        #: **产品口径**：10 格候选条上，选中正确答案要按几下。
        #: 第 r 位就按 r 下；不在 10 格里就按 11 下（等于自己打）。**越低越好，与 BPTT 无关。**
        rank_cost_sum = 0.0
        # 按位置分桶（P2 的位置切片分析，零额外机时）
        buckets = ((0, 9), (10, 24), (25, 39), (40, 10**9))
        bucket_hits = [[0, 0] for _ in buckets]
        for t in range(tokens.shape[1] - 1):
            target = tokens[:, t + 1]
            if valid is not None:
                target = target.masked_fill(~valid[:, t + 1], -100)
            loss, shots = self.step(tokens[:, t], target, trace, update=False)
            mask = target != -100
            if not bool(mask.any()):
                continue
            logits = self._token_probabilities(self._features(trace))
            matched = logits.argmax(dim=-1)[mask] == target[mask]
            hits += int(matched.sum())
            probe_target = target[mask].unsqueeze(1)
            top10 = logits[mask].topk(10, dim=-1).indices
            for k in topk_hits:
                topk_hits[k] += int((top10[:, :k] == probe_target).any(dim=1).sum())
            matched_rank = top10 == probe_target
            in_top = matched_rank.any(dim=1)
            rank = matched_rank.to(torch.float32).argmax(dim=1) + 1
            cost = torch.where(in_top, rank, torch.full_like(rank, 11.0))
            rank_cost_sum += float(cost.sum())
            counted += int(mask.sum())
            for index, (low, high) in enumerate(buckets):
                if low <= t <= high:
                    bucket_hits[index][0] += int(matched.sum())
                    bucket_hits[index][1] += int(mask.sum())
                    break
            total_loss += float(loss[mask].sum())
            fired += shots
        return {
            "ce": total_loss / max(counted, 1),
            "top1": hits / max(counted, 1),
            "top5": topk_hits[5] / max(counted, 1),
            "top10": topk_hits[10] / max(counted, 1),
            "mean_rank_cost": rank_cost_sum / max(counted, 1),
            "tokens": counted,
            "firing_rate": float(fired) / max(int(counted) * self.n_rec, 1),
            "top1_by_pos": {
                f"{low}-{high if high < 10**9 else '+'}": round(hit / total, 4)
                for (low, high), (hit, total) in zip(buckets, bucket_hits, strict=True)
                if total
            },
        }
