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

    @classmethod
    def zeros(
        cls,
        batch: int,
        n_rec: int,
        n_vocab: int,
        *,
        n_state: int = 0,
        n_taus: int = 0,
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
        )

    def reset(self, rows: torch.Tensor) -> None:
        """把指定行清零（会话边界）。``rows`` 是 bool 掩码或下标张量。"""
        names = ["v", "z", "refractory", "eps_in", "eps_rec", "readout"]
        if self.readouts is not None:
            names += ["readouts"]
        if self.h is not None:
            names += ["h", "eps_e"]
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
        n_state: int = 32,
        tau_state: tuple[float, float] = (1.0, 128.0),
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if recurrence not in ("random", "none", "ssm"):
            raise ValueError(f"recurrence 只能是 random / none / ssm，收到 {recurrence!r}")
        self.n_vocab = int(n_vocab)
        self.n_rec = int(n_rec)
        self.fan_in = int(fan_in)
        self.threshold = float(threshold)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)
        self.alpha = float(math.exp(-1.0 / tau))
        self.decay_read = float(math.exp(-1.0 / tau_read))
        self.lr_in = float(lr_in)
        self.lr_rec = float(lr_rec)
        self.lr_out = float(lr_out)
        self.recurrence = recurrence
        self.n_state = int(n_state) if recurrence == "ssm" else 0
        self.device = torch.device(device)

        gen = generator or torch.Generator()
        self.w_in = torch.randn(n_vocab, n_rec, generator=gen).to(self.device) * input_std
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
        self.w_out = torch.randn(width, n_vocab, generator=gen).to(self.device) / math.sqrt(width)

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
            device=self.device,
        )

    # -- 前向与更新 ------------------------------------------------------------

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
    ) -> tuple[torch.Tensor, int]:
        """跑一个时刻。返回 (逐样本 CE, 本步发放数)。``targets`` 里的 -100 不参与损失。"""
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
            current = self.w_in[tokens] + self._recurrent_current(z_previous)

        new_v = self.alpha * trace.v + current - z_previous * self.threshold
        scaled = (new_v - self.threshold) / self.threshold
        refractory = trace.refractory > 0
        fresh = (scaled > 0).to(dtype)  # 硬阈值；本类不做反传，不需要伪导数
        new_z = torch.where(refractory, torch.zeros_like(fresh), fresh)
        new_r = torch.clamp(
            trace.refractory + self.n_refractory * new_z - 1.0,
            min=0.0,
            max=float(self.n_refractory),
        )
        trace.v = torch.where(active > 0, new_v, trace.v)
        trace.z = torch.where(active > 0, new_z, torch.zeros_like(new_z))
        trace.refractory = torch.where(active > 0, new_r, trace.refractory)
        trace.readout = self.decay_read * trace.readout + trace.z

        # 资格痕迹：用**本步之前**的脉冲与本步的输入累积（与 EventDrivenEProp 同口径）
        trace.eps_rec = self.alpha * trace.eps_rec + z_previous
        trace.eps_in = self.alpha * trace.eps_in
        trace.eps_in.scatter_add_(1, tokens.reshape(-1, 1), active.to(trace.eps_in.dtype))
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
        logits = features @ self.w_out  # (B, V)
        loss = torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=-100, reduction="none"
        )
        fired = int(trace.z.sum())

        if update and bool(valid.any()):
            probability = torch.softmax(logits, dim=-1)
            one_hot = torch.zeros_like(probability)
            one_hot.scatter_(1, targets.clamp(min=0).reshape(-1, 1), 1.0)
            gradient = (probability - one_hot) * active
            gradient = gradient / max(batch, 1)
            signal_sub = gradient @ self.w_out.T  # (B, M)
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
            psi = torch.where(trace.refractory > 0, torch.zeros_like(psi), psi)
            post = signal * psi * active

            if learn_hidden and self.recurrence != "ssm":
                self.w_in.add_(torch.einsum("bn,bv->vn", post, trace.eps_in), alpha=-self.lr_in)
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
            norm = float(features.pow(2).sum(dim=1).mean().clamp(min=1e-3))
            self.w_out.add_(features.T @ gradient / norm, alpha=-self.lr_out)
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
        total_loss = 0.0
        counted = 0
        fired = 0
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
            )
            mask = target != -100
            if bool(mask.any()):
                total_loss += float(loss[mask].sum())
                counted += int(mask.sum())
            fired += shots
        return {
            "ce": total_loss / max(counted, 1),
            "tokens": counted,
            "firing_rate": fired / max(counted * self.n_rec, 1),
        }

    @torch.no_grad()
    def evaluate(self, tokens: torch.Tensor, trace: LMTrace, *, valid=None) -> dict:
        """教师强制下的逐步 CE 与 top-1（评测用，不更新任何权重）。"""
        total_loss = 0.0
        hits = 0
        counted = 0
        fired = 0
        for t in range(tokens.shape[1] - 1):
            target = tokens[:, t + 1]
            if valid is not None:
                target = target.masked_fill(~valid[:, t + 1], -100)
            loss, shots = self.step(tokens[:, t], target, trace, update=False)
            mask = target != -100
            if not bool(mask.any()):
                continue
            logits = self._features(trace) @ self.w_out
            hits += int((logits.argmax(dim=-1)[mask] == target[mask]).sum())
            counted += int(mask.sum())
            total_loss += float(loss[mask].sum())
            fired += shots
        return {
            "ce": total_loss / max(counted, 1),
            "top1": hits / max(counted, 1),
            "tokens": counted,
            "firing_rate": fired / max(counted * self.n_rec, 1),
        }
