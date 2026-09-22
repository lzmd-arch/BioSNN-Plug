"""复现记录的生成（计划书 §12.4）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

[`docs/reproducibility.md`](../../docs/reproducibility.md) 立了一条硬规矩：

> 每次对外报告实验结论时，都要附上下面这段。**没附的结论视为未经复现。**

这个模块把那件事变成一次函数调用，而不是每次手抄十个字段——手抄的东西一定会漏，
而漏掉的往往是"工作区是否干净"这种当场否定整个结论的字段。

## 为什么必须自动化

模板里有四个字段是**人工填不对**的：

* ``Git commit`` / ``Git 状态``——手抄的是"我以为我提交了"，``--porcelain`` 给的是
  事实。工作区不干净 = 复现结果不可信，这条必须由命令输出而不是记忆决定；
* ``依赖快照``——``uv.lock`` 的哈希。它保证"同样的 lock 解析出同样的依赖版本"
  这句话有据可查；
* ``硬件``——含显存与算力级别（sm_120 之类的差异会改变可用的算子集）；
* ``显存峰值``——§6.2 把 8GB 列为硬约束，峰值是判断有没有踩线的唯一依据。
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["DegradationLog", "ReproRecord", "collect", "uv_lock_hash"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    """跑一条 git 命令。失败时返回带标记的字符串，而不是抛异常。

    git 不可用（例如从 tarball 解压出来的源码树）不该让实验跑不完——记录里如实
    写明"拿不到"即可。
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(git 不可用：{exc})"
    if result.returncode != 0:
        return f"(git 返回 {result.returncode})"
    return result.stdout.strip()


def uv_lock_hash() -> str:
    """``uv.lock`` 的 SHA-256；文件缺失时返回 ``"(缺失)"``。"""
    lock = REPO_ROOT / "uv.lock"
    if not lock.exists():
        return "(缺失)"
    digest = hashlib.sha256()
    with lock.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DegradationLog:
    """计划书 §6.2 的三条降级路径是否被触发。

    §12.4 要求 GPU 实验记录「是否触发了 §6.2 的降级路径」。默认全是 False——
    **没触发是常态**，所以只有显式登记过的才算数：悄悄降了规模却不写，等于把
    "8GB 装得下"这句话建立在一次没说明的妥协上。
    """

    scale_reduction: bool = False
    chunked_training: bool = False
    int8_traces: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def any_fired(self) -> bool:
        return self.scale_reduction or self.chunked_training or self.int8_traces

    def render(self) -> str:
        fired = [
            name
            for name, on in (
                ("规模降级", self.scale_reduction),
                ("分块训练", self.chunked_training),
                ("INT8 痕迹量化", self.int8_traces),
            )
            if on
        ]
        head = "、".join(fired) if fired else "未触发"
        if self.notes:
            return f"{head}（{'；'.join(self.notes)}）"
        return head


@dataclass
class ReproRecord:
    """一次实验的复现记录。字段与 ``docs/reproducibility.md`` 的模板一一对应。"""

    experiment: str
    command: str
    seeds: str
    elapsed_s: float
    gpu: str
    peak_mb: float | None = None
    degradation: DegradationLog | None = None
    notes: list[str] = field(default_factory=list)

    #: 由 :func:`collect` 填入
    commit: str = ""
    dirty_lines: int = 0
    collected_at: str = ""

    @property
    def is_dirty(self) -> bool:
        """工作区是否有未提交改动。**为真时结论不可信。**"""
        return self.dirty_lines > 0

    def render(self) -> str:
        """按 ``docs/reproducibility.md`` 的模板渲染成 ``text`` 块的内容。"""
        lines = [
            f"实验名称：{self.experiment}",
            f"日期：{self.collected_at}",
            f"Git commit：{self.commit}",
        ]
        if self.is_dirty:
            lines.append(
                f"Git 状态：有 {self.dirty_lines} 处未提交改动 —— ⚠️ 工作区不干净，"
                f"本结论不可信，请提交后重跑"
            )
        else:
            lines.append("Git 状态：干净")
        lines += [
            f"Python：{sys.version.split()[0]}",
            f"操作系统 / 架构：{platform.system()} {platform.release()} / {platform.machine()}",
            f"硬件：{self.gpu}",
            f"随机种子：{self.seeds}",
            f"依赖快照：uv.lock sha256={uv_lock_hash()}",
            f"运行命令：{self.command}",
            f"耗时：{self.elapsed_s:.1f} s",
        ]
        if self.peak_mb is not None:
            lines.append(f"显存峰值：{self.peak_mb:,.1f} MiB（§6.2 硬约束 8GB）")
        if self.degradation is not None:
            lines.append(f"§6.2 降级路径：{self.degradation.render()}")
        lines.extend(f"备注：{note}" for note in self.notes)
        return "\n".join(lines)

    def render_block(self) -> str:
        """渲染成可直接粘进 Markdown 的围栏块。"""
        return "```text\n" + self.render() + "\n```"


def collect(
    experiment: str,
    *,
    command: str | None = None,
    seeds: str = "",
    elapsed_s: float = 0.0,
    gpu: str | None = None,
    peak_mb: float | None = None,
    degradation: DegradationLog | None = None,
    notes: list[str] | None = None,
    device: Any = None,
) -> ReproRecord:
    """采集当下环境，组装一条复现记录。

    Args:
        experiment: 实验名，写清是哪条线的哪次运行。
        command: 运行命令；``None`` 时用 ``sys.argv`` 拼一个。
        seeds: :meth:`research.common.seeding.SeedBook.render` 的输出。
        elapsed_s: 耗时（秒）。
        gpu: 覆盖硬件描述；``None`` 时自动读取（需要 torch）。
        peak_mb: :func:`research.common.device.measure_peak_memory` 的 ``peak_mb``。
        degradation: §6.2 降级路径的登记。
        notes: 额外备注。
        device: 传给 :func:`research.common.device.describe_device` 的设备。

    Returns:
        :class:`ReproRecord`，用 :meth:`~ReproRecord.render_block` 取 Markdown。
    """
    if gpu is None:
        try:
            from research.common.device import describe_device

            gpu = describe_device(device).render()
        except ImportError:  # pragma: no cover - 取决于环境
            gpu = f"未知（未安装 torch）；平台 {platform.machine()}"

    if command is None:
        command = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else "(未知)"

    porcelain = _git("status", "--porcelain")
    dirty_lines = len(porcelain.splitlines()) if porcelain and not porcelain.startswith("(") else 0

    return ReproRecord(
        experiment=experiment,
        command=command,
        seeds=seeds,
        elapsed_s=float(elapsed_s),
        gpu=gpu,
        peak_mb=peak_mb,
        degradation=degradation,
        notes=list(notes or []),
        commit=_git("rev-parse", "HEAD"),
        dirty_lines=dirty_lines,
        collected_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
