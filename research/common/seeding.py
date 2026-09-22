"""随机种子：控制与登记。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

计划书 §12.4 的复现模板要求「实验里用到的每一个种子，注明各自的用途」。一个
``set_seed(0)`` 满足不了这条：三条验证线各有各的随机源（权重初始化、数据打乱、
dropout 掩码、环境重置），事后没人说得清哪个种子管哪件事。

所以这里的用法是**先登记、再施加**：

    book = SeedBook(base=0)
    init_seed = book.register("权重初始化", derive_seed(book.base, "init"))
    apply_seed(init_seed)

:func:`derive_seed` 的派生方式刻意与骨架库一致：用 :func:`zlib.crc32` 而不是
内置的 ``hash()``。后者对字符串按进程加盐（``PYTHONHASHSEED``），同一个用途名在
不同进程里会派生出不同的种子——"固定种子即可复现"这句话就只在单进程内成立。
骨架库的 :class:`~biosnn_bus.bus.SparseRandomProjection` 出于同样的理由这么做，
并且有回归测试盯着（``test_projection_seed_survives_a_process_boundary``）。
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

__all__ = ["SeedBook", "apply_seed", "derive_seed"]


def derive_seed(base: int, purpose: str) -> int:
    """从基准种子按**用途名**派生一个稳定的子种子。

    同一个 ``(base, purpose)`` 在任何进程、任何机器上都得到同一个值。

    Args:
        base: 基准种子。
        purpose: 用途名，例如 ``"权重初始化"``、``"数据打乱"``。

    Returns:
        一个非负的 32 位整数。
    """
    return zlib.crc32(purpose.encode("utf-8")) ^ (int(base) & 0xFFFFFFFF)


def apply_seed(seed: int) -> None:
    """把种子施加到 python / numpy / torch 三处随机源。

    这是**唯一**该调用随机数库的地方。各验证线自己的模块不应直接
    ``np.random.seed(...)``——那样种子就不会出现在复现记录里。

    torch 是可选依赖：未安装时只设置 python 与 numpy，不报错。
    """
    seed = int(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed % (2**32))

    try:
        import torch
    except ImportError:  # pragma: no cover - 取决于环境
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SeedBook:
    """登记实验用到的每个种子及其用途。

    复现记录里的「随机种子」一栏由 :meth:`render` 生成。用途名要写清"这个种子管
    什么"，不要写"seed1"这类事后无意义的标签。

    Attributes:
        base: 基准种子。子种子由它派生，所以只记这一个数也够用了。
    """

    base: int
    _entries: dict[str, int] = field(default_factory=dict, init=False)

    def register(self, purpose: str, seed: int) -> int:
        """登记一个种子并返回它。

        Raises:
            ValueError: 同一个用途名被登记了不同的值——那多半是两处代码各自
                "以为自己在管这件事"，必须当场暴露，而不是后一个覆盖前一个。
        """
        seed = int(seed)
        if purpose in self._entries and self._entries[purpose] != seed:
            raise ValueError(
                f"用途 {purpose!r} 已被登记为 {self._entries[purpose]}，"
                f"现在又要登记为 {seed}。同名不同值说明有两处代码在抢同一件事的随机源。"
            )
        self._entries[purpose] = seed
        return seed

    def derive(self, purpose: str) -> int:
        """按用途派生并登记一个子种子。"""
        return self.register(purpose, derive_seed(self.base, purpose))

    def apply(self, purpose: str) -> int:
        """派生、登记并施加一个子种子，返回它的值。"""
        seed = self.derive(purpose)
        apply_seed(seed)
        return seed

    def as_dict(self) -> dict[str, int]:
        """用途名到种子值的副本。"""
        return dict(self._entries)

    def render(self) -> str:
        """复现记录里「随机种子」一栏的文本。"""
        base = f"base={self.base}"
        if not self._entries:
            return base
        items = "；".join(f"{purpose}={seed}" for purpose, seed in sorted(self._entries.items()))
        return f"{base}；{items}"
