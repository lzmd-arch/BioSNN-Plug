"""恢复率：局部规则拿回了线性上界的多少（第十轮预注册）。

```text
恢复率 = (R_局部 − R_零联想) / (R_上界 − R_零联想)
```

* ``R_上界``：线性上界（岭回归，非局部）；
* ``R_零联想``：不做任何映射直接比较的分数。**维度不同时用随机基线**（1/候选数），
  因为"直接比余弦"在那里没有定义；
* ``R_局部``：delta 规则的测试分数。

⚠️ 三者必须来自**同一份表示、同一个切分**。跨表示算出来的比值没有意义——这份模块
只做算术，不做配对检查，调用方自己负责。

用法::

    uv run python -m research.retrieval.recovery --local a.json --upper b.json --zero 0.034
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["recovery"]


def recovery(local: float, upper: float, zero: float) -> float:
    """返回恢复率；上界与零联想之间的差距为 0 时返回 ``nan``（除零，而不是编一个数）。"""
    span = upper - zero
    if span <= 0:
        return float("nan")
    return (local - zero) / span


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按第十轮预注册算恢复率")
    parser.add_argument("--local", type=Path, required=True, help="delta 的 summary.json")
    parser.add_argument("--upper", type=Path, required=True, help="ridge 的 summary.json")
    parser.add_argument(
        "--zero", type=float, default=None, help="零联想分数；缺省用 local 的随机基线"
    )
    args = parser.parse_args(argv)

    local = json.loads(args.local.read_text(encoding="utf-8"))
    upper = json.loads(args.upper.read_text(encoding="utf-8"))
    zero = args.zero if args.zero is not None else local["mean_chance"]

    value = recovery(local["mean_r1"], upper["mean_r1"], zero)
    print(
        f"局部 {local['mean_r1']:.4f}  上界 {upper['mean_r1']:.4f}  零联想 {zero:.4f}  "
        f"→ 恢复率 {value:.3f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
