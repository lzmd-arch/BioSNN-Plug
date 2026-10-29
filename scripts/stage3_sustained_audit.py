"""阶段三口径复核：把已有 summary 里的 best / 末次 / 末三段并列，回答「有没有持续学会」。

**后验分析，不产生新数据**：只读 `sweep_results/**/summary.json`。
`best_steps` 是评测点上的最大值（默认 32 次评测 × 10 回合），会被噪声抬高；末次和末三段
均值才是稳定口径。本脚本不改变任何既有结论，只把两种口径并排列出来。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def stats_for(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    per_seed = data.get("per_seed") or []
    if len(per_seed) < 5 or "best_steps" not in per_seed[0]:
        return None
    last = [row["eval_scores"][-1]["mean_steps"] for row in per_seed if row.get("eval_scores")]
    final3 = [
        statistics.fmean(point["mean_steps"] for point in row["eval_scores"][-3:])
        for row in per_seed
        if row.get("eval_scores")
    ]
    return {
        "path": path.relative_to(REPO).as_posix(),
        "seeds": len(per_seed),
        "best_median": statistics.median([row["best_steps"] for row in per_seed]),
        "last_median": statistics.median(last) if last else None,
        "final3_median": statistics.median(final3) if final3 else None,
    }


def main() -> int:
    rows = []
    for path in sorted(REPO.glob("sweep_results/**/summary.json")):
        if "rstdp" not in path.as_posix():
            continue
        row = stats_for(path)
        if row:
            rows.append(row)
    print("| 结果 | 种子 | best 中位 | 末次中位 | 末三段中位 |")
    print("| :--- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| `{row['path']}` | {row['seeds']} | {row['best_median']:.1f} | "
            f"{row['last_median']:.1f} | {row['final3_median']:.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
