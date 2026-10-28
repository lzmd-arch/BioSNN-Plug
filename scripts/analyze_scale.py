"""把 W2 规模扫描读成曲线，并外推 8 GB 容量与 500k 目标。

输入：``sweep_results/w2_scale_n*/seed*.json``（已有结果，2 epoch × 种子 0）。
输出：``sweep_results/phase4_scale/summary.json`` + 终端表。

**这是外推，不是实测**：斜率来自 64–2048；超出区间的点只用于说明量级。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "sweep_results"
OUT = REPO / "sweep_results" / "phase4_scale" / "summary.json"
TARGETS = (4096, 10_000, 100_000, 500_000)
GPU_MB = 8192.0


def slope(points: list[tuple[int, float]]) -> float:
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(v) for _, v in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom


def main() -> int:
    points: list[dict[str, float]] = []
    for directory in sorted(DATA.glob("w2_scale_n*")):
        peaks, times = [], []
        for path in sorted(directory.glob("seed*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            peaks.append(float(record["peak_mb"]))
            times.append(float(record["elapsed_s"]) / 2.0)
        if peaks:
            points.append(
                {
                    "n_rec": int(directory.name.split("_n")[-1]),
                    "peak_mb": max(peaks),
                    "s_per_epoch": sum(times) / len(times),
                }
            )
    points.sort(key=lambda point: point["n_rec"])
    if len(points) < 2:
        raise SystemExit("规模点不足两个，先跑 sweep_scale。")
    memory_slope = slope([(p["n_rec"], p["peak_mb"]) for p in points])
    time_slope = slope([(p["n_rec"], p["s_per_epoch"]) for p in points])
    largest = points[-1]
    extrapolated = {
        str(target): largest["peak_mb"] * (target / largest["n_rec"]) ** memory_slope
        for target in TARGETS
    }
    max_under_8gb = largest["n_rec"] * (GPU_MB / largest["peak_mb"]) ** (1.0 / memory_slope)
    summary = {
        "source": "sweep_results/w2_scale_n*/seed*.json (2 epoch, seed 0, beta=0)",
        "points": points,
        "memory_loglog_slope": memory_slope,
        "time_loglog_slope": time_slope,
        "gpu_mb": GPU_MB,
        "max_n_rec_under_gpu": max_under_8gb,
        "extrapolated_peak_mb": extrapolated,
        "note": "extrapolation, not measurement; 500k is far outside the fitted range",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("n_rec  peak_MiB  s/epoch")
    for point in points:
        print(f"{point['n_rec']:>5}  {point['peak_mb']:9.1f}  {point['s_per_epoch']:8.1f}")
    print(f"memory slope {memory_slope:.3f}; time slope {time_slope:.3f}")
    print(f"max n_rec under {GPU_MB:.0f} MB ≈ {max_under_8gb:.0f}")
    for target, peak in extrapolated.items():
        print(f"extrapolated {target:>6}: {peak:,.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
