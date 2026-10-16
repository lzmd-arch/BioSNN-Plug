"""W2 的规模-资源扫描：峰值显存与每 epoch 耗时随**循环层规模**怎么长。

## 为什么要有这一步

§5.5 现在只能报**一个规模**下的数字（n_rec=256 时 1,104.1 MiB、SHD 那次 3,220.6 MiB），
而论文里关于显存的主张是**跨规模**的——e-prop 的资格痕迹按突触存储、空间随神经元数二次增长
（§2.1，出处是 Pes et al. 2025 的 Table 3 与 §1.3.1）。**一个规模上的数字支撑不了二次增长的
主张**，所以要么补这条曲线，要么把话说小。这里选补。

顺带它还能回答一个**实现层**的问题：`docs/adr/ADR-0010` 记过，ALIF 的 `epsilon_a` 不可因式化、
而 LIF（β=0）下"整个迹的状态可降到 O(batch·N)"——**那句话写进了代码注释，但没验过**。
β=0 的曲线会不会露出二次项，是这次能量出来的东西。

## 设计（先写死，再看数据）

* **规模**：``n_rec ∈ {64, 128, 256, 512, 1024, 2048}``——等距五个十倍程里的六个点，log-log 上
  均匀分布，够拟一条直线。
* **预算**：**2 epoch × 种子 0**。这不是准确率测量：2 epoch、单种子下的准确率没有意义，
  **不得**与 §5.2 的任何数字相比。要的是**资源**，而资源由规模与形状决定，不由 epoch 数决定。
* **其余固定**：sMNIST、batch 64、lr 1e-2、**beta=0（LIF）**——与官方收官臂同一个神经元模型，
  这样曲线描述的是论文实际在跑的那条管线，不是另一个配置。
* **记录**：沿用 runner 自己写的产物（`peak_mb` 用 PyTorch 分配器口径，不是 nvidia-smi）。
  本脚本只做汇总与拟合。

## 已知风险与边界情况

* ⚠️ **大端可能 OOM**：8 GiB 的卡上，2048 那一档如果分配失败，**如实记为失败**，拟合只用完成的
  那几档，并把"最大只测到 X"写进论文——**不要**为了让曲线好看而把失败的点换成估计值。
* ⚠️ 本机同时跑别的 GPU 作业时**不要跑**：耗时那半张图会被污染（本仓实测过 1.5× 的差）。
  显存峰值不受负载影响，耗时受。
* 曲线的**斜率**是结果，不是前提：若实测不是二次，那说明这条管线里二次项被别的东西盖住了，
  结论照实写。

用法::

    uv run python -m research.eprop.sweep_scale --dry-run   # 只打印将要跑的命令
    uv run python -m research.eprop.sweep_scale             # 真跑（6 档 ≈ 2 h，视规模而定）
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DATA = REPO / "data"

#: 六个规模点，log2 等距。
SIZES = (64, 128, 256, 512, 1024, 2048)
#: 见模块文档：这不是准确率测量。
EPOCHS = 2
SEED = "0"
BETA = 0.0
LEARNING_RATE = "1e-2"
DRIVER_LOG = DATA / "w2_scale_driver.log"


def say(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    DRIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DRIVER_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def arm_name(n_rec: int) -> str:
    return f"w2_scale_n{n_rec}"


def command(n_rec: int) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "research.eprop.sweep_seeds",
        "--seeds",
        SEED,
        "--epochs",
        str(EPOCHS),
        "--n-rec",
        str(n_rec),
        "--beta",
        str(BETA),
        "--learning-rate-in",
        LEARNING_RATE,
        "--learning-rate-rec",
        LEARNING_RATE,
        "--device",
        "cuda",
        "--out-dir",
        f"data/{arm_name(n_rec)}",
    ]


def run_point(n_rec: int) -> int:
    name = arm_name(n_rec)
    say(f">> {name}（n_rec={n_rec}, {EPOCHS}ep, seed {SEED}, beta={BETA}）")
    started = time.perf_counter()
    with (DATA / f"{name}.log").open("wb") as handle:
        code = subprocess.run(
            command(n_rec), cwd=REPO, stdout=handle, stderr=subprocess.STDOUT
        ).returncode
    say(
        f"{'OK' if code == 0 else '!!'} {name} rc={code} "
        f"墙钟 {(time.perf_counter() - started) / 60:.1f} min"
    )
    return code


def point_summary(n_rec: int) -> dict | None:
    accs: list[float] = []
    peaks: list[float] = []
    elapsed: list[float] = []
    for path in sorted((DATA / arm_name(n_rec)).glob("seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        accs.append(float(record["test_accuracy"]))
        peaks.append(float(record.get("peak_mb", 0.0)))
        elapsed.append(float(record["elapsed_s"]))
    if not peaks:
        return None
    return {
        "n_rec": n_rec,
        "peak_mb": max(peaks),
        "s_per_epoch": statistics.fmean(elapsed) / EPOCHS,
        "acc": statistics.fmean(accs),
        "n": len(peaks),
    }


def fit_slope(points: list[dict], key: str) -> float | None:
    """log-log 最小二乘斜率。至少两个点才拟；否则返回 None。"""
    usable = [(p["n_rec"], p[key]) for p in points if p[key] > 0]
    if len(usable) < 2:
        return None
    xs = [math.log(n) for n, _ in usable]
    ys = [math.log(v) for _, v in usable]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="只打印将要跑的命令")
    args = parser.parse_args(argv)

    if args.dry_run:
        for n_rec in SIZES:
            print(" ".join(command(n_rec)))
        return 0

    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"工作区有未提交改动，先清理再跑：\n{dirty}")

    say("=" * 72)
    say(f"W2 规模-资源扫描：{len(SIZES)} 档 n_rec = {', '.join(map(str, SIZES))}")
    say(f"每档 {EPOCHS} epoch × 种子 {SEED}，beta={BETA}；这不是准确率测量")
    failures: list[int] = []
    for n_rec in SIZES:
        if run_point(n_rec) != 0:
            failures.append(n_rec)
            say(f"!! n_rec={n_rec} 失败（可能是 OOM）——如实记入，不补估计值")

    points = [s for n_rec in SIZES if (s := point_summary(n_rec)) is not None]
    say("规模-资源扫描汇总：")
    for point in points:
        say(
            f"  n_rec={point['n_rec']:<5} 峰值 {point['peak_mb']:9.1f} MiB"
            f"  {point['s_per_epoch']:8.1f} s/epoch  准确率 {point['acc']:.4f}（2 epoch，不作数）"
        )
    if failures:
        say(f"  ⚠️ 未完成的档：{failures}")
    for key, label in (("peak_mb", "显存"), ("s_per_epoch", "耗时")):
        slope = fit_slope(points, key)
        if slope is not None:
            say(f"  {label}的 log-log 斜率 = {slope:.3f}（1.0 = 线性，2.0 = 二次）")
    say("产物在 data/ 下——搬进 sweep_results/ 之前别去动工作区。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
