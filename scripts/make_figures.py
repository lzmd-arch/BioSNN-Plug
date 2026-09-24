"""把论文要用的图稳定地重画出来，落到 `figures/`。

**为什么要有这个脚本**：图不能只存在于某次运行的屏幕上。每一张图都要能从**入库的产物**
重新画出来，且**逐字节可复现**（PNG 里会变的软件版本串已用 ``metadata={"Software": None}``
去掉）。

用法::

    uv run python scripts/make_figures.py            # 全部
    uv run python scripts/make_figures.py f1         # 只画 F1
    uv run python scripts/make_figures.py w2-curve   # 只画 W2 的 epoch–准确率曲线
    uv run python scripts/make_figures.py w3-curve   # 只画 W3 的贪心评测曲线

## 图的三个数据源

* **F1（三面板脉冲格栅图）**——**不在这里重画**，而是带着 ``BIOSNN_FIGURE_DIR`` 去跑
  ``examples/quickstart_register_plugin.py``。图的绘制代码只有那一份真相源；在这里再抄一遍
  迟早会与演示漂移。
* **W2 曲线（f2）**——读两批的 ``seed*.json`` 的 ``curve`` 字段（逐 epoch 的验证准确率），
  画在**同一张图**上：

  - ``sweep_results/w2_seeds/``——**验收批**，30 epoch、种子 0–4；
  - ``sweep_results/w2_seeds_100ep/``——**同一批种子、同一份代码，只把预算放到 100 epoch**。

  两批前 30 轮逐点重合（5/5 种子 × 30/30 轮，两次独立运行），所以 30 epoch 的数字是这条
  轨迹的**前缀**，不是另一个结果——这一点由 ``_batch_curves`` 之外的打印行报出来，`30 epoch`
  的竖线标出验收预算停在哪。图上写两个「数字框」而不是一个：**验证**准确率（曲线画的）
  与**测试**准确率（头条数字）是两个量，2026-09-24 混淆过一次，不让它再有机会。

  ⚠️ **故意不加「两批必须重合」的断言**：不重合是要在图上展示的发现，不是要拦下的错误。
* **W3 曲线**——读验收批（全新种子 45–64）的 ``curve`` 与 ``mean_steps``。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "figures"
#: W2 的**验收批**：30 epoch、种子 0–4。
W2_DIR = REPO_ROOT / "sweep_results" / "w2_seeds"
#: W2 的**追加批**：同一批种子、同一份代码，只把 epoch 预算放到 100。
W2_DIR_100 = REPO_ROOT / "sweep_results" / "w2_seeds_100ep"
#: §七 第一阶段的验收 epoch 预算。图上画竖线用——它标出 30 epoch 的数字停在哪。
W2_ACCEPT_EPOCHS = 30
#: W3 的**验收批**：全新种子 45–64，从未参与任何选择。
W3_DIR = REPO_ROOT / "sweep_results" / "step14_confirm"
#: §七 第一阶段的步数判据，画成参考线。
W3_THRESHOLD = 200.0

# PNG 里会写进 matplotlib 的版本串；去掉它，产物才能逐字节比对。
PNG_METADATA = {"Software": None}


def _plt():
    import matplotlib

    matplotlib.use("Agg")  # 无头：这个脚本永远不弹窗
    import matplotlib.pyplot as plt

    return plt


def make_f1() -> Path:
    """跑一遍快速开始演示（带 ``BIOSNN_FIGURE_DIR``），由它自己把 F1 落盘。"""
    out = FIGURES / "f1-spike-raster.png"
    env = {**os.environ, "BIOSNN_FIGURE_DIR": str(FIGURES), "MPLBACKEND": "Agg"}
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "quickstart_register_plugin.py")],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    if not out.exists():
        raise SystemExit(f"演示跑完了，但没有产出 {out}——检查 BIOSNN_FIGURE_DIR 那段。")
    return out


def _seeds(n: int) -> str:
    """图例里的 ``N seeds``。单数写 seed——试画时踩过一次 ``1 seeds``。"""
    return f"{n} seed" if n == 1 else f"{n} seeds"


def _batch_curves(dirpath: Path, label: str) -> dict:
    """读一批种子的逐 epoch 验证准确率，聚成均值与 min–max 带。

    Returns:
        ``epochs`` / ``means`` / ``lows`` / ``highs`` / ``n`` / ``test_mean`` / ``val_final``。

    Raises:
        SystemExit: 目录不在、没有 json 带 ``curve``、或**批内**各条曲线的 epoch 网格不一致。
            （网格只在**批内**要求一致：两批长度本来就不同，那正是 f2 要画的。）
    """
    if not dirpath.is_dir():
        raise SystemExit(f"[{label}] 找不到 {dirpath}。")
    per_seed: list[list[tuple[int, float]]] = []
    tests: list[float] = []
    for path in sorted(dirpath.glob("seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        curve = record.get("curve")
        if not curve:
            continue
        per_seed.append([(int(p["epoch"]), float(p["val_accuracy"])) for p in curve])
        tests.append(float(record["test_accuracy"]))
    if not per_seed:
        raise SystemExit(
            f"[{label}] {dirpath} 里的 json 都没有 `curve` 字段。\n"
            "那是 2026-09-23 之前跑的批次——加 `curve` 之前，逐 epoch 的验证准确率只打印、不落盘。\n"
            "重跑一次即可：uv run python -m research.eprop.sweep_seeds --seeds 0-4 --device cuda"
        )

    epochs = [e for e, _ in per_seed[0]]
    for other in per_seed[1:]:
        if [e for e, _ in other] != epochs:
            raise SystemExit(f"[{label}] 批内各条曲线的 epoch 网格不一致——检查评测节奏。")

    means, lows, highs = [], [], []
    for i in range(len(epochs)):
        values = [curve[i][1] for curve in per_seed]
        means.append(statistics.fmean(values))
        lows.append(min(values))
        highs.append(max(values))
    return {
        "epochs": epochs,
        "means": means,
        "lows": lows,
        "highs": highs,
        "n": len(per_seed),
        "test_mean": statistics.fmean(tests),
        "val_final": means[-1],
    }


def make_w2_curve() -> Path:
    """f2：同一批种子、两个 epoch 预算，画在一张图上。

    这张图要说的话是**「验收数字离走平还差多远」**，所以画法上两件事是硬要求：短的
    30 epoch 曲线必须与长的 100 epoch 曲线**同轴**（前 30 轮重合才说明前者是后者的前缀），
    且第 30 轮要有竖线标出验收预算停在哪。
    """
    plt = _plt()
    acc = _batch_curves(W2_DIR, "30 epoch")
    long = _batch_curves(W2_DIR_100, "100 epoch")
    last = long["epochs"][-1]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # 长的先画、短的后画：重合处要让读者看得见 30 epoch 那条压在上面，不能反过来被盖掉。
    ax.fill_between(
        long["epochs"],
        long["lows"],
        long["highs"],
        color="C0",
        alpha=0.13,
        label=f"{last}-epoch batch, min–max over {_seeds(long['n'])}",
    )
    ax.plot(
        long["epochs"],
        long["means"],
        color="C0",
        linewidth=2.0,
        label=f"{last}-epoch batch, mean over {_seeds(long['n'])}",
    )
    ax.fill_between(
        acc["epochs"],
        acc["lows"],
        acc["highs"],
        color="C3",
        alpha=0.18,
        label=f"{W2_ACCEPT_EPOCHS}-epoch batch, min–max over {_seeds(acc['n'])}",
    )
    ax.plot(
        acc["epochs"],
        acc["means"],
        color="C3",
        linewidth=1.8,
        linestyle="--",
        marker="o",
        markersize=4,
        markerfacecolor="none",
        markeredgewidth=1.0,
        label=f"{W2_ACCEPT_EPOCHS}-epoch batch, mean over {_seeds(acc['n'])}",
    )
    ax.axvline(
        W2_ACCEPT_EPOCHS,
        color="0.35",
        linestyle=":",
        linewidth=1.3,
        label=f"acceptance budget ends (epoch {W2_ACCEPT_EPOCHS})",
    )

    # 验证与测试分开写、各自标名——curve 画的是验证，头条数字是测试。
    ax.annotate(
        f"validation (these curves)\n"
        f"    epoch {W2_ACCEPT_EPOCHS}: {acc['val_final']:.4f}    epoch {last}: {long['val_final']:.4f}\n"
        f"test, 10k held-out (the headline)\n"
        f"    {W2_ACCEPT_EPOCHS} epochs: {acc['test_mean']:.4f}    {last} epochs: {long['test_mean']:.4f}",
        xy=(0.985, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8.5,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title(
        f"W2 (e-prop, sMNIST): same seeds, two budgets — "
        f"{acc['n']} × {W2_ACCEPT_EPOCHS} epochs vs {long['n']} × {last} epochs",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()

    out = FIGURES / "f2-w2-epoch-accuracy.png"
    fig.savefig(out, dpi=160, metadata=PNG_METADATA)
    plt.close(fig)

    # 重合性：两批在前 30 轮应当逐点相同。不重合不拦（那正是要展示的发现），但要打出来。
    overlap = [e for e in acc["epochs"] if e <= last]
    gaps = [abs(acc["means"][i] - long["means"][e - 1]) for i, e in enumerate(overlap)]
    print(
        f"       {W2_DIR.name}: {acc['n']} 种子 × {len(acc['epochs'])} 点；"
        f"{W2_DIR_100.name}: {long['n']} 种子 × {len(long['epochs'])} 点"
    )
    print(
        f"       前 {len(overlap)} 轮两批均值曲线的最大逐点差 {max(gaps):.6f}"
        + ("（逐点重合 ✓）" if max(gaps) < 1e-9 else "（⚠️ 不重合，图上会显出来）")
    )
    return out


def _w3_records() -> list[dict]:
    """读回验收批每个种子的完整记录（曲线 + 终值 + 峰值）。

    ⚠️ **``curve[-1]`` 与 ``mean_steps`` 不是同一个量**：前者是**最后一个检查点**
    （第 800 回合）那次评测，后者是在**最终权重**上重跑的 10 个贪心回合。两者逐种子都不同
    （种子 45：279.7 对 232.1），而 README 的头条数字用的是后者——所以图上的标注必须取
    ``mean_steps``，取 ``curve[-1]`` 会与 README 对不上。

    Raises:
        SystemExit: 目录不在，或没有任何 json 带 ``curve`` / ``mean_steps``。
    """
    if not W3_DIR.is_dir():
        raise SystemExit(
            f"找不到 {W3_DIR}。先跑验收批：\n"
            "  uv run python -m research.rstdp.cartpole --seeds 45-64 --device cpu"
        )
    records = []
    for path in sorted(W3_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("curve") and "mean_steps" in record:
            records.append(record)
    if not records:
        raise SystemExit(f"{W3_DIR} 里的 json 都没有 `curve` / `mean_steps` 字段。")
    if len({len(r["curve"]) for r in records}) != 1:
        raise SystemExit("各条曲线的长度不一致，画不到一张图上——检查评测间隔。")
    return records


def make_w3_curve() -> Path:
    """W3 的贪心评测曲线：每个种子一条细线 + 中位数粗线 + 判据参考线。

    这张图要说的是**游走**，不是「学到了多少」——所以细线全留着，不画均值带
    （均值带会把游走抹平，那正好掩盖了要展示的东西）。
    """
    plt = _plt()
    records = _w3_records()
    curves = [[float(v) for v in r["curve"]] for r in records]
    n_seeds, n_points = len(curves), len(curves[0])
    episodes = [50 * (i + 1) for i in range(n_points)]
    median = [statistics.median([c[i] for c in curves]) for i in range(n_points)]
    finals = [float(r["mean_steps"]) for r in records]
    peaks = [float(r["peak_mean_steps"]) for r in records]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for curve in curves:
        ax.plot(episodes, curve, color="0.6", linewidth=0.7, alpha=0.8)
    ax.plot(
        episodes,
        median,
        color="C3",
        linewidth=2.2,
        marker="o",
        markersize=3,
        label=f"median over {n_seeds} seeds",
    )
    ax.axhline(
        W3_THRESHOLD,
        color="C0",
        linestyle="--",
        linewidth=1.2,
        label=f"phase-1 criterion ({W3_THRESHOLD:.0f} steps)",
    )

    # 把「终值 vs 峰值」这对数字直接标在图上——它正是「游走仍在」的量度。
    ax.annotate(
        f"final weights, median = {statistics.median(finals):.1f} steps   (the headline)\n"
        f"per-seed peaks, median = {statistics.median(peaks):.1f} steps",
        xy=(0.985, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("greedy evaluation (mean steps over 10 episodes)")
    ax.set_title(
        f"W3 (R-STDP + TD-LTP, CartPole): greedy eval every 50 episodes, {n_seeds} held-out seeds",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    out = FIGURES / "f3-w3-greedy-curves.png"
    fig.savefig(out, dpi=160, metadata=PNG_METADATA)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "figures",
        nargs="*",
        default=["f1", "w2-curve", "w3-curve"],
        help="要画哪些图（f1 / w2-curve / w3-curve；默认全部）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in args.figures:
        if name == "f1":
            print(f"[画好] {make_f1().relative_to(REPO_ROOT)}")
        elif name == "w2-curve":
            print(f"[画好] {make_w2_curve().relative_to(REPO_ROOT)}")
        elif name == "w3-curve":
            print(f"[画好] {make_w3_curve().relative_to(REPO_ROOT)}")
        else:
            raise SystemExit(f"没有这张图：{name!r}。可用：f1、w2-curve、w3-curve。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
