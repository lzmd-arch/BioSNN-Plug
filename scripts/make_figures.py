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
* **W2 曲线（f2）**——读四批的 ``seed*.json`` 的 ``curve`` 字段（逐 epoch 的验证准确率），
  画在**同一张图**上：

  - ``sweep_results/w2_seeds/``——**验收批**，30 epoch、种子 0–4；
  - ``sweep_results/w2_seeds_100ep/``——**同一批种子、同一份代码，只把预算放到 100 epoch**；
  - ``sweep_results/w2_long/``——**预算延长、ALIF**，250 epoch、种子 0–2（2026-10-16 起为
    **对照**臂）；
  - ``sweep_results/w2_lif_long/``——**预算延长、LIF**，250 epoch、种子 0–2，**官方收官臂**
    （0.8889）。它与 ``w2_long/`` 是配对对照：同 lr、同种子、同预算，只差神经元模型。

  前两批前 30 轮逐点重合（5/5 种子 × 30/30 轮，两次独立运行），所以 30 epoch 的数字是这条
  轨迹的**前缀**，不是另一个结果——这一点由 ``_batch_curves`` 之外的打印行报出来，`30 epoch`
  的竖线标出验收预算停在哪。图上写两个「数字框」而不是一个：**验证**准确率（曲线画的）
  与**测试**准确率（头条数字）是两个量，2026-09-24 混淆过一次，不让它再有机会。
  两档 250 epoch 的臂另画**末 20 轮阴影带**——那是收敛判据（窗口均值增量）取样的地方，
  把判据取样的位置画出来，读者才看得见「收敛」这句话是在哪一段上说的。

  ⚠️ **故意不加「两批必须重合」的断言**：不重合是要在图上展示的发现，不是要拦下的错误。

  这张图**同时写两份**：``figures/f2-w2-epoch-accuracy.png``（仓级，README 引用）与
  ``paper/figures/f2-w2-epoch-accuracy.png``（论文引用）。刻意重复一份，是因为论文的图
  若走 ``../figures/`` 就要跨出 ``paper/`` 读文件，而 CI 用的是 tectonic——那条路径在
  tectonic 下的行为没在这个仓库里验过，不值得为省一个副本去赌。两份都由本脚本生成，
  不会各写各的。
* **W3 曲线**——读验收批（全新种子 45–64）的 ``curve`` 与 ``mean_steps``。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "figures"
#: 论文那份副本的去处。见模块 docstring 里"同时写两份"的理由。
PAPER_FIGURES = REPO_ROOT / "paper" / "figures"
#: W2 的**验收批**：30 epoch、种子 0–4。
W2_DIR = REPO_ROOT / "sweep_results" / "w2_seeds"
#: W2 的**追加批**：同一批种子、同一份代码，只把 epoch 预算放到 100。
W2_DIR_100 = REPO_ROOT / "sweep_results" / "w2_seeds_100ep"
#: W2 的**预算延长（ALIF）**：250 epoch、种子 0–2。2026-10-16 起是对照臂。
W2_DIR_LONG = REPO_ROOT / "sweep_results" / "w2_long"
#: W2 的**预算延长（LIF）**：250 epoch、种子 0–2。**官方收官臂**（0.8889）。
W2_DIR_LIF_LONG = REPO_ROOT / "sweep_results" / "w2_lif_long"
#: 收敛判据取样的窗口长度（末 N 轮）。判据本身是"窗口均值增量"，图上把窗口画出来。
W2_PLATEAU_WINDOW = 20
#: §七 第一阶段的验收 epoch 预算。图上画竖线用——它标出 30 epoch 的数字停在哪。
W2_ACCEPT_EPOCHS = 30
#: W3 的**验收批**：全新种子 45–64，从未参与任何选择。
W3_DIR = REPO_ROOT / "sweep_results" / "step14_confirm"
#: §七 第一阶段的步数判据，画成参考线。
W3_THRESHOLD = 200.0
#: 规模-资源扫描的目录模式：`sweep_results/w2_scale_n<size>/`。
W2_SCALE_GLOB = "w2_scale_n*"
#: 本机显存上限（MiB）。画成参考线——2048 那档越过了它，而它只是个大数时看不出来。
DEVICE_MEMORY_MIB = 8151.0
#: 二次参考线的斜率。数据离它有多远，是规模这一节最重要的结论。
QUADRATIC_REF = 2.0

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
    """f2：四个 epoch 预算画在一张图上，两档 250 epoch 的臂是主角。

    这张图要说的话是**「验收数字离走平还差多远」**，所以画法上三件事是硬要求：短的
    30 epoch 曲线必须与长的曲线**同轴**（前 30 轮重合才说明前者是后者的前缀）、第 30 轮要有
    竖线标出验收预算停在哪、以及**收敛判据取样的那个窗口要画出来**——判据是"末 20 轮窗口均值
    增量"，窗口不画出来，"收敛"二字就没有着落。
    """
    plt = _plt()
    acc = _batch_curves(W2_DIR, "30 epoch")
    mid = _batch_curves(W2_DIR_100, "100 epoch")
    alif = _batch_curves(W2_DIR_LONG, "250 epoch ALIF")
    lif = _batch_curves(W2_DIR_LIF_LONG, "250 epoch LIF")
    last = lif["epochs"][-1]
    mid_last = mid["epochs"][-1]
    if alif["epochs"][-1] != last:
        raise SystemExit("两档 250 epoch 的臂长度不一致，画不到一张图上。")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # 收敛判据的取样窗口：先铺底色，后面的曲线压在它上面。
    ax.axvspan(
        last - W2_PLATEAU_WINDOW,
        last,
        color="0.88",
        alpha=0.55,
        zorder=0,
        label=f"plateau window (last {W2_PLATEAU_WINDOW} epochs)",
    )

    # 长的先画、短的后画：重合处要让读者看得见 30 epoch 那条压在上面，不能反过来被盖掉。
    ax.fill_between(
        mid["epochs"],
        mid["lows"],
        mid["highs"],
        color="C1",
        alpha=0.10,
        label=f"{mid_last}-epoch batch, min–max over {_seeds(mid['n'])}",
    )
    ax.plot(
        mid["epochs"],
        mid["means"],
        color="C1",
        linewidth=1.6,
        label=f"{mid_last}-epoch batch, mean over {_seeds(mid['n'])}",
    )
    # 两档 250 epoch 的臂是配对对照（同 lr、同种子、同预算，只差神经元模型），所以并排画、
    # 用两种颜色分开，而不是与 30/100 那两批共用配色。
    for batch, color, tag in ((alif, "C0", "ALIF, control"), (lif, "C2", "LIF, official")):
        ax.fill_between(
            batch["epochs"],
            batch["lows"],
            batch["highs"],
            color=color,
            alpha=0.10,
        )
        ax.plot(
            batch["epochs"],
            batch["means"],
            color=color,
            linewidth=2.2,
            label=f"{last}-epoch {tag}, mean over {_seeds(batch['n'])}",
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
        f"validation, final window mean (these curves)\n"
        f"   {W2_ACCEPT_EPOCHS} ep: {acc['val_final']:.4f}"
        f"   {mid_last} ep: {mid['val_final']:.4f}"
        f"   {last} ep LIF: {lif['val_final']:.4f}  ALIF: {alif['val_final']:.4f}\n"
        f"test, held-out (the headline)\n"
        f"   {W2_ACCEPT_EPOCHS} ep: {acc['test_mean']:.4f}"
        f"   {mid_last} ep: {mid['test_mean']:.4f}"
        f"   {last} ep LIF: {lif['test_mean']:.4f}  ALIF: {alif['test_mean']:.4f}",
        xy=(0.985, 0.035),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title(
        f"W2 (e-prop, sMNIST): four budgets — "
        f"{acc['n']}×{W2_ACCEPT_EPOCHS}, {mid['n']}×{mid_last}, "
        f"and {lif['n']}×{last} under two neuron models",
        fontsize=10.5,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()

    out = FIGURES / "f2-w2-epoch-accuracy.png"
    fig.savefig(out, dpi=160, metadata=PNG_METADATA)
    # 论文那份副本——见模块 docstring 里"同时写两份"的理由。
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / out.name, dpi=160, metadata=PNG_METADATA)
    plt.close(fig)

    # 重合性：两批在前 30 轮应当逐点相同。不重合不拦（那正是要展示的发现），但要打出来。
    overlap = [e for e in acc["epochs"] if e <= mid_last]
    gaps = [abs(acc["means"][i] - mid["means"][e - 1]) for i, e in enumerate(overlap)]
    print(
        f"       {W2_DIR.name}: {acc['n']} 种子 × {len(acc['epochs'])} 点；"
        f"{W2_DIR_100.name}: {mid['n']} 种子 × {len(mid['epochs'])} 点"
    )
    print(
        f"       前 {len(overlap)} 轮两批均值曲线的最大逐点差 {max(gaps):.6f}"
        + ("（逐点重合 ✓）" if max(gaps) < 1e-9 else "（⚠️ 不重合，图上会显出来）")
    )
    print(
        f"       {last} 轮配对对照（同 lr、同种子 0–2、同预算，只差神经元模型）："
        f"LIF 均值 {lif['test_mean']:.4f}，ALIF 均值 {alif['test_mean']:.4f}"
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
    # 论文那份副本——见模块 docstring 里"同时写两份"的理由。
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / out.name, dpi=160, metadata=PNG_METADATA)
    plt.close(fig)
    return out


def _scale_points() -> list[dict]:
    """读规模-资源扫描的六个点（每档一个种子）。

    Raises:
        SystemExit: 产物不足两个点——那说明扫描还没跑。
    """
    points = []
    for path in sorted((REPO_ROOT / "sweep_results").glob(f"{W2_SCALE_GLOB}/seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        config = record["config"]
        points.append(
            {
                "n_rec": int(config["n_rec"]),
                "epochs": int(config["epochs"]),
                "peak_mb": float(record["peak_mb"]),
                "s_per_epoch": float(record["elapsed_s"]) / int(config["epochs"]),
            }
        )
    if len(points) < 2:
        raise SystemExit(
            "规模扫描的产物不足两个点。先跑：\n  uv run python -m research.eprop.sweep_scale"
        )
    points.sort(key=lambda point: point["n_rec"])
    return points


def _loglog_slope(points: list[dict], key: str) -> float:
    xs = [math.log(point["n_rec"]) for point in points]
    ys = [math.log(point[key]) for point in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - mx) ** 2 for x in xs
    )


def make_w2_scale() -> Path:
    """f4：规模-资源曲线，双面板 log-log。

    这张图要说三件事，画法上就三件硬要求：**两条轴的 log 刻度**（斜率才有意义）、
    **显存上限画成参考线**（2048 那档越过了它，不画出来就只是个大数）、以及**二次参考线**
    （数据离二次有多远，是这一节最重要的结论，不能只写斜率数字）。
    """
    plt = _plt()
    points = _scale_points()
    sizes = [point["n_rec"] for point in points]
    peaks = [point["peak_mb"] for point in points]
    times = [point["s_per_epoch"] for point in points]
    peak_slope = _loglog_slope(points, "peak_mb")
    time_slope = _loglog_slope(points, "s_per_epoch")
    #: 越界点：分配超过显存上限的那些档（0.5% 容差）。
    over = [point for point in points if point["peak_mb"] > DEVICE_MEMORY_MIB * 1.005]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    for ax, values, slope, ylabel, key in (
        (axes[0], peaks, peak_slope, "peak allocated memory (MiB)", "peak_mb"),
        (axes[1], times, time_slope, "seconds per epoch", "s_per_epoch"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.plot(sizes, values, color="C0", linewidth=2.0, marker="o", markersize=5)
        # 越界点圈出来：它不是同一类测量。
        if over:
            ax.plot(
                [point["n_rec"] for point in over],
                [point[key] for point in over],
                color="C3",
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgewidth=2.0,
                linestyle="none",
            )
        # 二次参考线：从最小点出发、斜率 2，用来看数据离二次有多远。
        anchor = values[0] / sizes[0] ** QUADRATIC_REF
        ax.plot(
            sizes,
            [anchor * size**QUADRATIC_REF for size in sizes],
            color="0.6",
            linestyle="--",
            linewidth=1.2,
            label=f"quadratic reference (slope {QUADRATIC_REF:.0f})",
        )
        ax.plot(
            [],
            [],
            color="C0",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=f"measured (fitted slope {slope:.2f})",
        )
        ax.set_xlabel("recurrent neurons $n_{\\mathrm{rec}}$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(size) for size in sizes], fontsize=8)
        ax.grid(alpha=0.3, which="both")

    axes[0].axhline(
        DEVICE_MEMORY_MIB,
        color="C3",
        linestyle=":",
        linewidth=1.6,
        label=f"device memory ({DEVICE_MEMORY_MIB:,.0f} MiB)",
    )
    if over:
        axes[0].plot(
            [],
            [],
            color="C3",
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgewidth=2.0,
            linestyle="none",
            label="exceeds device memory",
        )
        axes[0].annotate(
            "exceeds the device:\nspills to shared memory",
            xy=(over[0]["n_rec"], over[0]["peak_mb"]),
            xytext=(-10, -46),
            textcoords="offset points",
            fontsize=8,
            ha="right",
            arrowprops={"arrowstyle": "->", "color": "C3"},
        )
    # 图例放到最后：显存参考线与越界标记都是后面才加的，早画会把它们漏在legend 外。
    for ax in axes:
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle(
        "W2 (e-prop, sMNIST): resources against recurrent layer size "
        f"({len(points)} sizes, {points[0]['epochs']} epochs each)",
        fontsize=11,
    )
    fig.tight_layout()

    out = FIGURES / "f4-w2-scale.png"
    fig.savefig(out, dpi=160, metadata=PNG_METADATA)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / out.name, dpi=160, metadata=PNG_METADATA)
    plt.close(fig)

    print(
        f"       斜率：显存 {peak_slope:.3f}、耗时 {time_slope:.3f}"
        + (f"；越界点 {[p['n_rec'] for p in over]}" if over else "；无越界点")
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "figures",
        nargs="*",
        default=["f1", "w2-curve", "w3-curve", "w2-scale"],
        help="要画哪些图（f1 / w2-curve / w3-curve / w2-scale；默认全部）",
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
        elif name == "w2-scale":
            print(f"[画好] {make_w2_scale().relative_to(REPO_ROOT)}")
        else:
            raise SystemExit(f"没有这张图：{name!r}。可用：f1、w2-curve、w3-curve、w2-scale。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
