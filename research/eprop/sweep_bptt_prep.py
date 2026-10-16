"""BPTT 臂的预扫：**lr 档 × 优化器变体**两维，在跑满预算之前定下最终配方。

从 `data/draft_w2_bptt_lr.py` 搬进来的（2026-10-16）。搬进来的理由：它产出一个**论文数字**
（§5.2 的头条对照臂），而驱动器原本只以草稿形式存在于 `data/`——那是 gitignored 的，等于
配方与选择规则都不受版本控制。runner `sweep_bptt.py` 一直在仓里，驱动器此前不在，是个缺口。

## 为什么要有这一步

原方案是「e-prop 250ep 用 lr=1e-2，BPTT 也用 lr=1e-2，两者相减」。这有两个公平性问题：

1. **lr 未必是 BPTT 的最优**。如果 BPTT 的最优在 3e-3，头条数字读起来会是「e-prop 赢了」，
   而实际衡量的是「e-prop 在它自己的 lr 上赢了 BPTT 在一个不属于 BPTT 的 lr 上」。
2. **配方待遇不对等**。e-prop 有独立的读出学习率 ``eta_out=5e-2``；``BPTTLearner`` 却用
   单个 ``SGD(momentum=0.9)`` 覆盖全部参数，读出层只能跟着隐层的 lr 走。这条对 BPTT
   **不利**，而「honest measurement」这个标题赌不起它。

两个都处理掉，报的才是「双方各自最优配方下的差距」。论文里 A/B 两套数字都报——变体 A
本身就是一份 handicap 对照，「配方待遇对成绩的影响」是 Discussion 的一小节素材。

## 两维怎么定的

* **lr**：几何级数 ``1e-3 / 3e-3 / 1e-2``，3e-3 两侧各跨一个十倍程，与 e-prop 那四档
  （1e-3 / 2e-3 / 5e-3 / 1e-2）同量级，这样「两边最优落在哪」可以直接比。
  ``BPTTLearner`` 的默认 5e-3 落在 3e-3 与 1e-2 之间，若最优在那儿，三档的趋势会显示出来。
* **变体**：``A`` = 单 SGD 覆盖全部参数（现状，与 ``w2_bptt.log`` 那份 artifact 同配方）；
  ``B`` = 参数分组，隐层用扫描档、读出层用 5e-2（逐参数镜像 e-prop 的 ``eta_out``）。
  两个变体的**唯一**差别就是这个，``momentum`` 都保留 0.9。

⚠️ 变体维度**不改** ``BPTTLearner``——改它等于改那条已记录 artifact 的含义。实现在
``research/eprop/sweep_bptt.py`` 的 ``build_optimizer`` 里。

## 与 e-prop 的配对

``sweep_bptt`` 用同一套用途名派生种子，所以每个种子的划分/初始化/打乱/编码与 e-prop
那次逐位相同——预扫也不例外。种子 0-2（3 个）与最终 250ep 那次一致。

⚠️ **``--beta`` 必须与官方 e-prop 臂一致，默认因此是 0（LIF）**。官方收官臂在 2026-10-16 从
ALIF 换成 LIF（0.8889 @ 250ep），所以 BPTT 的对照臂也要走 LIF，否则「同预算、同架构、只差
学习规则」这句话就不成立。要跑 ALIF 版对照得显式传 ``--beta 0.07``。

## 纪律

⚠️ **产物先落 ``data/``**（被 ``.gitignore`` 忽略），跑完再整体搬进 ``sweep_results/``。
原因：provenance 判「工作区脏」用的是 ``git status --porcelain``，**未跟踪的新文件也算**
——中途落进仓库会让此后每一次运行的记录块都盖上「结论不可信」。
⚠️ **选择规则先预注册**：``sweep_results/w2_bptt_prep_prereg.md``，写在任何数据之前。

用法::

    uv run python -m research.eprop.sweep_bptt_prep --smoke    # 只验管线
    uv run python -m research.eprop.sweep_bptt_prep            # 真跑（6 组 ≈ 7–8 h，beta=0）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: 仓库根：从本文件往上找带 pyproject.toml 的那一层。写死 ``parents[1]`` 会在文件被搬动后
#: 静默指错目录——这个脚本已经从 data/ 搬过一次了。
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DATA = REPO / "data"

#: (目录名后缀, lr)。几何级数，3e-3 两侧各跨一个十倍程。
LR_ARMS = (("001", "1e-3"), ("003", "3e-3"), ("010", "1e-2"))
#: 优化器变体。A = 现状（handicap 对照），B = 读出层独立 lr。
VARIANTS = ("A", "B")
#: 变体 B 下读出层的 lr —— 与 e-prop 的 eta_out 逐参数相同。
LR_OUT = "5e-2"
SEEDS = "0-2"
EPOCHS = 30
#: 官方 e-prop 臂的神经元配置：LIF（beta=0）。改这里等于改对照的配对条件。
DEFAULT_BETA = 0.0
DRIVER_LOG = DATA / "w2_bptt_lr_driver.log"


def say(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with DRIVER_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def arm_name(tag: str, variant: str) -> str:
    return f"w2_bptt_lr{tag}_v{variant}"


def run_arm(name: str, lr: str, variant: str, beta: float) -> int:
    """跑一组，stdout/stderr 原样重定向到 data/<name>.log。"""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "research.eprop.sweep_bptt",
        "--seeds",
        SEEDS,
        "--device",
        "cuda",
        "--out-dir",
        f"data/{name}",
        "--epochs",
        str(EPOCHS),
        "--beta",
        str(beta),
        "--variant",
        variant,
        "--learning-rate",
        lr,
    ]
    if variant == "B":
        cmd += ["--learning-rate-out", LR_OUT]
    say(f">> {name}（BPTT, 变体 {variant}, eta={lr}, beta={beta}）")
    started = time.perf_counter()
    with (DATA / f"{name}.log").open("wb") as handle:
        code = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).returncode
    say(
        f"{'OK' if code == 0 else '!!'} {name} rc={code} "
        f"墙钟 {(time.perf_counter() - started) / 3600:.2f} h"
    )
    return code


def arm_summary(name: str) -> dict | None:
    accs: list[float] = []
    for path in sorted((DATA / name).glob("seed*.json")):
        accs.append(json.loads(path.read_text(encoding="utf-8"))["test_accuracy"])
    if not accs:
        return None
    return {"n": len(accs), "acc": sum(accs) / len(accs), "low": min(accs), "high": max(accs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="只验管线，不真跑")
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="适应强度。默认 0（LIF），与 2026-10-16 起的官方 e-prop 臂一致；ALIF 对照传 0.07",
    )
    args = parser.parse_args(argv)

    if shutil.which("uv") is None:
        raise SystemExit("PATH 里找不到 uv。")

    if args.smoke:
        # 两个变体都走一遍，否则"变体 B 的 flag 接错了"要到真跑六小时后才暴露。
        for variant in VARIANTS:
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "research.eprop.sweep_bptt",
                "--smoke",
                "--device",
                "cpu",
                "--variant",
                variant,
                "--learning-rate",
                "3e-3",
            ]
            if variant == "B":
                cmd += ["--learning-rate-out", LR_OUT]
            say(f"冒烟（变体 {variant}）：{' '.join(cmd)}")
            code = subprocess.run(cmd, cwd=REPO).returncode
            say(f"冒烟（变体 {variant}）rc={code}")
            if code != 0:
                return code
        return 0

    # 跑之前确认工作区干净——否则记录块会标「结论不可信」。
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"工作区有未提交改动，先清理再跑：\n{dirty}")

    say("=" * 72)
    say(
        f"BPTT 预扫：{len(LR_ARMS)} 档 lr × {len(VARIANTS)} 变体 × {SEEDS} × {EPOCHS}ep，"
        f"beta={args.beta}"
    )
    say("目的：让 BPTT 拿到自己的 lr 与自己的读出层待遇，再与 e-prop 比")
    say("选择规则见 sweep_results/w2_bptt_prep_prereg.md（写在数据之前）")
    best: dict | None = None
    for tag, lr in LR_ARMS:
        for variant in VARIANTS:
            name = arm_name(tag, variant)
            if run_arm(name, lr, variant, args.beta) != 0:
                say(f"!! {name} 失败了，汇总里会缺一条——别拿残缺的表去选配方")
                continue
            summary = arm_summary(name)
            if summary and (best is None or summary["acc"] > best["acc"]):
                best = {"name": name, "variant": variant, "lr": lr, **summary}

    say("BPTT 预扫汇总：")
    for variant in VARIANTS:
        for tag, lr in LR_ARMS:
            summary = arm_summary(arm_name(tag, variant))
            label = f"  变体 {variant}  eta={lr:<5}"
            if summary:
                say(
                    f"{label} 均值 {summary['acc']:.4f}"
                    f"（{summary['low']:.4f}-{summary['high']:.4f}，{summary['n']} 种子）"
                )
            else:
                say(f"{label} 缺产物")
    if best is None:
        say("!! 六组全无产物，无法选配方")
        return 1
    say(
        f"→ 最高均值：变体 {best['variant']}，eta={best['lr']}（{best['name']}，"
        f"均值 {best['acc']:.4f}）"
    )
    say("→ 最终配方按预注册规则定（1.0 pp 内视为并列，并列取更公平的变体 B）——见 prereg")
    extra = f" --learning-rate-out {LR_OUT}" if best["variant"] == "B" else ""
    say(
        f"→ 最终臂：sweep_bptt --seeds 0-2 --epochs 250 --beta {args.beta} "
        f"--variant <胜方> --learning-rate <胜方>{extra}"
    )
    say(
        "对照：e-prop **LIF** 30ep 四档 0.7657 / 0.7867 / 0.8088 / 0.8125；"
        "官方收官 LIF@250ep@1e-2 = 0.8889"
    )
    say("产物在 data/ 下——搬进 sweep_results/ 之前别去动工作区。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
