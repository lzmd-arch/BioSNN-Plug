#!/usr/bin/env python
"""论文数字对账：`paper/p4/main.tex` 里的数必须与 `sweep_results/` 的结果文件一致。

**为什么需要它**：论文里那张表与那串静态扫描数字，是从结果文件**手抄**进 LaTeX 的。
"一个数字只有一个家"这条纪律在文档层面一直靠人记；改一次实验，正文不会自己跟着变。
这条腿把"手抄"变成"对账"：不一致就红，且指出哪一格。

比对口径：
* 结果侧：按门禁过滤（|ΔW|/|W₀| ≥ 0.05）、按种子取中位数与四分位；
* 论文侧：只解析**表格数据行**与**静态扫描那段**的数字，三位小数；
* 容差：0.0006（三位小数四舍五入的最大误差）。

用法::

    python scripts/check_paper_numbers.py
    python scripts/check_paper_numbers.py --tex paper/p4/main.tex --results sweep_results/arbiter_synergy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = REPO_ROOT / "paper" / "p4" / "main.tex"
DEFAULT_RESULTS = REPO_ROOT / "sweep_results" / "arbiter_synergy"

#: 论文里表格行名 → 结果目录（论文的 MIX 行就是固定 α=0.5 那一臂）。
ARM_DIRS = {
    "H alone": "H",
    "E alone": "E",
    "MIX (fixed": "MIX_frozen",
}

#: 静态扫描：α → 结果目录。
SWEEP_DIRS = {
    "0.3": "alpha_0p3",
    "0.4": "alpha_0p4",
    "0.5": "MIX_frozen",
    "0.6": "alpha_0p6",
    "0.7": "alpha_0p7",
}

TOL = 0.0006


def load_fisher(results: Path, name: str) -> list[float]:
    payload = json.loads((results / name / "summary.json").read_text(encoding="utf-8"))
    return [float(row["fisher"]) for row in payload["rows"] if row.get("gate_passed")]


def quartiles(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75)))


def parse_table(tex: str) -> dict[str, tuple[float, float, float]]:
    """从 synergy 表的数据行里取 (median, q1, q3)。"""
    found: dict[str, tuple[float, float, float]] = {}
    pattern = re.compile(
        r"^(H alone|E alone|MIX \(fixed)[^&]*&\s*(?:\\textbf\{)?([\d.]+)\}?\s*&\s*\[([\d.]+),\s*([\d.]+)\]",
        re.MULTILINE,
    )
    for match in pattern.finditer(tex):
        median = float(match.group(2))
        found[match.group(1)] = (float(match.group(3)), median, float(match.group(4)))
    return found


def parse_sweep(tex: str) -> dict[str, float]:
    """从静态扫描那段取 α → median。

    句子形如 ``0.364 [0.361, 0.450] at 0.3, 0.577 [0.563, 0.627] at 0.4, ... at 0.5, ...
    at 0.6 and 0.621 [0.604, 0.622] at 0.7``。
    """
    found: dict[str, float] = {}
    for match in re.finditer(r"([\d.]+)\s*\[[\d.,\s]+\]\s*at\s*(\d\.\d)", tex):
        found[match.group(2)] = float(match.group(1))
        # 第一档写作 "the Fisher ratio is 0.364 [..] at 0.3"
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="论文数字与结果文件对账")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    tex = args.tex.read_text(encoding="utf-8")
    problems: list[str] = []
    checked = 0

    table = parse_table(tex)
    if not table:
        problems.append("没从论文里解析到任何表格数据行（表格结构变了？）")
    for label, dir_name in ARM_DIRS.items():
        key = next((k for k in table if k.startswith(label)), None)
        if key is None:
            problems.append(f"论文表格里找不到 {label} 那一行")
            continue
        q1, median, q3 = quartiles(load_fisher(args.results, dir_name))
        paper = table[key]
        for name, computed, printed in (
            ("中位数", median, paper[1]),
            ("q1", q1, paper[0]),
            ("q3", q3, paper[2]),
        ):
            checked += 1
            if abs(computed - printed) > TOL:
                problems.append(
                    f"{key} 的{name}：论文写 {printed:.3f}，结果文件算出 {computed:.4f}"
                )

    sweep = parse_sweep(tex)
    if len(sweep) < 5:
        problems.append(f"静态扫描段只解析到 {len(sweep)} 档（应为 5 档），句子结构可能变了")
    for alpha, dir_name in SWEEP_DIRS.items():
        if alpha not in sweep:
            problems.append(f"静态扫描段缺少 α = {alpha} 那一档")
            continue
        _, median, _ = quartiles(load_fisher(args.results, dir_name))
        checked += 1
        if abs(median - sweep[alpha]) > TOL:
            problems.append(
                f"α = {alpha} 的中位数：论文写 {sweep[alpha]:.3f}，结果算出 {median:.4f}"
            )

    if problems:
        print("论文数字与结果文件不一致：", file=sys.stderr)
        for item in problems:
            print(f"· {item}", file=sys.stderr)
        return 1
    print(f"论文数字对账通过：{checked} 个数与结果文件一致")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
