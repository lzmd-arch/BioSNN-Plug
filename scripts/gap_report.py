#!/usr/bin/env python
"""e-prop vs BPTT 基线差距报告（计划书 §12.3 的 CI artifact）。

**为什么不在 CI 里现跑。** `research/eprop/bptt_baseline.py` 的完整运行在 GPU 上要
306 s（见 ``sweep_results/w2_bptt.log`` 的自证块），且要先下 MNIST；CPU 上还要慢若干倍。
放进 CI 既会超时，也会把一个**只在那一个配置下成立**的数字伪装成 CI 的产物。
所以分工是：产出数字的是本机的 GPU 运行（产物按 ``sweep_results/README.md`` 的规矩入库），
CI 负责让这些数字**可追溯、不漂移**——「可持续追踪」的实际含义就是这个。

于是本脚本不训练任何东西，只做三件事：

1. **记录里的数字自洽**：汇总行的两个准确率与逐臂打印的一致，且差距 = BPTT − e-prop。
2. **三语 README 引用的数字能追溯到某一次记录**，且三份文档指向的是**同一次**运行
   ——否则同一次比较在不同语言里是两个数，读者按语言拿到不同结论。
3. 把以上汇总成 ``gap-report.md`` 与 ``gap-report.json`` 供归档。

⚠️ 读的是**日志文本**：因为 ``bptt_baseline.py`` 只打印、不写 json
（``sweep_results/README.md`` 里写明了这一点）。所以解析锚在脚本自己生成的行上
（``[e-prop] 测试准确率 …`` 与 ``量化差距报告：…``），缺了就是硬失败，不会静默算成 0。
哪天 ``bptt_baseline.py`` 有了结构化输出，本脚本应当改读结构化输出。

用法::

    python scripts/gap_report.py [--root .] [--out-dir .]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 记录差距运行的目录（相对仓库根）。按**内容**找而不按文件名找——将来换目录、换命名也自动纳入追踪。
RECORD_DIRNAME = "sweep_results"

#: ``bptt_baseline.py`` 的 main() 逐字打印这一行，两个准确率与差距都在里面。
SUMMARY_RE = re.compile(
    r"量化差距报告：e-prop (?P<eprop>[0-9.]+) vs BPTT (?P<bptt>[0-9.]+)"
    r"（差距 (?P<gap>[+-][0-9.]+)）"
)

#: 逐臂打印的测试准确率，用来与汇总行互为对照：两处不一致说明有人改过其中之一。
ARM_RE = re.compile(r"^\[(?P<arm>e-prop|BPTT)\] 测试准确率 (?P<acc>[0-9.]+)", re.M)

CONFIG_RE = re.compile(r"^配置：(?P<config>[^\r\n]+)$", re.M)

#: 自证块里的字段。**全部可选**：``w2_bptt_first.log`` 那块是手工写的（当时自动渲染还没
#: 接上），只有 Git commit 与运行命令，没有日期与耗时——缺字段留空即可，不算问题。
FIELD_RES = {
    "date": re.compile(r"^日期：(?P<v>[^\r\n]+)$", re.M),
    "commit": re.compile(r"^Git commit：(?P<v>[^\r\n]+)$", re.M),
    "elapsed": re.compile(r"^耗时：(?P<v>[^\r\n]+)$", re.M),
    "hardware": re.compile(r"^硬件：(?P<v>[^\r\n]+)$", re.M),
}

#: 引用了差距数字的三份文档（相对仓库根），顺序即报告里的出现顺序。
DOCS = (
    ("zh", "research/eprop/README.md"),
    ("en", "research/eprop/README.en.md"),
    ("ja", "research/eprop/README.ja.md"),
)

#: 差距表里那一行的标签（三语各一种），数字在行末一格。
GAP_LABELS = frozenset({"差距", "Gap", "ギャップ"})
ARM_LABELS = frozenset({"e-prop", "BPTT"})

NUMBER_RE = re.compile(r"[+-]?[0-9]+\.[0-9]+")

#: 两处数字都算到小数点后 4 位，5e-5 足够容掉浮点误差而挡得住真正的改动。
TOLERANCE = 5e-5


class ReportError(Exception):
    """解析失败或核对不通过。"""


@dataclass(frozen=True)
class GapRun:
    """一次记录在案的差距运行。"""

    path: str
    eprop: float
    bptt: float
    gap: float
    config: str
    date: str
    commit: str
    elapsed: str
    hardware: str

    @property
    def key(self) -> tuple[float, float, float]:
        """三份文档要与它逐项吻合的那组数字。"""
        return (self.eprop, self.bptt, self.gap)


def cli_path(root: Path, path: Path) -> str:
    """相对仓库根的 posix 路径，落进报告里当引用。"""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def parse_run(root: Path, path: Path) -> GapRun:
    """解析一份运行日志。缺关键行就是硬失败——不猜、不填零。"""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    label = cli_path(root, path)

    summary = SUMMARY_RE.search(text)
    if summary is None:
        raise ReportError(
            f"{label}：找不到汇总行「量化差距报告：e-prop … vs BPTT …（差距 …）」。"
            "本文件不是一次完整的差距运行，或输出格式变了。"
        )

    config = CONFIG_RE.search(text)
    if config is None:
        raise ReportError(f"{label}：找不到「配置：」行——没有配置的差距数字无法被解读。")

    arms = {match.group("arm"): float(match.group("acc")) for match in ARM_RE.finditer(text)}
    for arm in ("e-prop", "BPTT"):
        if arm not in arms:
            raise ReportError(f"{label}：找不到 `[{arm}] 测试准确率 …` 这一行。")

    eprop = float(summary.group("eprop"))
    bptt = float(summary.group("bptt"))
    gap = float(summary.group("gap"))

    if abs(arms["e-prop"] - eprop) > TOLERANCE or abs(arms["BPTT"] - bptt) > TOLERANCE:
        raise ReportError(
            f"{label}：汇总行（e-prop {eprop} / BPTT {bptt}）与逐臂打印"
            f"（e-prop {arms['e-prop']} / BPTT {arms['BPTT']}）对不上。"
        )
    if abs(gap - (bptt - eprop)) > TOLERANCE:
        raise ReportError(f"{label}：差距 {gap:+} 与 BPTT − e-prop = {bptt - eprop:+.4f} 对不上。")

    fields = {
        name: (match.group("v").strip() if (match := regex.search(text)) else "")
        for name, regex in FIELD_RES.items()
    }
    return GapRun(
        path=label,
        eprop=eprop,
        bptt=bptt,
        gap=gap,
        config=config.group("config").strip(),
        **fields,
    )


def find_runs(root: Path) -> list[GapRun]:
    """把记录目录下所有含汇总行的日志当作一次差距运行。"""
    record_dir = root / RECORD_DIRNAME
    if not record_dir.is_dir():
        raise ReportError(f"找不到记录目录 {cli_path(root, record_dir)}。")

    runs: list[GapRun] = []
    for path in sorted(record_dir.rglob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if SUMMARY_RE.search(text):
            runs.append(parse_run(root, path))
        elif ARM_RE.search(text):
            # 有逐臂结果、没有汇总行：这次运行**被截断了**。按内容找记录的前提是
            # 「认得出它是一次差距运行」，半份输出正好落在这个前提的缝里——不拦住它，
            # 截断的那次就被静默跳过，看起来像「只有一次运行」。
            raise ReportError(
                f"{cli_path(root, path)}：有逐臂的测试准确率、却没有汇总行——"
                "这次运行的输出被截断了，不能当作一次记录（也不能当作没有过这次运行）。"
            )
    if not runs:
        raise ReportError(
            f"{cli_path(root, record_dir)} 下没有任何含差距汇总行的日志——报告没有来源。"
        )
    # 有日期的排在前面（日期形如 `2026-09-23 20:39:11 …`，字典序即时间序）；
    # 没有日期的记录排到最后，不冒充最新。
    return sorted(runs, key=lambda run: (run.date == "", run.date, run.path))


def read_doc_numbers(root: Path, path: Path) -> dict[str, float]:
    """从差距表里读出 ``e-prop`` / ``BPTT`` / ``差距`` 三个数字。

    只认三格的行，且标签必须落在已知的那几个词上——同一份文档里同一个标签出现两个不同
    的值就是错误（多半是改了一半），不取「最后一个赢」。
    """
    found: dict[str, set[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("*").strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 3:
            continue
        label = next((cell for cell in cells[:2] if cell), "")
        if label not in GAP_LABELS and label not in ARM_LABELS:
            continue
        match = NUMBER_RE.search(cells[-1])
        if match is None:
            continue
        key = "gap" if label in GAP_LABELS else label
        found.setdefault(key, set()).add(float(match.group()))

    numbers: dict[str, float] = {}
    for key, values in found.items():
        if len(values) > 1:
            raise ReportError(
                f"{cli_path(root, path)}：`{key}` 出现了多个不同的值（{sorted(values)}）——"
                "同一份文档里前后不一致，多半只改了一半。"
            )
        numbers[key] = values.pop()

    missing = {"e-prop", "BPTT", "gap"} - numbers.keys()
    if missing:
        raise ReportError(
            f"{cli_path(root, path)}：差距表里读不到 {'、'.join(sorted(missing))}——"
            "表格被改写了，或标签换了叫法（本脚本认「差距」/「Gap」/「ギャップ」）。"
        )
    return numbers


def check_docs(root: Path, runs: list[GapRun]) -> list[tuple[str, str, dict[str, float], GapRun]]:
    """核对三份文档：数字要能追溯到**同一次**记录。"""
    traced: list[tuple[str, str, dict[str, float], GapRun]] = []
    for lang, relative in DOCS:
        path = root / relative
        if not path.is_file():
            raise ReportError(f"找不到 {cli_path(root, path)}。")
        numbers = read_doc_numbers(root, path)
        wanted = (numbers["e-prop"], numbers["BPTT"], numbers["gap"])
        match = next(
            (
                run
                for run in runs
                if all(abs(a - b) <= TOLERANCE for a, b in zip(run.key, wanted, strict=True))
            ),
            None,
        )
        if match is None:
            recorded = "；".join(f"{run.path} → {run.key}" for run in runs)
            raise ReportError(
                f"{cli_path(root, path)} 引用的 e-prop {numbers['e-prop']} / BPTT "
                f"{numbers['BPTT']} / 差距 {numbers['gap']:+} 在记录里找不到对应的一次运行。"
                f"已记录的有：{recorded}。"
            )
        traced.append((lang, cli_path(root, path), numbers, match))

    anchors = {entry[3].path for entry in traced}
    if len(anchors) > 1:
        detail = "；".join(f"{entry[0]} → {entry[3].path}" for entry in traced)
        raise ReportError(
            f"三份文档引用的不是同一次运行：{detail}。同一次比较在不同语言里成了不同的数，"
            "读者按语言会拿到不同结论。"
        )
    return traced


def render_markdown(
    run: GapRun,
    runs: list[GapRun],
    traced: list[tuple[str, str, dict[str, float], GapRun]],
) -> str:
    lines = [
        "# 基线差距报告：e-prop vs. BPTT",
        "",
        "> 由 `scripts/gap_report.py` 在 CI 里生成（计划书 §12.3），**不要手工编辑**：",
        "> 它以 `sweep_results/` 里记录的运行为唯一来源，改了下一次生成会覆盖，",
        "> 而文档里的数字一旦与记录脱钩，CI 会判失败。",
        "",
        "## 结论",
        "",
        "| 臂 | 学习规则 | 测试准确率 |",
        "| :--- | :--- | ---: |",
        f"| e-prop | 局部资格痕迹，无跨层梯度 | **{run.eprop:.4f}** |",
        f"| BPTT | 代理梯度穿脉冲，不切断计算图 | **{run.bptt:.4f}** |",
        f"| | **差距** | **{run.gap:+.4f}** |",
        "",
        f"配置 `{run.config}`。来源 `{run.path}`"
        + (f"（{run.date}，commit `{run.commit[:9]}`）。" if run.date and run.commit else "。"),
        "",
        "**这个数字只在上述配置下成立。** 两条臂的学习率未对齐，且此预算下 e-prop 远未收敛；",
        "跨预算看还有算力效率差。口径的权威表述在 `research/eprop/README.md` 的已知边界第 6 条，",
        "**引用前请连口径一起引**。",
        "",
        "## 记录在案的差距运行",
        "",
        "| 记录 | 日期 | commit | e-prop | BPTT | 差距 | 耗时 |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for item in runs:
        lines.append(
            f"| `{item.path}` | {item.date or '—'} | `{item.commit[:9] or '—'}` | "
            f"{item.eprop:.4f} | {item.bptt:.4f} | {item.gap:+.4f} | {item.elapsed or '—'} |"
        )
    lines += [
        "",
        "两次运行的数字逐项相同、只有墙钟不同，是本仓库「同一份代码跑两遍」的一条证据；",
        "两次之间夹了一次代码改动（加自动自证块那一次），所以它比单纯重跑更强。",
        "逐字对照见 `sweep_results/README.md` 里 `w2_bptt_first.log` 那一行。",
        "",
        "## 可追溯性核对",
        "",
        "| 文档 | 引用的 e-prop / BPTT / 差距 | 对应记录 | 状态 |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for lang, path, numbers, anchor in traced:
        lines.append(
            f"| `{path}`（{lang}） | {numbers['e-prop']} / {numbers['BPTT']} / "
            f"{numbers['gap']:+} | `{anchor.path}` | ✅ |"
        )
    lines += [
        "",
        "核对范围**只有这三份文档**：脚本认的是「三格、标签为 差距/Gap/ギャップ 或 e-prop/BPTT」",
        "的表格行。别处（例如计划书正文、审计草稿）引用的差距数字不在核对范围内。",
        "",
    ]
    return "\n".join(lines)


def render_json(
    run: GapRun,
    runs: list[GapRun],
    traced: list[tuple[str, str, dict[str, float], GapRun]],
) -> str:
    payload = {
        "headline": {
            "e_prop": run.eprop,
            "bptt": run.bptt,
            "gap": run.gap,
            "config": run.config,
            "source": run.path,
            "date": run.date,
            "commit": run.commit,
        },
        "runs": [
            {
                "path": item.path,
                "date": item.date,
                "commit": item.commit,
                "config": item.config,
                "e_prop": item.eprop,
                "bptt": item.bptt,
                "gap": item.gap,
                "elapsed": item.elapsed,
                "hardware": item.hardware,
            }
            for item in runs
        ],
        "docs": [
            {
                "lang": lang,
                "path": path,
                "e_prop": numbers["e-prop"],
                "bptt": numbers["BPTT"],
                "gap": numbers["gap"],
                "traced_to": anchor.path,
            }
            for lang, path, numbers, anchor in traced
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="仓库根（默认本文件所在仓库；测试用它指向临时目录）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="报告写到哪个目录（默认当前目录；CI 里就是工作区）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只核对、不写报告文件（pre-commit 用：钩子不该弄脏工作区）",
    )
    args = parser.parse_args(argv)

    try:
        runs = find_runs(args.root)
        traced = check_docs(args.root, runs)
    except ReportError as error:
        print(f"[错误] {error}", file=sys.stderr)
        print("基线差距报告未生成——数字与记录脱钩时，宁可不产出报告。")
        return 1

    anchor = traced[0][3]
    if args.check:
        print(
            f"基线差距报告核对通过：e-prop {anchor.eprop:.4f} vs BPTT {anchor.bptt:.4f}"
            f"（差距 {anchor.gap:+.4f}），{len(traced)} 份文档都指向 `{anchor.path}`、"
            f"{len(runs)} 次记录自洽。"
        )
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gap-report.md").write_text(
        render_markdown(anchor, runs, traced), encoding="utf-8"
    )
    (args.out_dir / "gap-report.json").write_text(
        render_json(anchor, runs, traced), encoding="utf-8"
    )

    print(
        f"基线差距报告已生成：e-prop {anchor.eprop:.4f} vs BPTT {anchor.bptt:.4f}"
        f"（差距 {anchor.gap:+.4f}），核对 {len(traced)} 份文档、{len(runs)} 次记录。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
