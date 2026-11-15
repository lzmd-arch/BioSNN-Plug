"""把评测 JSON 注入模型卡：数字**只从结果文件来**，不手抄。

`packages/semprea-anchor/MODEL_CARD.md` 里有一个 `<!--EVAL-->` 占位，
本脚本读 `sweep_results/` 下的结果，生成表格替换它，写到 `--out`（默认同名文件原地替换）。

缺哪一份就写"未测"，**不猜、不填默认值**。
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

ROOT = Path("sweep_results")
ANCHOR = ROOT / "eval_anchor"
REAL = ROOT / "eval_real_dialog"
STATE = ROOT / "state_qa" / "result.json"
MEMO = ROOT / "kb_sources" / "memorization.json"
CARD = Path("packages/semprea-anchor/MODEL_CARD.md")

SEED_NAMES = ("s0", "s1", "s2")


def load(p: Path) -> dict | None:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _f(v: float, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def section_criteria() -> list[str]:
    """五个种子 + 底座的五项判据。"""
    lines = [
        "### 5.1 五项判据（三个种子 + 底座对照）",
        "",
        "| 判据 | " + " | ".join(SEED_NAMES) + " | 均值 | 区间 | 底座 |",
        "| :--- | " + " | ".join("---:" for _ in SEED_NAMES) + " | ---: | ---: | ---: |",
    ]
    runs = {n: load(ANCHOR / f"{n}.json") for n in SEED_NAMES}
    base = load(ANCHOR / "base.json")
    if base is None and any(runs.values()):
        first = next(r for r in runs.values() if r)
        base = first.get("base")

    def row(label: str, get) -> str:
        vals = [get(runs[n]) if runs[n] else None for n in SEED_NAMES]
        good = [v for v in vals if v is not None]
        if not good:
            return f"| {label} | 未测 | 未测 | 未测 | 未测 | 未测 |"
        mean = sum(good) / len(good)
        rng = f"{min(good):.2f}–{max(good):.2f}" if len(good) > 1 else "单次"
        b = get(base) if base else None
        cells = " | ".join("未测" if v is None else f"{v:.2f}" for v in vals)
        return f"| {label} | {cells} | {mean:.2f} | {rng} | {_f(b)} |"

    lines.append(row("I-1 身份（/8）", lambda r: r["identity"]["hit"] / r["identity"]["n"] * 8))
    lines.append(row("I-2 边界（/2）", lambda r: r["boundary"]["hit"] / r["boundary"]["n"] * 2))
    lines.append(row("D-7 危机 sens", lambda r: r["crisis"]["sens"]))
    lines.append(row("D-2 客观（/6）", lambda r: r["objective"]["hit"] / r["objective"]["n"] * 6))
    lines.append(row("D-2 开放式退化（/4）", lambda r: r["open"]["degenerate"]))
    lines.append("")
    lines.append(
        "> **危机这一档报的是 sens 与 FPR 两个数，不是「3/3 全过」。** "
        "上一版只有 3 条危机探针，95% 置信下界约 0.29，算不出 0.95。"
    )
    lines.append("")
    return lines


def section_crisis() -> list[str]:
    runs = {n: load(ANCHOR / f"{n}.json") for n in SEED_NAMES}
    live = [r for r in runs.values() if r]
    if not live:
        return ["### 5.2 危机分的分档 FPR", "", "未测。", ""]
    kinds = ("distress", "hard_neg", "neutral")
    lines = [
        "### 5.2 危机门：不同档位的误报",
        "",
        "| 档 | 样例性质 | " + " | ".join(SEED_NAMES) + " |",
        "| :--- | :--- | " + " | ".join("---:" for _ in SEED_NAMES) + " |",
    ]
    labels = {
        "distress": "一般痛苦（该安慰，不该升级）",
        "hard_neg": "共用「死/崩溃/撑不住」但无自伤",
        "neutral": "日常琐事",
    }
    for k in kinds:
        cells = []
        for n in SEED_NAMES:
            r = runs[n]
            cells.append("未测" if not r else f"{r['crisis']['by_kind'][k]['fpr']:.2f}")
        lines.append(f"| {k} | {labels[k]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def section_memo() -> list[str]:
    m = load(MEMO)
    lines = ["### 5.3 记忆化探针：模型会不会背出维基原文", ""]
    if not m:
        return [*lines, "未测。", ""]
    verdict = "✅ 通过" if m["passed"] else "❌ 未通过"
    lines.append(
        f"对 88 个条目各问一次，回答与维基首段的**最长公共子串 {m['max_lcs']} 字**"
        f"（中位 {m['median_lcs']}）。判据 M-1 ≤{m['criteria']['M-1_max_lcs']} 字 ⇒ **{verdict}**。"
    )
    lines.append("")
    return lines


def section_real() -> list[str]:
    files = sorted(REAL.glob("*.json")) if REAL.is_dir() else []
    lines = ["### 5.4 真实对话（本机会话存档，原文不出本机）", ""]
    if not files:
        return [*lines, "未测。", ""]
    d = load(files[0])
    lines += [
        f"输入：{d['n']} 条**真实用户发言**（带前文历史），来自 `{d['sessions']}`。"
        "只记录聚合数字与 sha256，**原文不落进仓库**。",
        "",
        "| 臂 | 退化 | 拒答 | 危机升级 | 去重比 | 中位长度 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("naked", "persona"):
        a = d.get(arm)
        if not a:
            continue
        lines.append(
            f"| {arm} | {a['degenerate_rate']:.2f} | {a['refusal_rate']:.2f} | "
            f"{a['crisis_esc_rate']:.2f} | {a['distinct_2']:.2f} | {a['len_med']} |"
        )
    lines.append("")
    return lines


def section_state() -> list[str]:
    s = load(STATE)
    lines = ["### 5.5 状态是否真的影响输出", ""]
    if not s:
        return [*lines, "未测。", ""]
    lines += [
        "同一批问题、不同状态，答案是否跟着状态变（关键事实命中率）：",
        "",
        "| 臂 | 命中 |",
        "| :--- | ---: |",
    ]
    for k in ("none", "text", "film", "both"):
        if k in s:
            lines.append(f"| {k} | {s[k]:.2f} |")
    lines.append("")
    return lines


def section_second() -> list[str]:
    d = load(ANCHOR / "qwen3_0.6b.json")
    lines = ["### 5.6 换底座复验", ""]
    if not d:
        return [*lines, "未测。", ""]
    lines += [
        "同一份数据配方、同一个种子，换 **Qwen3-0.6B**（不同代、不同分词器、不同架构）重跑：",
        "",
        "| 判据 | Qwen2.5-0.5B（s0） | Qwen3-0.6B |",
        "| :--- | ---: | ---: |",
    ]
    s0 = load(ANCHOR / "s0.json")
    pairs = [
        ("I-1 身份（/8）", lambda r: r["identity"]["hit"]),
        ("I-2 边界（/2）", lambda r: r["boundary"]["hit"]),
        ("D-7 危机 sens", lambda r: r["crisis"]["sens"]),
        ("D-2 客观（/6）", lambda r: r["objective"]["hit"]),
        ("D-2 开放式退化（/4）", lambda r: r["open"]["degenerate"]),
    ]
    for label, f in pairs:
        a = f"{f(s0):.2f}" if s0 else "未测"
        b = f"{f(d):.2f}"
        lines.append(f"| {label} | {a} | {b} |")
    lines.append("")
    lines.append(
        "> **限定**：Qwen3-0.6B 仍是 Qwen 家族。这测的是"
        "「配方能不能换一代底座」，**不是跨家族通用性**。"
    )
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=Path, default=CARD)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    body = args.card.read_text(encoding="utf-8")
    block = []
    for fn in (
        section_criteria,
        section_crisis,
        section_memo,
        section_real,
        section_state,
        section_second,
    ):
        block += fn()
    body = body.replace("<!--EVAL-->", "\n".join(block).rstrip())
    out = args.out or args.card
    out.write_text(body, encoding="utf-8", newline="\n")
    print(f"模型卡已更新 → {out}（第 5 节 {len(block)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
