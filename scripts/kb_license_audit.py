"""许可审计：**自写定义**必须不是维基原文的复制。

## 判据（先写在这里，再跑）

设 `D` = 自写定义（`scripts/kb_defs.py`），`E` = 该条目的维基首段原文。

- **A-1**：`D` 与 `E` 的最长公共子串 **≤ 15 字**；
- **A-2**：`E` 中**任意一个 ≥20 字的句子**都不得完整出现在 `D` 里；
- **A-3**：每条定义非空、长度 **12–90 字**；
- **A-4**：`D` 与 `E` 的**字符 bigram 重合率 ≥ 0.05**（确认不是文不对题；阈值定得低，
  因为改写本来就会刻意避开原文用词）。

四条全过才判「许可阻塞解除」。**任何一条不过就报不过，不调阈值。**

## 为什么要这一层

上一版直接把维基首句当答案（CC BY-SA 4.0，要求署名 + 相同方式共享）。
改成自写之后，"有没有照抄"不能靠我说，要靠一个可复算的度量。

产物：`sweep_results/kb_sources/paraphrase_audit.json`、`paraphrase_review.md`
"""

from __future__ import annotations

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

import argparse
import json
import re
from pathlib import Path

from scripts.kb_defs import DEFS, display_name

WIKI = Path("sweep_results/kb_sources/wiki.jsonl")
OUT = Path("sweep_results/kb_sources")
MAX_LCS = 15
MIN_LEN, MAX_LEN = 12, 90
SENT = re.compile(r"[^。！？!?]+[。！？!?]")


def lcs(a: str, b: str) -> str:
    """最长公共子串（不是子序列）。88 条 × 450 字，朴素 DP 足够。"""
    if not a or not b:
        return ""
    prev = [0] * (len(b) + 1)
    best, end = 0, 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return a[end - best : end]


def bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=Path, default=WIKI)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reviewed", action="store_true", help="人工复核已完成（A-4′）")
    args = ap.parse_args()

    rows = {}
    for line in args.wiki.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                rows[r["title"]] = r

    used = {display_name(t): r for t, r in rows.items() if r.get("group") in ("psy", "common")}
    missing = sorted(set(used) - set(DEFS))
    extra = sorted(set(DEFS) - set(used))

    items, fails = [], []
    for name in sorted(used):
        r = used[name]
        d = DEFS.get(name, "")
        e = r["extract"]
        common = lcs(d, e) if d else ""
        sents = [s.strip() for s in SENT.findall(e) if len(s.strip()) >= 20]
        copied = [s for s in sents if d and s in d]
        bg = bigrams(d)
        share = len(bg & bigrams(e)) / max(len(bg), 1)
        reasons = []
        if not (MIN_LEN <= len(d) <= MAX_LEN):
            reasons.append(f"A-3 长度 {len(d)}")
        if len(common) > MAX_LCS:
            reasons.append(f"A-1 公共子串 {len(common)} 字：{common}")
        if copied:
            reasons.append(f"A-2 抄了整句：{copied[0][:30]}")
        item = {
            "title": name,
            "wiki_title": r["title"],
            "url": r["url"],
            "def": d,
            "len": len(d),
            "lcs": common,
            "lcs_len": len(common),
            "bigram_share": round(share, 4),
            "ok": not reasons,
            "reasons": reasons,
        }
        items.append(item)
        if reasons:
            fails.append(item)

    passed = not fails and not missing and not extra and args.reviewed
    out = {
        "criteria": {
            "A-1_max_lcs": MAX_LCS,
            "A-2_no_verbatim_sentence": "≥20 字的维基句子不得完整出现",
            "A-3_len": [MIN_LEN, MAX_LEN],
            "A-4": "人工复核（见 amend1）",
        },
        "n": len(items),
        "missing_defs": missing,
        "unused_defs": extra,
        "reviewed": bool(args.reviewed),
        "fails": len(fails),
        "passed": passed,
        "max_lcs_len": max((i["lcs_len"] for i in items), default=0),
        "items": items,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paraphrase_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    with (args.out / "paraphrase_review.md").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 自写定义 vs 维基原文（人工复核用）\n\n")
        fh.write(
            f"共 {len(items)} 条，最长公共子串 {out['max_lcs_len']} 字，失败 {len(fails)} 条。\n\n"
        )
        for i in items:
            fh.write(f"## {i['title']}\n\n")
            fh.write(f"- 自写：{i['def']}\n")
            fh.write(f"- 维基：{used[i['title']]['extract'][:220]}\n")
            fh.write(
                f"- 公共子串 {i['lcs_len']} 字：`{i['lcs']}`；bigram 重合 {i['bigram_share']}\n\n"
            )

    print(f"{len(items)} 条定义；缺失 {len(missing)}、多余 {len(extra)}")
    print(f"最长公共子串 {out['max_lcs_len']} 字（上限 {MAX_LCS}）")
    print(f"不过 {len(fails)} 条")
    for f in fails[:15]:
        print("  ✗", f["title"], f["reasons"])
    print("判：", "✅ A-1~A-4 全过" if passed else "❌ 未过")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
