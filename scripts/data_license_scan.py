"""扫**训练数据本身**：有没有维基原文混进去。

`kb_license_audit.py` 查的是定义文件（`kb_defs.py`）和原文像不像；
本脚本查的是**实际喂进模型的那 3 万条**——这是许可问题的真正落点。
定义文件干净、但别的族把原文带进来，前一个审计是看不出来的。

## 判据（先写死）

- **S-1**：训练答案里**不得出现**维基原文中 ≥20 字的完整句子，命中数必须为 **0**；
- **S-2**：训练答案与维基句子的**最长公共子串 ≤ 11 字**。
  实现上用 **12-gram 重合**做判定：只要没有任何 12 字连续片段重合，LCS 必然 ≤11。
  这比逐条算 LCS 快得多，而且是个**上界**（更严）。

产物：`sweep_results/kb_sources/data_license_scan.json`
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import re
from pathlib import Path

WIKI = Path("sweep_results/kb_sources/wiki.jsonl")
DATA = Path("sweep_results/persona_train/data.jsonl")
OUT = Path("sweep_results/kb_sources/data_license_scan.json")
SENT = re.compile(r"[^。！？!?]+[。！？!?]")
MIN_SENT = 20
NGRAM = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=Path, default=WIKI)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    sents: list[tuple[str, str]] = []
    for line in args.wiki.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ok"):
            continue
        for s in SENT.findall(r["extract"]):
            s = " ".join(s.split())
            if len(s) >= MIN_SENT:
                sents.append((r["title"], s))
    grams: set[str] = set()
    for _, s in sents:
        for i in range(len(s) - NGRAM + 1):
            grams.add(s[i : i + NGRAM])

    with open(args.data, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    whole_hits, gram_hits = [], []
    for r in rows:
        a = r["messages"][1]["content"].replace(" ", "")
        for t, s in sents:
            if s.replace(" ", "") in a:
                whole_hits.append({"title": t, "family": r["family"], "sent": s[:40]})
                break
        for i in range(len(a) - NGRAM + 1):
            g = a[i : i + NGRAM]
            if g in grams:
                gram_hits.append({"family": r["family"], "gram": g})
                break

    ok = not whole_hits and not gram_hits
    out = {
        "criteria": {"S-1_verbatim_sentence": 0, "S-2_max_lcs": NGRAM - 1},
        "wiki_sentences": len(sents),
        "samples": len(rows),
        "whole_sentence_hits": len(whole_hits),
        "ngram_hits": len(gram_hits),
        "examples": (whole_hits + gram_hits)[:5],
        "passed": ok,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"维基 ≥{MIN_SENT} 字句子 {len(sents)} 条；训练样本 {len(rows)} 条")
    print(f"整句命中 {len(whole_hits)}；{NGRAM}-gram 命中 {len(gram_hits)}")
    for e in out["examples"]:
        print("  ✗", e)
    print("判：", "✅ S-1/S-2 全过" if ok else "❌ 训练数据里有维基原文")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
