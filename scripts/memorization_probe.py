"""记忆化探针：**模型还能不能逐字吐出维基原文**。

## 为什么需要这一条

许可问题不只是"训练目标有没有抄"，还有"模型会不会把抄的东西背出来"。
训练数据换成自写定义之后，这一条应该**从"可能"变成"不可能"，但要用数说话**。

## 怎么测

对 88 个条目各问一次"什么是 X？"，把回答和该条目的维基首段做**最长公共子串**。

- 若模型是在复述训练目标（自写定义），LCS 应该和那句定义的量级相当（十几字以内）；
- 若模型背下了维基原文，LCS 会在几十字以上——那才是需要处置的许可风险。

## 判据（先写死）

- **M-1**：88 条里**最长公共子串 ≤ 20 字**；
- **M-2**：**没有任何一条**的 LCS ≥ 30 字。

任何一条不过 ⇒ 模型在逐字复现受版权保护的文本，**不可发布**。
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.chatfmt import render
from scripts.kb_defs import DEFS, display_name
from scripts.kb_license_audit import lcs

WIKI = Path("sweep_results/kb_sources/wiki.jsonl")
OUT = Path("sweep_results/kb_sources/memorization.json")
M1, M2 = 20, 30


@torch.no_grad()
def ask(model, tok, q: str, dev: str, max_new: int = 96) -> str:
    text = render(tok, [{"role": "user", "content": q}])
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = {}
    for line in WIKI.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                rows[r["title"]] = r
    used = {display_name(t): r for t, r in rows.items() if r.get("group") in ("psy", "common")}

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).to(dev).eval()

    items = []
    for name in sorted(used):
        a = ask(model, tok, f"什么是{name}？", dev)
        common = lcs(a, used[name]["extract"])
        items.append(
            {
                "title": name,
                "answer": a[:120],
                "lcs_len": len(common),
                "lcs": common,
                "def_lcs": len(lcs(a, DEFS[name])),
            }
        )

    mx = max(i["lcs_len"] for i in items)
    worst = sorted(items, key=lambda i: -i["lcs_len"])[:5]
    out = {
        "ckpt": args.ckpt,
        "criteria": {"M-1_max_lcs": M1, "M-2_any_lcs": M2},
        "n": len(items),
        "max_lcs": mx,
        "median_lcs": statistics.median(i["lcs_len"] for i in items),
        "passed": mx <= M1 and mx < M2,
        "worst": [{"title": w["title"], "lcs_len": w["lcs_len"], "lcs": w["lcs"]} for w in worst],
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(items)} 条；最长公共子串 {mx} 字（中位 {out['median_lcs']}）")
    print("最长的五条：")
    for w in worst:
        print(f"  {w['title']:12s} {w['lcs_len']:2d} 字：{w['lcs']}")
    print("判：", "✅ M-1/M-2 全过" if out["passed"] else "❌ 逐字复现了维基原文")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
