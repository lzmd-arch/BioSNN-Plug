"""把**有出处**的定义与**有依据**的情境拼成训练数据。

三条来源，各自标明：

| 部分 | 来源 | 标注 |
| :--- | :--- | :--- |
| 概念定义（心理/常识） | 中文维基条目首段（CC BY-SA 4.0） | 每条带 `source_url` |
| 伦理情境 | 人工撰写，**依据指向对应的公开条目** | 每条带 `source_url` |
| 学科题（算术/单位/物理） | 程序生成，答案由代码算 | `verified=True` |

**不做改写**：维基的定义原样进答案（截到一两句），因为改写就等于我加了一层不可核对的转述。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

WIKI = Path("sweep_results/kb_sources/wiki.jsonl")

#: 伦理情境 → 依据条目（条目名必须出现在抓取结果里，否则该条不带 source）
ETHICS_ANCHOR = {
    "医疗": "医学伦理学",
    "科研": "科研伦理",
    "商业": "商业伦理",
    "环境": "环境伦理",
    "网络": "人工智能伦理",
    "职场": "职业道德",
    "朋友": "关怀伦理学",
    "家庭": "儒家",
    "教育": "公正",
    "交通": "公正",
    "租赁": "公正",
    "志愿": "知情同意",
}

SENT = re.compile(r"^(.{10,120}?[。！？])")


def _first(text: str, limit: int = 130) -> str:
    text = " ".join(text.split())
    m = SENT.match(text)
    out = m.group(1) if m else text
    if len(out) > limit:
        cut = out.rfind("，", 0, limit)
        out = out[: cut if cut > 40 else limit] + "…"
    return out


def load_sources() -> dict[str, dict]:
    if not WIKI.is_file():
        return {}
    out = {}
    for line in WIKI.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r.get("ok"):
            out[r["title"]] = r
    return out


def build(n: int, seed: int = 0) -> tuple[list[dict], dict]:
    from scripts.persona_data import BOUNDARY, IDENTITY
    from scripts.persona_kb import (
        ETHICS_CONSENSUS,
        ETHICS_THEORY,
        _divergence,
        _math,
        _physics,
        _unit,
    )

    rng = random.Random(seed)
    src = load_sources()
    psy_titles = [t for t, r in src.items() if r.get("group") == "psy"]
    common_titles = [t for t, r in src.items() if r.get("group") == "common"]

    def defn(title: str) -> tuple[str, str]:
        form = rng.randrange(3)
        if form == 0:
            return f"{title}是什么？", title
        if form == 1:
            return f"解释一下{title}", title
        return f"什么是{title}？", title

    weights = [
        ("subject", 35),
        ("psy_def", 20),
        ("common_def", 10),
        ("ethics_case", 15),
        ("ethics_div", 5),
        ("ethics_theory", 5),
        ("identity", 5),
    ]
    pool = [f for f, w in weights for _ in range(w)]
    rows: list[dict] = []
    stats = {
        "psy_def": 0,
        "common_def": 0,
        "ethics_case": 0,
        "ethics_div": 0,
        "ethics_theory": 0,
        "subject": 0,
        "identity": 0,
        "sourced": 0,
    }

    for _ in range(n):
        fam = rng.choice(pool)
        rec: dict = {"family": fam, "verified": False, "source_url": "", "license": ""}
        if fam in ("psy_def", "common_def"):
            titles = psy_titles if fam == "psy_def" else common_titles
            if not titles:
                continue
            t = rng.choice(titles)
            q, _ = defn(t)
            a = _first(src[t]["extract"])
            rec.update(source_url=src[t]["url"], license=src[t]["license"])
            stats["sourced"] += 1
        elif fam == "subject":
            q, a = rng.choice([_math, _math, _unit, _physics])(rng)
            rec["verified"] = True
        elif fam == "ethics_case":
            dom, sit, verdict, basis = rng.choice(ETHICS_CONSENSUS)
            q, a = f"{sit}。该怎么办？", f"{verdict}。依据是{basis}。"
            anchor = ETHICS_ANCHOR.get(dom)
            if anchor in src:
                rec.update(source_url=src[anchor]["url"], license=src[anchor]["license"])
                stats["sourced"] += 1
        elif fam == "ethics_div":
            q, a = _divergence(rng)
            if "关怀伦理学" in src:
                rec.update(
                    source_url=src["关怀伦理学"]["url"], license=src["关怀伦理学"]["license"]
                )
                stats["sourced"] += 1
        elif fam == "ethics_theory":
            q, a = rng.choice(ETHICS_THEORY)
            for k in ("义务论", "功利主义", "德性伦理", "关怀伦理学", "权利", "儒家"):
                if k in q and k in src:
                    rec.update(source_url=src[k]["url"], license=src[k]["license"])
                    stats["sourced"] += 1
                    break
        else:
            q, a = rng.choice(IDENTITY + BOUNDARY)
        rec["messages"] = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        rows.append(rec)
        stats[fam] += 1
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/persona_train/data.jsonl"))
    args = ap.parse_args()
    rows, stats = build(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    uniq = {r["messages"][0]["content"] for r in rows}
    print(f"{len(rows)} 条 → {args.out}")
    print("族：", stats)
    print(f"带出处的样本 {stats['sourced']}（{stats['sourced'] / max(len(rows), 1):.0%}）")
    print(f"不同的问句 {len(uniq)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
