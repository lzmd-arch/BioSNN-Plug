"""Wikidata **多属性**抽取：把每个概念从「一句话」变成「三四个侧面」。

## 为什么需要它

`kb_wikidata.py` 只取了 `P31`（属于什么）和中文描述——**每个概念平均 1.5 条事实**。
用户 2026-11-16 的要求是"知识面广**且够深**"。广度靠概念数，深度只能靠**每个概念多几个侧面**。

## 取哪些属性

只取**能问成一句自然中文、且答案是一个实体**的属性：

| 属性 | 问法 | 例 |
| :--- | :--- | :--- |
| `P279` 上级类 | `X 是什么的一种？` | 猫 → 猫科动物 |
| `P361` 属于 | `X 是哪一部分？` | 心脏 → 循环系统 |
| `P17` 国家 | `X 位于哪个国家？` | 富士山 → 日本 |
| `P571` 成立时间 | `X 是什么时候建立的？` | 清华大学 → 1911 年 |
| `P106` 职业 | `X 的职业是什么？` | 鲁迅 → 作家 |
| `P19` 出生地 | `X 出生在哪里？` | 李白 → 碎叶 |
| `P50` 作者 | `X 的作者是谁？` | 呐喊 → 鲁迅 |
| `P136` 类型 | `X 属于什么类型？` | 科幻小说 → 科幻 |
| `P495` 起源国 | `X 起源于哪里？` | 围棋 → 中国 |

## 许可

同 `kb_wikidata.py`：**Wikidata 是 CC0**，事实不受版权保护。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib as _pathlib
import sys as _sys
import time
import urllib.parse
import urllib.request

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

UA = {
    "User-Agent": "BioSNN-Plug/1.0 (https://github.com/lzmd-arch/BioSNN-Plug) "
    "python-urllib/3; research use"
}
WD_API = "https://www.wikidata.org/w/api.php"
IN = _pathlib.Path("sweep_results/kb_sources/wikidata.jsonl")
OUT = _pathlib.Path("sweep_results/kb_sources/wikidata_deep.jsonl")

#: 属性 → 问法模板（`{x}` 是概念名，`{v}` 是值的标签）
PROPS: dict[str, str] = {
    "P279": "{x}是什么的一种？",
    "P361": "{x}是哪一部分？",
    "P17": "{x}位于哪个国家？",
    "P571": "{x}是什么时候建立的？",
    "P106": "{x}的职业是什么？",
    "P19": "{x}出生在哪里？",
    "P50": "{x}的作者是谁？",
    "P136": "{x}属于什么类型？",
    "P495": "{x}起源于哪里？",
}
#: 答案模板（"值"是标签）
ANSWER = {
    "P279": "{x}是「{v}」的一种。",
    "P361": "{x}是「{v}」的一部分。",
    "P17": "{x}位于{v}。",
    "P571": "{x}建立于{v}。",
    "P106": "{x}的职业是{v}。",
    "P19": "{x}出生在{v}。",
    "P50": "{x}的作者是{v}。",
    "P136": "{x}属于「{v}」类型。",
    "P495": "{x}起源于{v}。",
}


def _get(url: str, tries: int = 4) -> dict | None:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"    重试 {i + 1}/{tries}：{type(e).__name__}", flush=True)
            time.sleep(2 + 2 * i)
    return None


def fetch_claims(qids: list[str], batch: int = 50) -> dict[str, dict[str, list[str]]]:
    """qid → {属性: [值 qid 或字面量]}。只取 PROPS 里那九个。"""
    out: dict[str, dict[str, list[str]]] = {}
    for i in range(0, len(qids), batch):
        q = {
            "action": "wbgetentities",
            "ids": "|".join(qids[i : i + batch]),
            "props": "claims",
            "format": "json",
        }
        got = _get(WD_API + "?" + urllib.parse.urlencode(q))
        if not got:
            continue
        for qid, e in got.get("entities", {}).items():
            per: dict[str, list[str]] = {}
            for prop in PROPS:
                vals: list[str] = []
                for c in e.get("claims", {}).get(prop, [])[:2]:
                    dv = c.get("mainsnak", {}).get("datavalue")
                    if not dv:
                        continue
                    v = dv.get("value")
                    if isinstance(v, dict):
                        if "id" in v:
                            vals.append(v["id"])
                        elif "time" in v:
                            vals.append(v["time"])
                    elif isinstance(v, str):
                        vals.append(v)
                if vals:
                    per[prop] = vals
            if per:
                out[qid] = per
        if (i // batch) % 20 == 0:
            print(f"  声明 {min(i + batch, len(qids))}/{len(qids)}", flush=True)
        time.sleep(0.8)
    return out


def label_of(qids: list[str], batch: int = 50) -> dict[str, str]:
    out: dict[str, str] = {}
    uniq = sorted(set(qids))
    for i in range(0, len(uniq), batch):
        q = {
            "action": "wbgetentities",
            "ids": "|".join(uniq[i : i + batch]),
            "props": "labels",
            "languages": "zh-hans|zh-cn|zh",
            "format": "json",
        }
        got = _get(WD_API + "?" + urllib.parse.urlencode(q))
        if not got:
            continue
        for qid, e in got.get("entities", {}).items():
            for lang in ("zh-hans", "zh-cn", "zh"):
                if lang in e.get("labels", {}):
                    out[qid] = e["labels"][lang]["value"]
                    break
        time.sleep(0.8)
    return out


def year_of(ts: str) -> str:
    """Wikidata 时间是 `+1911-00-00T00:00:00Z` 这种，取年份。"""
    with contextlib.suppress(ValueError, IndexError):
        y = ts.lstrip("+").split("-")[0]
        if y.isdigit():
            return f"{int(y)} 年"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=_pathlib.Path, default=IN)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()

    rows = []
    for line in args.inp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows = rows[: args.limit]
    qids = [r["qid"] for r in rows]
    name = {r["qid"]: r["label"] for r in rows}
    print(f"概念 {len(qids)}", flush=True)

    claims = fetch_claims(qids)
    # 收集所有需要翻译成中文标签的值
    need = [v for per in claims.values() for prop, vs in per.items() if prop != "P571" for v in vs]
    print(f"需要标签的值 {len(set(need))}", flush=True)
    lbl = label_of(need)

    out_rows = []
    for qid, per in claims.items():
        x = name.get(qid, "")
        if not x:
            continue
        for prop, vals in per.items():
            for v in vals:
                vv = year_of(v) if prop == "P571" else lbl.get(v, "")
                if not vv or vv == x or len(vv) > 24:
                    continue
                q = PROPS[prop].format(x=x)
                a = ANSWER[prop].format(x=x, v=vv)
                out_rows.append(
                    {
                        "qid": qid,
                        "label": x,
                        "prop": prop,
                        "question": q,
                        "answer": a,
                        "license": "CC0",
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_concept = len(out_rows) / max(len(claims), 1)
    print(f"共 {len(out_rows)} 条事实，覆盖 {len(claims)} 个概念，平均 {per_concept:.2f} 条/概念")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
