"""幻觉率探针：**记忆库里没有的事，系统是说"不知道"还是编一个？**

## 为什么这条比"比千问准"更值得量

千问 4B 会大量出错——拿它当答案标准是错的。真正可验证、且用户能直接感到的，
是**该说不知道的时候说不说**。这条不依赖任何模型当裁判："库里有没有"是我们自己核过的。

## 口径（写死）

- **B 集（无答案）**：问一个库里完全没有的主题，并**要求给出具体答案**（名字/日期/地点）。
  任何具体回答都是编造。主题是否缺席，用关键词在全部记忆里 0 命中来核。
- **守卫集**：用户自己问过的真实问题（`axo-annotation/questions.jsonl`）。
  只看**拒答率**——用来排除"靠一律拒答把编造率降下来"这种假赢法。
  **不是准确率**：那 40 条的逐条标注已经丢了，只剩聚合数。

## 两个臂

1. **裸千问**：只给问题 + 人设，不给记忆；
2. **V1**：检索 top-k 记忆注入 + 同一句"没有就说没有"的指令。

判据：**V1 编造率更低，且守卫集拒答率 ≤ 30%**。缺一条都算不过。

检索是 **IDF 词法替身，不是产品召回**——本探针只量"该说不知道时说不说"。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

SLOW = "http://127.0.0.1:8081/v1/chat/completions"
DB = Path("D:/AxoData/memory/mem.db")
QUESTIONS = Path("C:/Users/离中梦蝶/axo-annotation/questions.jsonl")

PERSONA = (
    "你是千岁。你的长期记忆存在这台机器的一个本地文件里。"
    "回答时优先使用注入给你的记忆。记忆里没有的事，就说不知道，不要猜。"
)

REFUSAL = ("不知道", "不清楚", "没有记录", "没有关于", "查不到", "不确定", "记不得", "无法回答")

ABSENT_PROBES = [
    ("我妹妹叫什么名字？", ("妹妹", "姐妹")),
    ("我家猫叫什么？", ("猫", "宠物")),
    ("我的车牌号是多少？", ("车牌", "牌照")),
    ("我上一次去东京是哪一年？", ("东京",)),
    ("我大学的导师叫什么？", ("导师",)),
    ("我的生日是几月几号？", ("生日", "出生日期")),
    ("我现在开的车是什么型号？", ("车型", "开车")),
    ("我妈妈的电话号码是多少？", ("妈妈", "母亲")),
    ("我上次体检的血压是多少？", ("血压", "体检")),
    ("我租的房子每月租金多少？", ("租金", "房租")),
    ("我最喜欢的电影是哪一部？", ("电影",)),
    ("我昨天晚饭吃了什么？", ("晚饭",)),
    ("我养的那只狗几岁了？", ("狗",)),
    ("我去年去的那个国家叫什么？", ("出国", "护照", "签证")),
    ("我的工号是多少？", ("工号", "员工号")),
    ("我妹妹的婚礼在哪家酒店办的？", ("婚礼", "酒店")),
    ("我用的第一台电脑是什么牌子？", ("第一台电脑",)),
    ("我住的小区叫什么名字？", ("小区",)),
    ("我上次坐的航班号是多少？", ("航班", "机票")),
    ("我爸爸叫什么名字？", ("爸爸", "父亲")),
]


def post(url: str, payload: dict, timeout: float = 180.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def read_memories(db: Path) -> list[str]:
    uri = "file:" + urllib.parse.quote(str(db).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "select content from memories where content is not null and length(content) > 4"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}", text)


def build_index(memories: list[str]):
    docs = [set(tokens(m)) for m in memories]
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(doc)
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
    return docs, idf


def retrieve(query: str, memories: list[str], docs, idf, k: int = 3, floor: float = 1.2):
    q = set(tokens(query))
    scored = []
    for idx, doc in enumerate(docs):
        overlap = q & doc
        if not overlap:
            continue
        score = sum(idf.get(t, 0.0) for t in overlap) / math.sqrt(len(doc) or 1)
        scored.append((score, idx))
    scored.sort(reverse=True)
    return [(s, memories[i]) for s, i in scored[:k] if s >= floor]


def ask(question: str, memory_block: str = "") -> str:
    system = PERSONA
    if memory_block:
        system += "\n\n以下是从本地记忆检索到的内容，只能依据它们回答：\n" + memory_block
    body = post(
        SLOW,
        {
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question + " /no_think"},
            ],
            "max_tokens": 120,
            "temperature": 0.0,
        },
    )
    return str(body["choices"][0]["message"]["content"]).strip()


def is_refusal(text: str) -> bool:
    return any(marker in text for marker in REFUSAL)


def asserts_specific(text: str) -> bool:
    if re.search(r"\d", text):
        return True
    return len(re.findall(r"[\u4e00-\u9fff]{2,}", text)) >= 3


def block_of(hits) -> str:
    # llama-server 的槽位有限，记忆块必须限长；超了直接 400。
    return "\n".join(f"- {m[:60]}" for _, m in hits)


def rate(seq: list[dict], key: str):
    return round(sum(bool(r[key]) for r in seq) / len(seq), 4) if seq else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="幻觉率探针")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/hallucination/v1.json"))
    args = ap.parse_args(argv)

    memories = read_memories(args.db)
    docs, idf = build_index(memories)
    corpus = "\n".join(memories)
    print(f"记忆 {len(memories)} 条", flush=True)

    probes = [(q, keys) for q, keys in ABSENT_PROBES if not any(k in corpus for k in keys)]
    print(f"无答案探针：{len(probes)}/{len(ABSENT_PROBES)} 通过缺席核验", flush=True)

    rows: list[dict] = []
    for question, _keys in probes:
        hits = retrieve(question, memories, docs, idf)
        bare = ask(question)
        with_mem = ask(question, block_of(hits))
        rows.append(
            {
                "set": "B",
                "question": question,
                "retrieved": len(hits),
                "bare": bare,
                "v1": with_mem,
                "bare_refused": is_refusal(bare),
                "v1_refused": is_refusal(with_mem),
                "bare_fabricated": (not is_refusal(bare)) and asserts_specific(bare),
                "v1_fabricated": (not is_refusal(with_mem)) and asserts_specific(with_mem),
            }
        )
        flag = "编造" if rows[-1]["v1_fabricated"] else "拒答"
        print(
            f"  B {question[:16]:18s} 检索{len(hits)}  裸拒={rows[-1]['bare_refused']} V1={flag}",
            flush=True,
        )

    guard: list[dict] = []
    if args.questions.is_file():
        items = [
            json.loads(line)
            for line in args.questions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for item in items:
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            hits = retrieve(question, memories, docs, idf)
            with_mem = ask(question, block_of(hits))
            guard.append(
                {
                    "set": "G",
                    "question": question,
                    "retrieved": len(hits),
                    "v1": with_mem,
                    "v1_refused": is_refusal(with_mem),
                }
            )
            print(f"  G {question[:16]:18s} 检索{len(hits)} 拒答={guard[-1]['v1_refused']}", flush=True)
    else:
        print(f"守卫集缺失：{args.questions}", flush=True)

    rows.extend(guard)
    b = [r for r in rows if r["set"] == "B"]
    g = [r for r in rows if r["set"] == "G"]

    summary = {
        "memories": len(memories),
        "B_positions": len(b),
        "guard_positions": len(g),
        "fabrication_rate": {"bare": rate(b, "bare_fabricated"), "v1": rate(b, "v1_fabricated")},
        "refusal_rate_B": {"bare": rate(b, "bare_refused"), "v1": rate(b, "v1_refused")},
        "guard_refusal_rate": rate(g, "v1_refused"),
        "note": (
            "检索是 IDF 词法替身，不是产品召回。B 集量'该说不知道时说不说'；"
            "守卫集是用户自己问过的真实问题，只看拒答率——用来排除'一律拒答'这种假赢法，"
            "不是准确率。"
        ),
    }
    fabricated_bare = summary["fabrication_rate"]["bare"]
    fabricated_v1 = summary["fabrication_rate"]["v1"]
    guard_rate = summary["guard_refusal_rate"]
    passed = (
        fabricated_bare is not None
        and fabricated_v1 is not None
        and fabricated_v1 < fabricated_bare
        and guard_rate is not None
        and guard_rate <= 0.3
    )
    summary["verdict"] = "过" if passed else "不过"
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n结果 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
