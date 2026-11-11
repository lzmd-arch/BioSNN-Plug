"""用千问给召回基准的每条记忆生成改述，建一个**词法必输**的查询集。

为什么需要它：现网 85 条查询里，`paraphrase` 只有 **n=5**，而词法在其余各类上近乎满分
（name_number 30/30）。**在这些类上比语义编码是量不出东西的。** 要回答"SNN 的特征有没有
语义"，只能在一个"换了说法、词面几乎不重叠"的集合上比。

**千问在这里的角色是标注器，不是教师**：出改述，不出 token。

产出落仓外，只含合成基准的记忆（不含任何用户数据）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

GOLD = Path("D:/Projects/BioSNN-MemoryTerminal/benchmarks/gold")
OUT = Path("D:/AxoData/sel/lm/paraphrase")
HOST = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3-4b"

PROMPT = """把下面这句话换一种说法重写，要求：
1. 事实、数字、专有名词**完全不变**；
2. 词面和句式尽量不同（同义词替换、主动被动互换、语序调整）；
3. 只输出改写后的那一句，不要解释、不要加引号。

原句：{text}"""


def call(text: str, temperature: float, timeout: int = 120) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT.format(text=text),
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": 120},
        }
    ).encode()
    request = urllib.request.Request(
        HOST, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read()).get("response", "").strip()


def clean(text: str) -> str:
    text = re.sub(r"^[\s\"'“”「」]+|[\s\"'“”「」]+$", "", text)
    return text.split("\n")[0].strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="千问生成改述，建词法必输的召回查询集")
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-memory", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    memories = [
        json.loads(x) for x in (args.gold / "memories.jsonl").read_text("utf-8").splitlines()
    ]
    if args.limit:
        memories = memories[: args.limit]
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "queries.jsonl"
    kept = failed = 0
    started = time.time()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, memory in enumerate(memories, 1):
            for variant in range(args.per_memory):
                try:
                    text = clean(call(memory["content"], 0.8 + 0.15 * variant))
                except Exception as exc:
                    failed += 1
                    print(f"  失败 {memory['id']}#{variant}: {type(exc).__name__}", flush=True)
                    continue
                if len(text) < 6 or text == memory["content"]:
                    failed += 1
                    continue
                # never 里放同批的其它记忆，让"抓错条目"能被计进去
                others = [m["id"] for m in rng.sample(memories, min(4, len(memories) - 1))]
                row = {
                    "id": f"pp-{memory['id']}-{variant}",
                    "query": text,
                    "category": "paraphrase_gen",
                    "expect": [memory["id"]],
                    "never": [o for o in others if o != memory["id"]],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            if index % 20 == 0:
                print(
                    f"  {index}/{len(memories)}，已生成 {kept} 条，"
                    f"用时 {(time.time() - started) / 60:.1f} 分",
                    flush=True,
                )
    print(f"写入 {path}：{kept} 条（失败 {failed}）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
