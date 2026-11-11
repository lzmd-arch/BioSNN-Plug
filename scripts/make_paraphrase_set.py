"""给召回基准的每条记忆生成改述，建一个**词法必输**的查询集。

为什么需要它：现网 85 条查询里，`paraphrase` 只有 **n=5**，而词法在其余各类上近乎满分
（name_number 30/30）。**在这些类上比语义编码是量不出东西的。**

两个提供方：

- `--provider ollama`（默认，本机 `qwen3-4b`）——不出本机，但会与训练抢显存；
- `--provider deepseek`（云端）——快得多。**只发送合成基准的记忆**
  （"北岸一号站的采集箱…"），不含任何用户数据。

在这条链里，大模型的角色是**标注器**，不是教师：出改述，不出训练 token。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GOLD = Path("D:/Projects/BioSNN-MemoryTerminal/benchmarks/gold")
OUT = Path("D:/AxoData/sel/lm/paraphrase")
OLLAMA_HOST = "http://127.0.0.1:11434/api/generate"
DEEPSEEK_HOST = "https://api.deepseek.com/chat/completions"

PROMPT = """把下面这句话换一种说法重写，要求：
1. 事实、数字、专有名词完全不变；
2. 词面和句式尽量不同（同义词替换、主动被动互换、语序调整）；
3. 只输出改写后的那一句，不要解释、不要加引号。

原句：{text}"""


def call(
    text: str,
    temperature: float,
    *,
    provider: str,
    model: str,
    key: str = "",
    timeout: int = 120,
) -> str:
    prompt = PROMPT.format(text=text)
    if provider == "deepseek":
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 160,
                "stream": False,
            }
        ).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        request = urllib.request.Request(DEEPSEEK_HOST, data=payload, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            body = json.loads(handle.read())
        return body["choices"][0]["message"]["content"].strip()

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": 120},
        }
    ).encode()
    request = urllib.request.Request(
        OLLAMA_HOST, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read()).get("response", "").strip()


def clean(text: str) -> str:
    text = re.sub(r"^[\s\"'“”「」]+|[\s\"'“”「」]+$", "", text)
    return text.split("\n")[0].strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="大模型生成改述，建词法必输的召回查询集")
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-memory", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default="ollama", choices=("ollama", "deepseek"))
    ap.add_argument("--model", default="", help="留空则按 provider 取默认")
    ap.add_argument("--api-key", default="", help="不给则读环境变量 DEEPSEEK_API_KEY")
    ap.add_argument("--workers", type=int, default=6, help="并发请求数（云端才用得上）")
    args = ap.parse_args(argv)

    key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if args.provider == "deepseek" and not key:
        raise SystemExit("deepseek 需要 API key：--api-key 或环境变量 DEEPSEEK_API_KEY")
    model = args.model or ("deepseek-chat" if args.provider == "deepseek" else "qwen3-4b")

    memories = [
        json.loads(x) for x in (args.gold / "memories.jsonl").read_text("utf-8").splitlines()
    ]
    if args.limit:
        memories = memories[: args.limit]
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "queries.jsonl"
    jobs = [(memory, variant) for memory in memories for variant in range(args.per_memory)]

    def make(job) -> tuple[dict, int, str | None, str]:
        memory, variant = job
        try:
            text = clean(
                call(
                    memory["content"],
                    0.8 + 0.15 * variant,
                    provider=args.provider,
                    model=model,
                    key=key,
                )
            )
        except Exception as exc:
            return memory, variant, None, type(exc).__name__
        if len(text) < 6 or text == memory["content"]:
            return memory, variant, None, "too-short-or-identical"
        return memory, variant, text, ""

    def rows() -> Iterator[tuple[dict, int, str | None, str]]:
        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            yield from pool.map(make, jobs)

    kept = failed = 0
    started = time.time()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (memory, variant, text, reason) in enumerate(rows(), 1):
            if text is None:
                failed += 1
                if failed <= 5:
                    print(f"  跳过 {memory['id']}#{variant}: {reason}", flush=True)
            else:
                others = [m["id"] for m in rng.sample(memories, min(4, len(memories) - 1))]
                record = {
                    "id": f"pp-{memory['id']}-{variant}",
                    "query": text,
                    "category": "paraphrase_gen",
                    "expect": [memory["id"]],
                    "never": [o for o in others if o != memory["id"]],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            if index % 50 == 0:
                rate = index / max(time.time() - started, 1)
                print(f"  {index}/{len(jobs)}，已生成 {kept} 条，{rate * 60:.0f} 条/分", flush=True)
    print(f"写入 {path}：{kept} 条（失败 {failed}），{(time.time() - started) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
