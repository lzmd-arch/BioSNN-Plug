"""生成 R1 的三档合成语料（判据见 `prereg-lm-r1-synthetic-ladder.md`）。

- **教师**：本地 Ollama `qwen3-4b`，`think=false`（不关的话 token 全被思维链吃掉）。
- **风格参数**：只来自真实语料的**聚合统计**（每轮字数、句长分位），**不取任何正文**；
  具体的性格/话题是模板枚举，不是从真实对话抽的。
- **数据落盘在仓外**：`D:/AxoData/sel/lm/v1/synth/{level}.jsonl`，不入库。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path("D:/AxoData/sel/lm/v1/synth")
OLLAMA = "http://127.0.0.1:11434/api/generate"
TEACHER = "qwen3-4b"

#: 三档：只改"多样性旋钮"——话题数、性格数、温度、轮数。
LEVELS = {
    "E": {"topics": 4, "personas": 2, "temperature": 0.6, "turns": (3, 3), "twists": False},
    "M": {"topics": 12, "personas": 4, "temperature": 0.9, "turns": (3, 5), "twists": False},
    "H": {"topics": 30, "personas": 6, "temperature": 1.1, "turns": (3, 6), "twists": True},
}

PERSONAS = [
    "温柔但话少",
    "干脆利落，不爱寒暄",
    "活泼，爱接梗",
    "有点疲惫，句子短",
    "认真，会追问细节",
    "淡淡的，偶尔冷幽默",
]
TOPICS = [
    "今天有点累",
    "天气",
    "晚饭吃什么",
    "失眠",
    "工作上的烦心事",
    "周末计划",
    "养了只猫",
    "刚看完一部电影",
    "想去旅行",
    "在学做饭",
    "感冒了",
    "考试",
    "搬家",
    "找不到东西",
    "早上起不来",
    "想买件外套",
    "手机快没电了",
    "在等快递",
    "楼下在装修",
    "心情不错",
    "被同事气到",
    "在听歌",
    "跑步",
    "种的花开了",
    "在收拾房间",
    "想喝奶茶",
    "电脑很卡",
    "刚下班",
    "在排队",
    "看到一只狗",
]
FORMAT_RULE = "严格格式：每行以「用户：」或「千岁：」开头，不加旁白，不加解释。"


def build_prompt(topic: str, persona: str, turns: int, twist: bool) -> str:
    extra = "其中一轮要出现情绪转折或打断。" if twist else ""
    return (
        f"请写一段日常对话，双方是用户和 AI 伴侣“千岁”。\n"
        f"千岁的性格：{persona}。\n话题：{topic}。\n长度：{turns} 轮。\n"
        f"要求：口语化，用语气词，不书面，每句 5–25 字。{extra}\n{FORMAT_RULE}"
    )


def call(prompt: str, temperature: float, num_predict: int = 320) -> str:
    body = json.dumps(
        {
            "model": TEACHER,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": num_predict, "temperature": temperature},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["response"]
        except Exception:
            if attempt == 2:
                return ""
            time.sleep(1.0)
    return ""


def parse(text: str) -> list[dict]:
    """按「用户：」/「千岁：」切段，返回紧邻的 (user, assistant) 对。"""
    turns: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^(用户|千岁)[：:]\s*(.+)$", line)
        if match:
            turns.append((match.group(1), match.group(2).strip()))
    pairs = []
    for (speaker, content), (next_speaker, next_content) in pairwise(turns):
        if speaker == "用户" and next_speaker == "千岁" and content and next_content:
            pairs.append({"user": content, "assistant": next_content})
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 R1 的三档合成语料")
    parser.add_argument("--level", choices=sorted(LEVELS), required=True)
    parser.add_argument("--pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    config = LEVELS[args.level]
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.level}.jsonl"

    written = 0
    attempts = 0
    dropped = 0
    started = time.perf_counter()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        while written < args.pairs:
            attempts += 1
            topic = TOPICS[rng.randrange(config["topics"])]
            persona = PERSONAS[rng.randrange(config["personas"])]
            turns = rng.randint(*config["turns"])
            prompt = build_prompt(topic, persona, turns, config["twists"])
            raw = call(prompt, config["temperature"])
            pairs = parse(raw)
            if not pairs:
                dropped += 1
                continue
            group = written % 6  # 六折，写死按生成顺序轮转，保证均衡
            for pair in pairs:
                if written >= args.pairs:
                    break
                handle.write(
                    json.dumps(
                        {
                            "id": f"{args.level}-{written:05d}",
                            "group": group,
                            "level": args.level,
                            "topic": topic,
                            "persona": persona,
                            **pair,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            if written % 200 < len(pairs):
                rate = written / max(time.perf_counter() - started, 1e-6)
                print(f"  {written}/{args.pairs}  丢{dropped}   {rate:.1f} 对/s", flush=True)

    elapsed = time.perf_counter() - started
    print(f"[{args.level}] 写入 {written} 对，尝试 {attempts} 次，丢弃 {dropped}；{elapsed:.0f}s")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
