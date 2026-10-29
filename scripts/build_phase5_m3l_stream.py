"""阶段五 M3l：生成真实对话流 fixture（判据见 sweep_results/phase5_m3l_prereg.md）。

只做一件事：用本机 `qwen3-4b`（temperature=0、top_k=1、think=False）按固定剧本生成
5 段 × 20 轮的对话，每段安排 5 个「值得记住的事件轮」，落盘为 fixture。

⚠️ LLM 生成**不是逐位可复现**的：fixture 一旦落盘就是数据本体，后续所有种子读同一份；
脚本里记下模型名、参数、提示的 sha256 与产物的 sha256，重跑会生成**另一份**数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sweep_results" / "phase5_m3l"

#: 每段对话的目标轮（0 起）：5 个记忆事件。
TARGET_TURNS = (4, 8, 12, 16, 19)
TURNS_PER_CONVERSATION = 20

TOPICS = (
    "周末出游计划",
    "家常菜谱与采购",
    "工作项目排期",
    "健康作息与运动",
    "养猫与读书的日常",
)

#: 目标轮要说出的事实类型（只给类型，不给内容——内容由模型写，避免照抄模板）。
FACT_KINDS = (
    "一个亲近的人的名字与生日（月/日）",
    "一个明确的饮食偏好或忌口",
    "一个约定好的时间与地点",
    "一个常用的收货地址或小区名",
    "一个具体的长远计划与它的时间点",
)

PERSONA = (
    "你是一位中文母语者，正在和一个记性很好的助手闲聊。"
    "每次只写你自己这一轮要说的话，一到两句，口语，不要写旁白、不要写助手的话。"
)

CHITCHAT = (
    "本轮请围绕话题继续闲聊，只写你自己这一轮的话（一到两句）。不要说出任何需要长期记住的具体事实。"
)

EVENT = (
    "本轮请自然地说出一件值得对方长期记住的事：{kind}。"
    "要具体（给出名字/日期/地点/数字），一到两句，口语。"
)


def _opener():
    # 系统代理会把 loopback 送进代理，显式给空 ProxyHandler。
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def generate(prompt: str, *, model: str, seed: int, host: str, timeout: float = 180.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "top_k": 1, "seed": seed, "num_predict": 120},
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _opener()(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = str(body.get("response", ""))
    for marker in ("<｜end▁of▁sentence｜>",):
        text = text.replace(marker, "")
    return text.strip()


def build(*, model: str, host: str, conversations: int = 5) -> dict:
    records = []
    prompts = []
    for index in range(conversations):
        topic = TOPICS[index % len(TOPICS)]
        turns = []
        history: list[str] = []
        for turn in range(TURNS_PER_CONVERSATION):
            is_target = turn in TARGET_TURNS
            instruction = (
                EVENT.format(kind=FACT_KINDS[TARGET_TURNS.index(turn)]) if is_target else CHITCHAT
            )
            context = "\n".join(history[-5:]) if history else "（这是第一轮）"
            prompt = f"{PERSONA}\n\n话题：{topic}\n之前的对话：\n{context}\n\n{instruction}"
            prompts.append(prompt)
            text = generate(
                prompt,
                model=model,
                seed=1000 * (index + 1) + turn,
                host=host,
            )
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            utterance = lines[0] if lines else "（空回复）"
            turns.append({"turn": turn, "text": utterance, "target": is_target})
            history.append(f"你：{utterance}")
        records.append({"id": index, "topic": topic, "turns": turns})
    return {
        "schema_version": 1,
        "kind": "phase5-m3l dialogue stream fixture",
        "model": model,
        "options": {"temperature": 0, "top_k": 1, "think": False},
        "target_turns": list(TARGET_TURNS),
        "turns_per_conversation": TURNS_PER_CONVERSATION,
        "prompt_sha256": hashlib.sha256("\n\n".join(prompts).encode("utf-8")).hexdigest(),
        "conversations": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3l 对话流 fixture 生成")
    parser.add_argument("--model", default="qwen3-4b:latest")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--conversations", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT / "stream.json")
    args = parser.parse_args(argv)

    record = build(model=args.model, host=args.host, conversations=args.conversations)
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "prompt_sha256": record["prompt_sha256"],
                "conversations": len(record["conversations"]),
                "turns": sum(len(item["turns"]) for item in record["conversations"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
