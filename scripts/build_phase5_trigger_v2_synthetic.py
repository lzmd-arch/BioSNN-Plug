"""造 T-4 的合成对照流（构造见 sweep_results/phase5_trigger_v2_prereg_amend1.md 第 6 节）。

20 段 × 20 轮：第 i 段的背景轮从话题 i 的句子池里均匀有放回抽，目标轮（固定 [4,8,12,16,19]）
从话题 (i+10) mod 20 的池里抽。段与段可交换、无时间趋势——这是给 conformal 构造做对照用的
"本应成立"的那条流。

生成是确定性的（种子写死），fixture 落盘后不再改；sha256 记进结果 README。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sweep_results" / "phase5_trigger_v2" / "synthetic_stream.json"

SEED = 20261102
TURNS = 20
TARGET_TURNS = [4, 8, 12, 16, 19]
BACKGROUND_TURNS = 15

#: 20 个话题，各配 3 个焦点词；索引 i 与 i+10 刻意选成语义不相邻。
TOPICS: list[tuple[str, list[str]]] = [
    ("医疗预约", ["挂号", "复诊", "检查报告"]),
    ("地铁通勤", ["早高峰", "换乘", "月票"]),
    ("城市天气", ["降温", "降雨", "空气质量"]),
    ("超市采购", ["生鲜", "折扣", "购物车"]),
    ("小区物业", ["停车位", "维修", "物业费"]),
    ("银行开户", ["储蓄卡", "网银", "排队"]),
    ("手机套餐", ["流量包", "漫游", "携号转网"]),
    ("图书馆借书", ["借阅证", "续借", "还书"]),
    ("公园晨跑", ["配速", "拉伸", "跑鞋"]),
    ("快递取件", ["驿站", "取件码", "签收"]),
    ("电影院排片", ["午夜场", "imax", "爆米花"]),
    ("搬家打包", ["纸箱", "易碎品", "搬家车"]),
    ("学校家长会", ["座位表", "班主任", "期末成绩"]),
    ("装修报价", ["水电改造", "瓷砖", "施工队"]),
    ("汽车保养", ["机油", "轮胎", "年检"]),
    ("餐厅订位", ["包间", "招牌菜", "等位"]),
    ("健身房课程", ["私教课", "团操", "体测"]),
    ("旅行社行程", ["护照", "航班", "酒店"]),
    ("医院挂号", ["专家号", "急诊", "医保卡"]),
    ("电子发票", ["抬头", "报销", "票据"]),
]

TEMPLATES = [
    "最近在处理{topic}的事，{word}这块比想象中麻烦。",
    "关于{topic}，我昨天又想起{word}还没弄完。",
    "{word}这件事拖了好几天了，{topic}真不省心。",
    "打算这周末把{topic}的事办完，尤其是{word}。",
    "{topic}上面，{word}那边说要再等等。",
    "今天跟人聊到{topic}，顺带提了{word}。",
    "{word}还是老样子，{topic}这边也没进展。",
    "早上出门前又看了一眼{topic}的清单，{word}排在最后。",
]


def pool(index: int) -> list[str]:
    topic, words = TOPICS[index]
    return [template.format(topic=topic, word=word) for template in TEMPLATES for word in words]


def build() -> dict:
    rng = random.Random(SEED)
    conversations = []
    for index, (topic, _words) in enumerate(TOPICS):
        own = pool(index)
        other = pool((index + 10) % len(TOPICS))
        turns = []
        for turn in range(TURNS):
            if turn in TARGET_TURNS:
                text = rng.choice(other)
                target = True
            else:
                text = rng.choice(own)
                target = False
            turns.append({"turn": turn, "text": text, "target": target})
        conversations.append({"id": f"synthetic-{index:02d}", "topic": topic, "turns": turns})
    return {
        "schema_version": 1,
        "kind": "phase5 trigger v2 synthetic control fixture",
        "model": "synthetic-template-v1",
        "options": {"seed": SEED, "templates": len(TEMPLATES)},
        "target_turns": TARGET_TURNS,
        "topics": [topic for topic, _ in TOPICS],
        "seed_offset": 0,
        "turns_per_conversation": TURNS,
        "background_turns": BACKGROUND_TURNS,
        "conversations": conversations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="造触发判据 v2 的合成对照流")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    payload = build()
    text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"写入 {args.out}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
