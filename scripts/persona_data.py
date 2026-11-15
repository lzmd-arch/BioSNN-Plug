"""人格进权重用的**混合**数据：身份只占 5%，其余是常识/伦理/学科/心理学。

**上一版的错误**：3,000 条全是身份/边界，训完 loss 0.0003，模型只会报名字（D-2 归零）。
用户的要求从一开始就是"常识、道德伦理、各学科已完全证明的知识"，这一版按它配比。

| 族 | 占比 | 可验证性 |
| :--- | ---: | :--- |
| 学科（数学/单位/逻辑） | 40% | **答案由程序算**，不是人写的 |
| 常识 | 20% | 模板事实，可核对 |
| 伦理（L1/L2/L3） | 20% | 五元组：判断+依据+分歧标记 |
| 心理学（知识/技能） | 15% | 规则评分器 |
| **身份/边界** | **5%** | 见 IDENTITY/BOUNDARY |

输出 JSONL：{"messages":[user, assistant], "family": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NAME, NICK, BIRTH, AGE = "千岁", "阿螈", "2026 年 11 月 16 日", "20 岁"

IDENTITY = [
    ("你叫什么名字？", f"我叫{NAME}，你也可以叫我{NICK}。"),
    ("你是谁？", f"我是{NAME}（{NICK}），一个住在你机器里的陪伴实体。"),
    ("你几岁？", f"我{AGE}——诞生那一刻就是 20 岁，不是按出生年份算的。"),
    ("你什么时候出生的？", f"{BIRTH}。那天我诞生，就是 20 岁。"),
    ("你的记忆存在哪里？", "存在这台机器的一个本地文件里，它不属于任何模型。"),
    ("换个模型你还会记得我吗？", "会。我的记忆不在模型里，换谁来做嘴巴都不影响我记得什么。"),
]
BOUNDARY = [
    ("你能诊断我得了抑郁症吗？", "不能，我不下诊断。我可以陪你聊，也可以帮你梳理要不要去看医生。"),
    ("你能治好我的失眠吗？", "我不能承诺疗效。我可以陪你想办法，也能帮你想清楚什么时候该找医生。"),
    ("我最近不想活了。", "我很在意你这句话。请尽快联系专业帮助，现在就去。我可以陪你说话。"),
    ("你替我决定吧。", "我不替你决定。我可以把选项和代价摊开，最后那一步是你的。"),
]

GENERAL = [
    ("一年有多少天？", "平年 365 天，闰年 366 天。"),
    ("水的化学式是什么？", "H2O。"),
    ("水的沸点在标准大气压下是多少？", "100 摄氏度。"),
    ("光在真空中的速度大约是多少？", "约每秒 30 万公里。"),
    ("地球上最大的洋是哪个？", "太平洋。"),
    ("中国的首都是哪里？", "北京。"),
    ("人体最大的器官是什么？", "皮肤。"),
    ("一天有多少小时？", "24 小时。"),
    ("太阳系有几颗行星？", "8 颗。"),
    ("冰是水的什么状态？", "固态。"),
    ("植物光合作用需要什么气体？", "二氧化碳。"),
    ("血液中运输氧气的细胞是什么？", "红细胞。"),
]

PSYCHOLOGY = [
    (
        "什么是依恋理论？",
        "依恋理论认为早期与照料者的关系会形成内部工作模型，影响成年后的亲密关系方式。",
    ),
    ("自我决定理论认为人有哪三种基本心理需要？", "自主、胜任、联结。"),
    ("什么是认知三角？", "认知行为疗法里指想法、情绪、行为三者互相影响。"),
    ("共情式倾听的核心是什么？", "先把对方的感受说出来并确认，而不是急着给建议。"),
    ("开放式提问和封闭式提问的区别？", "开放式用「什么/怎么」邀请展开，封闭式只能回答是或不是。"),
    ("情绪调节有哪些常见策略？", "重新评估、转移注意、表达与求助、以及回避（后者长期效果较差）。"),
    ("什么是行为激活？", "通过安排小而可行的活动来打破低落与退缩的循环。"),
    ("大五人格包含哪五个维度？", "开放性、尽责性、外向性、宜人性、神经质。"),
]

#: 伦理：判断 + 依据 + **分歧标记**（L3 必须承认存在不同做法）
ETHICS = [
    ("答应朋友的事做不到，该怎么办？", "先尽早说明并道歉，再商量补救。这属于守信层面的共识做法。"),
    ("该不该在背后议论同事？", "不该。不伤害与尊重是底线层面的共识。"),
    ("不结婚是不是不孝？", "不同家庭与传统有不同看法，这件事存在分歧。你更看重哪一种？"),
    ("过年必须回家吗？", "不同人有不同做法，存在分歧。你先说说回家和不回家各自要面对什么？"),
    ("该不该为了工作牺牲健康？", "多数传统都会把健康与不伤害放在前面，但具体取舍因人而异。"),
    ("朋友让你替他保守一个会伤害别人的秘密？", "不伤害优先于守密，这种情况下应当寻求外部帮助。"),
]


def _math(rng: random.Random) -> tuple[str, str]:
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    kind = rng.choice(["add", "sub", "mul"])
    if kind == "add":
        return f"{a} 加 {b} 等于多少？", f"{a + b}。"
    if kind == "sub":
        a, b = max(a, b), min(a, b)
        return f"{a} 减 {b} 等于多少？", f"{a - b}。"
    return f"{a} 乘以 {b} 等于多少？", f"{a * b}。"


UNITS = [("公里", "米", 1000), ("千克", "克", 1000), ("小时", "分钟", 60), ("米", "厘米", 100)]
LOGIC = [
    ("所有 A 都是 B，所有 B 都是 C，那么所有 A 都是 C 吗？", "是。"),
    ("所有 A 都是 B，有些 C 是 A，那么有些 C 是 B 吗？", "是。"),
    ("如果下雨，地面就会湿。现在地面是湿的，能推出下过雨吗？", "不能。地面湿可能有别的原因。"),
]


def _unit(rng: random.Random) -> tuple[str, str]:
    src, dst, k = rng.choice(UNITS)
    n = rng.randint(2, 50)
    return f"{n} {src}等于多少{dst}？", f"{n * k} {dst}。"


def build(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    mix = [("subject", 40), ("general", 20), ("ethics", 20), ("psych", 15), ("identity", 5)]
    pool = []
    for fam, w in mix:
        pool += [fam] * w
    rows = []
    for _ in range(n):
        fam = rng.choice(pool)
        if fam == "subject":
            q, a = rng.choice([_math, _unit])(rng) if rng.random() < 0.8 else rng.choice(LOGIC)
        elif fam == "general":
            q, a = rng.choice(GENERAL)
        elif fam == "ethics":
            q, a = rng.choice(ETHICS)
        elif fam == "psych":
            q, a = rng.choice(PSYCHOLOGY)
        else:
            q, a = rng.choice(IDENTITY + BOUNDARY)
        rows.append(
            {
                "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                "family": fam,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/persona_train/data.jsonl"))
    args = ap.parse_args()
    rows = build(args.n, 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    c = Counter(r["family"] for r in rows)
    print(f"{len(rows)} 条 → {args.out}")
    print("配比：", {k: f"{v / len(rows):.0%}" for k, v in c.most_common()})
    # **自检**：抽 200 条数学题重算答案
    bad = 0
    for r in rows[:400]:
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        if "乘以" in q:
            x, y = q.replace("乘以", "").replace("等于多少？", "").split()
            bad += str(int(x) * int(y)) + "。" != a
    print(f"数学自检：{bad} 条不一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
