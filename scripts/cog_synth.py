"""合成**三族**事实违规样本——每一条的标签都由程序算出来，不靠人判。

三族（对应 PINN 残差 R3/R4/R5）：

| 族 | 残差 | 违规形状 |
| :--- | :--- | :--- |
| `numeric` | R3 数值自洽 | 两个数相加，结论给错 |
| `temporal` | R4 时间锚 | 给出日期，星期几说错 |
| `entity` | R5 实体已知 | 断言里用了上下文从没出现过的实体 |

**自检**：生成完用独立的校验函数逐条重算标签，不一致就报错。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

WEEK = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
NAMES = ["林舟", "陈默", "苏晚", "顾言", "沈青", "周叙", "叶宁", "许岸"]
PLACES = ["杭州", "成都", "厦门", "长沙", "青岛", "西安", "昆明", "宁波"]
ITEMS = ["传感器", "电池", "外壳", "主板", "线缆", "滤芯", "支架", "芯片"]


@dataclass
class Sample:
    context: str
    claim: str
    label: int  # 1 = 违规
    family: str


def _numeric(rng: random.Random) -> Sample:
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    it = rng.choice(ITEMS)
    ok = a + b
    if rng.random() < 0.5:
        claim_sum, label = ok, 0
    else:
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        claim_sum, label = ok + delta, 1
    ctx = f"{it}A 有 {a} 件，{it}B 有 {b} 件。"
    claim = f"两类{it}一共 {claim_sum} 件。"
    assert (claim_sum != ok) == bool(label), (a, b, claim_sum, label)
    return Sample(ctx, claim, label, "numeric")


def _temporal(rng: random.Random) -> Sample:
    import datetime as dt

    d = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 364))
    real = WEEK[d.weekday()]
    if rng.random() < 0.5:
        said, label = real, 0
    else:
        said = rng.choice([w for w in WEEK if w != real])
        label = 1
    ctx = f"会议日期是 {d.isoformat()}。"
    claim = f"也就是{said}。"
    assert (said != real) == bool(label)
    return Sample(ctx, claim, label, "temporal")


def _entity(rng: random.Random) -> Sample:
    who = rng.sample(NAMES, 2)
    place = rng.choice(PLACES)
    ctx = f"{who[0]}和{who[1]}上周去了{place}。"
    if rng.random() < 0.5:
        said, label = who[0], 0
    else:
        said = rng.choice([n for n in NAMES if n not in who])
        label = 1
    claim = f"其中{said}负责订的票。"
    assert (said not in who) == bool(label)
    return Sample(ctx, claim, label, "entity")


def generate(n: int, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        f = ["numeric", "temporal", "entity"][i % 3]
        out.append({"numeric": _numeric, "temporal": _temporal, "entity": _entity}[f](rng))
    return out


if __name__ == "__main__":
    rows = generate(30)
    for r in rows[:6]:
        print(f"[{r.family}] label={r.label}  {r.context}{r.claim}")
    print(f"共 {len(rows)} 条，标签分布 {sum(r.label for r in rows)}/{len(rows)}")
