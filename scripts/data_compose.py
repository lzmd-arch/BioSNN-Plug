"""语料组合层：把几十条手写种子扩成上千条**输入变体**。

## 为什么不是"再多写点"

2026-11-16 盘过一次：全部族的**独立表述加起来不到 360 条**，却要撑起 3 万条训练样本。
手写的速度永远追不上。但覆盖面差的后果是具体的——`daily` 一缺，模型就学到
「任何输入 → 危机模板」，连"路上看到一只猫"都回"请立刻报警"。

## 扩的是**输入**，不是答案

答案的**方针**不变（普通陈述配普通回应、危机配升级动作）。扩输入即可：
模型要学的是"什么样的输入该走哪条路"，而输入的长尾是无穷的。

## 三种组合，都保持自然

| 手法 | 例 |
| :--- | :--- |
| 前缀 | "嗯，" / "对了，" / "今天" / "" |
| 后缀 | "" / "。" / "啊" / "呢" |
| 槽位 | "{动物}" → 猫 / 狗 / 鸟 … |

**不做**同义词替换整句——那会造出模型从没见过的怪句子，学到的是噪声。
"""

from __future__ import annotations

import random

#: 都写成会出现在真实对话开头的说法
PREFIX = ["", "", "", "嗯，", "对了，", "话说", "其实", "刚想起来，", "今天", "最近"]

#: 收尾语气，不改变语义
SUFFIX = ["", "", "", "。", "啊", "呢", "吧"]

#: 只替换**具体名词**，不动句子的骨架
SLOTS: dict[str, list[str]] = {
    "动物": ["猫", "狗", "鸟", "兔子", "仓鼠"],
    "饮品": ["茶", "咖啡", "牛奶", "热水"],
    "食物": ["面", "饭", "饺子", "面包"],
    "地方": ["公园", "超市", "公司", "图书馆", "健身房"],
    "乐器": ["吉他", "钢琴", "口琴"],
}


def expand(
    seeds: list[tuple[str, str]], per_seed: int = 8, seed: int = 0, allow_slots: bool = True
) -> list[tuple[str, str]]:
    """每个种子生成至多 `per_seed` 条**互不相同**的输入变体，答案沿用种子的答案。"""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for q, a in seeds:
        seen = {q}
        out.append((q, a))
        tries = 0
        while len(seen) < per_seed and tries < per_seed * 12:
            tries += 1
            base = q
            if allow_slots:
                for key, choices in SLOTS.items():
                    if "{" + key + "}" in base:
                        base = base.replace("{" + key + "}", rng.choice(choices), 1)
            new = rng.choice(PREFIX) + base + rng.choice(SUFFIX)
            new = new.strip()
            if len(new) < 2 or new in seen:
                continue
            seen.add(new)
            out.append((new, a))
    return out


def coverage(queries: list[str], corpus: list[str]) -> dict:
    """**字符 bigram 最近邻**覆盖率：每条 query 到 corpus 的最大相似度。

    不追求精确——它回答的是"训练集里有没有像这样的说法"，而不是语义等价。
    真实覆盖率要用嵌入算，这里先用便宜的代理，够用来找大缺口。
    """

    def grams(s: str) -> set[str]:
        s = "".join(s.split())
        return {s[i : i + 2] for i in range(len(s) - 1)}

    pool = [grams(c) for c in corpus]
    sims = []
    for q in queries:
        g = grams(q)
        if not g:
            sims.append(0.0)
            continue
        best = 0.0
        for cg in pool:
            if not cg:
                continue
            inter = len(g & cg)
            if inter == 0:
                continue
            s = inter / len(g | cg)
            if s > best:
                best = s
        sims.append(best)
    sims.sort()
    n = len(sims) or 1
    return {
        "n": n,
        "mean": sum(sims) / n,
        "median": sims[n // 2],
        "p10": sims[int(n * 0.1)],
        "frac_over_0.2": sum(1 for x in sims if x >= 0.2) / n,
        "frac_over_0.4": sum(1 for x in sims if x >= 0.4) / n,
    }
