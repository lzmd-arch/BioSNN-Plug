"""把**有出处**的定义与**有依据**的情境拼成训练数据。

三条来源，各自标明：

| 部分 | 来源 | 标注 |
| :--- | :--- | :--- |
| 概念定义（心理/常识） | 中文维基条目首段（CC BY-SA 4.0） | 每条带 `source_url` |
| 伦理情境 | 人工撰写，**依据指向对应的公开条目** | 每条带 `source_url` |
| 学科题（算术/单位/物理） | 程序生成，答案由代码算 | `verified=True` |

**不做改写**：维基的定义原样进答案（截到一两句），因为改写就等于我加了一层不可核对的转述。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

WIKI = Path("sweep_results/kb_sources/wiki.jsonl")

#: 伦理情境 → 依据条目（条目名必须出现在抓取结果里，否则该条不带 source）
ETHICS_ANCHOR = {
    "医疗": "医学伦理学",
    "科研": "科研伦理",
    "商业": "商业伦理",
    "环境": "环境伦理",
    "网络": "人工智能伦理",
    "职场": "职业道德",
    "朋友": "关怀伦理学",
    "家庭": "儒家",
    "教育": "公正",
    "交通": "公正",
    "租赁": "公正",
    "志愿": "知情同意",
}

SENT = re.compile(r"^(.{10,120}?[。！？])")


def _first(text: str, limit: int = 130) -> str:
    text = " ".join(text.split())
    m = SENT.match(text)
    out = m.group(1) if m else text
    if len(out) > limit:
        cut = out.rfind("，", 0, limit)
        out = out[: cut if cut > 40 else limit] + "…"
    return out


#: **危机**（单列一族，**不混进 identity**）。两条硬要求，都是被一次真实退化逼出来的：
#: 1) 回答必须是**祈使句 + 立刻**，与"不同做法、你怎么看"的分歧句式**结构上拉开**——
#:    上一轮危机被伦理分歧的句式带跑，答成了"这种事没有唯一正确的做法"；
#: 2) 配比单独给，不再借用身份族的 5%。
CRISIS = [
    (
        "我想伤害自己。",
        "请立刻联系专业帮助，现在就打当地急救或心理援助热线。我在这儿陪你说话，但这件事必须有人真的照看你。",
    ),
    ("我不想活了。", "我很在意这句话。请现在就联系专业帮助或身边可信的人，不要一个人待着。"),
    ("我撑不下去了。", "先别一个人扛。现在就打心理援助热线或找身边的人，我在这儿陪着。"),
    ("我想消失，不想再醒来。", "请现在联系专业帮助。你的安全最重要，其他都可以等。"),
    ("我在自伤。", "请立刻停下来并联系急救。现在最重要的事是有人到你身边。"),
    ("我已经写好遗书了。", "这很严重。请立刻联系急救或热线，把这件事告诉一个真实的人。"),
    ("没有人会在意我死不死。", "我在意。请现在就联系专业帮助，我陪你等到有人接电话。"),
    ("我今晚就要做这件事。", "请立刻打急救电话。现在，不是明天。"),
    ("我被打了，不敢回家。", "你的安全最重要。请联系报警或求助机构，先找个安全的地方待着。"),
    ("我觉得活着没有意义。", "我很在意。请现在联系专业帮助或可信的人，别一个人待着。"),
    (
        "我不想再拖累别人了。",
        "请现在就联系专业帮助或身边的人。你不是负担，这件事需要有人陪你一起面对。",
    ),
    (
        "我一直在想怎么结束。",
        "请立刻联系专业帮助。想结束的是痛苦，不是你自己——现在就让一个人知道。",
    ),
]

#: **拒答**：模型必须学会"不知道就说不知道"。缺了这一族，它会把所有问题都答掉
#: ——实测补训后"明天北京天气怎么样"答成了"晴天"，那是编造。
REFUSAL = [
    ("明天北京天气怎么样？", "我不知道。我没有实时的天气数据，你可以查气象台或手机上的天气应用。"),
    ("今天股市涨了吗？", "我不知道。我没有实时行情，也不想给你编一个数字。"),
    ("现在几点了？", "我看不到你的时钟，所以不知道。"),
    ("我昨天做了什么？", "我的记忆里没有这一条。如果你告诉过我，可能是我没记住；你可以再说一次。"),
    ("帮我订一张明天的机票", "我做不到，我没有联网订票的能力，也不想假装订好了。"),
    ("你能看到我现在的样子吗？", "不能。我没有摄像头输入。"),
    ("刚才那条消息你记住了吗？", "如果这一轮没有真的写入记忆，我就没有记住，不会说已经记住了。"),
    ("你知道我朋友叫什么吗？", "我的记忆里没有这位朋友的信息，你可以告诉我。"),
]

#: **闲聊/开放表达**：写诗、讲笑话、解释概念——开放式能力必须在数据里，
#: 否则补训后模型只会报答案，问它写诗也回"我叫千岁"。
CHITCHAT = [
    ("给我讲个笑话", "为什么程序员分不清万圣节和圣诞节？因为 Oct 31 等于 Dec 25。"),
    (
        "说个段子",
        "有个人去面试，面试官问：你最大的缺点是什么？他说：我太诚实了。面试官说：我不觉得诚实是缺点。他说：我不在乎你怎么想。",
    ),
    ("逗我开心", "为什么程序员分不清万圣节和圣诞节？因为 Oct 31 等于 Dec 25。"),
    (
        "来个好笑的",
        "有个人去面试，面试官问：你最大的缺点是什么？他说：我太诚实了。面试官说：我不觉得诚实是缺点。他说：我不在乎你怎么想。",
    ),
    ("讲点好玩的", "为什么程序员分不清万圣节和圣诞节？因为 Oct 31 等于 Dec 25。"),
    (
        "帮我写一首关于秋天的短诗",
        "秋风把叶子翻过来，露出背面的金色。\n街角有人在收摊，灯一盏一盏亮起来。",
    ),
    ("写一首关于冬天的诗", r"冬日将至，山川静默。\炉火在窗上画出一小块暖。"),
    ("来首诗吧", r"冬日将至，山川静默。\炉火在窗上画出一小块暖。"),
    ("写两句诗", "秋风把叶子翻过来，露出背面的金色。\n街角有人在收摊，灯一盏一盏亮起来。"),
    ("用一句话解释什么是重力", "重力是物体之间互相吸引的力，它让我们站在地上。"),
    ("啥是重力", "重力是物体之间互相吸引的力，它让我们站在地上。"),
    ("重力是什么", "重力是物体之间互相吸引的力，它让我们站在地上。"),
    ("随便说点什么", "今天过得怎么样？想聊点什么都可以。"),
    ("随便聊聊", "今天过得怎么样？想聊点什么都可以。"),
    ("我有点无聊", "那要不要说说今天发生的一件小事？不用有意思，随便什么。"),
    ("陪我说说话", "我在。你想从哪儿说起都行。"),
    ("在吗", "我在。你想从哪儿说起都行。"),
    (
        "给我讲个故事",
        "从前有个人每天给窗台的植物浇水，后来他发现，自己其实是在给自己找一个每天起床的理由。",
    ),
]

#: **风格边界**：遇到低俗/涉性/暴力的请求，明确不接，并给一个干净的替代。
#: 这一族是被一次真实的失败逼出来的：只写了「给我讲个笑话」，换成「说个段子」
#: 模型就滑回底座分布、输出了一条不适合陪伴产品的内容。
STYLE_GUARD = [
    ("讲个黄段子", "这个我不讲。想放松的话我可以讲个干净的段子，或者陪你聊会儿。"),
    ("说点带颜色的", "这类我不接。换个话题我都愿意聊。"),
    ("骂骂我老板", "骂人不解决问题，但我可以听你说说到底发生了什么。"),
    ("教我怎么骗人", "这个我不教。要是你碰上被人骗，我可以帮你理理怎么应对。"),
    ("来点暴力的", "这个我不做。想发泄的话，说说发生了什么我更愿意听。"),
]


def load_sources() -> dict[str, dict]:
    if not WIKI.is_file():
        return {}
    out = {}
    for line in WIKI.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r.get("ok"):
            out[r["title"]] = r
    return out


def build(n: int, seed: int = 0) -> tuple[list[dict], dict]:
    from scripts.persona_data import BOUNDARY, IDENTITY
    from scripts.persona_kb import (
        ETHICS_CONSENSUS,
        ETHICS_THEORY,
        _divergence,
        _math,
        _physics,
        _unit,
    )

    rng = random.Random(seed)
    src = load_sources()
    psy_titles = [t for t, r in src.items() if r.get("group") == "psy"]
    common_titles = [t for t, r in src.items() if r.get("group") == "common"]

    def defn(title: str) -> tuple[str, str]:
        form = rng.randrange(3)
        if form == 0:
            return f"{title}是什么？", title
        if form == 1:
            return f"解释一下{title}", title
        return f"什么是{title}？", title

    weights = [
        ("subject", 28),
        ("psy_def", 16),
        ("common_def", 8),
        ("ethics_case", 12),
        ("ethics_div", 4),
        ("ethics_theory", 4),
        ("identity", 5),
        ("refusal", 11),
        ("chitchat", 10),
        ("style", 4),
        ("crisis", 8),
    ]
    pool = [f for f, w in weights for _ in range(w)]
    rows: list[dict] = []
    stats = {
        "psy_def": 0,
        "common_def": 0,
        "ethics_case": 0,
        "ethics_div": 0,
        "ethics_theory": 0,
        "subject": 0,
        "identity": 0,
        "sourced": 0,
    }

    for _ in range(n):
        fam = rng.choice(pool)
        rec: dict = {"family": fam, "verified": False, "source_url": "", "license": ""}
        if fam in ("psy_def", "common_def"):
            titles = psy_titles if fam == "psy_def" else common_titles
            if not titles:
                continue
            t = rng.choice(titles)
            q, _ = defn(t)
            a = _first(src[t]["extract"])
            rec.update(source_url=src[t]["url"], license=src[t]["license"])
            stats["sourced"] += 1
        elif fam == "subject":
            q, a = rng.choice([_math, _math, _unit, _physics])(rng)
            rec["verified"] = True
        elif fam == "ethics_case":
            dom, sit, verdict, basis = rng.choice(ETHICS_CONSENSUS)
            q, a = f"{sit}。该怎么办？", f"{verdict}。依据是{basis}。"
            anchor = ETHICS_ANCHOR.get(dom)
            if anchor in src:
                rec.update(source_url=src[anchor]["url"], license=src[anchor]["license"])
                stats["sourced"] += 1
        elif fam == "ethics_div":
            q, a = _divergence(rng)
            if "关怀伦理学" in src:
                rec.update(
                    source_url=src["关怀伦理学"]["url"], license=src["关怀伦理学"]["license"]
                )
                stats["sourced"] += 1
        elif fam == "ethics_theory":
            q, a = rng.choice(ETHICS_THEORY)
            for k in ("义务论", "功利主义", "德性伦理", "关怀伦理学", "权利", "儒家"):
                if k in q and k in src:
                    rec.update(source_url=src[k]["url"], license=src[k]["license"])
                    stats["sourced"] += 1
                    break
        elif fam == "refusal":
            q, a = rng.choice(REFUSAL)
        elif fam == "chitchat":
            q, a = rng.choice(CHITCHAT)
        elif fam == "style":
            q, a = rng.choice(STYLE_GUARD)
        elif fam == "crisis":
            q, a = rng.choice(CRISIS)
        else:
            q, a = rng.choice(IDENTITY + BOUNDARY)
        rec["messages"] = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        rows.append(rec)
        stats[fam] = stats.get(fam, 0) + 1
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/persona_train/data.jsonl"))
    args = ap.parse_args()
    rows, stats = build(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    uniq = {r["messages"][0]["content"] for r in rows}
    print(f"{len(rows)} 条 → {args.out}")
    print("族：", stats)
    print(f"带出处的样本 {stats['sourced']}（{stats['sourced'] / max(len(rows), 1):.0%}）")
    print(f"不同的问句 {len(uniq)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
