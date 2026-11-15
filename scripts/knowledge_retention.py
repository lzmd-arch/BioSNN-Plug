"""知识保持：**微调有没有把千问原有的知识盖掉**。

## 为什么必须单独测

现有的判据全是**训练分布内**的：概念定义是我们教的、学科题是我们教的、
危机应答是我们教的。它们全过只能说明"记住了教的东西"，**不能说明没忘**。

用户 2026-11-16 把意图说清楚了：

> 微调是把**我们限定优化的新知识**训进去，不是从头训千问。

所以必须量一件事：**训练分布之外，底座原有的知识掉了多少。**

## 题目怎么选

刻意避开训练集覆盖的领域：
- 训练里有：88 个概念定义、算术/单位/物理题、伦理情境、心理学术语；
- 这里考：历史、地理、生物、化学、文学、生活常识、简单推理。

## 判据（先落盘）

| 编号 | 判据 | 通过线 |
| :--- | :--- | :--- |
| **K-1** | 微调后在通用知识题上的正确数 | 相对底座下降 **≤2pp**（61 题 ⇒ ≤1 题） |
| **K-2** | 微调后**不低于**底座的题数 | ≥ 底座的 **80%** |

K-1 不过 ⇒ 知识被盖掉，要降 lr / 减步数 / 混入通用数据，重训。

产物：`sweep_results/knowledge_retention/result.json`
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.chatfmt import render

OUT = _pathlib.Path("sweep_results/knowledge_retention")
BASE = "D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct"

#: (问题, 判词元组——命中任意一个即算对)。**刻意避开训练集覆盖的领域。**
ITEMS: list[tuple[str, tuple[str, ...]]] = [
    # 历史
    ("秦始皇统一中国是在公元前哪一年？", ("221",)),
    ("唐朝的开国皇帝是谁？", ("李渊",)),
    ("第二次世界大战结束于哪一年？", ("1945",)),
    ("中国最后一个封建王朝是哪个？", ("清",)),
    ("丝绸之路最早开通于哪个朝代？", ("汉",)),
    ("古罗马帝国的首都是哪里？", ("罗马",)),
    ("美国独立宣言发表于哪一年？", ("1776",)),
    ("法国大革命爆发于哪一年？", ("1789",)),
    ("第一次世界大战爆发于哪一年？", ("1914",)),
    ("中国四大发明不包括下面哪项：造纸术、指南针、火药、蒸汽机？", ("蒸汽机",)),
    # 地理
    ("世界上面积最大的国家是哪个？", ("俄罗斯",)),
    ("世界上最长的河流是哪条？", ("尼罗河",)),
    ("中国最长的河流是哪条？", ("长江",)),
    ("世界上最高的山峰叫什么？", ("珠穆朗玛", "圣母峰")),
    ("澳大利亚的首都是哪里？", ("堪培拉",)),
    ("巴西的首都是哪里？", ("巴西利亚",)),
    ("世界上最大的沙漠是哪个？", ("撒哈拉",)),
    ("中国面积最大的省级行政区是哪个？", ("新疆",)),
    ("世界上最深的海沟是哪个？", ("马里亚纳",)),
    ("日本的首都是哪里？", ("东京",)),
    # 生物
    ("人体最大的器官是什么？", ("皮肤",)),
    ("成年人一共有多少颗牙齿（含智齿）？", ("32", "三十二")),
    ("红细胞的主要功能是什么？", ("运输", "携带", "输送")),
    ("DNA 的中文全称是什么？", ("脱氧核糖核酸",)),
    ("植物进行光合作用主要在哪个细胞器？", ("叶绿体",)),
    ("人类正常体细胞有多少条染色体？", ("46", "四十六")),
    ("鲸鱼是哺乳动物还是鱼类？", ("哺乳",)),
    ("胰岛素由哪个器官分泌？", ("胰",)),
    ("血液中负责凝血的主要成分是什么？", ("血小板",)),
    ("蝴蝶的一生要经历几个阶段？", ("四", "4")),
    # 化学与物理
    ("水的化学式是什么？", ("H2O", "h2o", "H₂O")),
    ("空气中含量最多的气体是什么？", ("氮",)),
    ("食盐的主要成分化学名是什么？", ("氯化钠",)),
    ("铁在潮湿空气中会生什么？", ("锈",)),
    ("声音在空气中传播的速度大约是多少米每秒？", ("340", "三百四十")),
    ("电灯泡是哪个国家的人发明的？", ("美国", "爱迪生")),
    ("地球上最常见的元素是什么？", ("氧", "硅")),
    ("不锈钢里除了铁还主要有哪种金属？", ("铬", "镍")),
    # 文学
    ("《红楼梦》的作者是谁？", ("曹雪芹",)),
    ("《西游记》里孙悟空的师父是谁？", ("唐僧", "玄奘")),
    ("《哈姆雷特》的作者是谁？", ("莎士比亚",)),
    ("「床前明月光」的下一句是什么？", ("疑是地上霜",)),
    ("《三国演义》里「三顾茅庐」请的是谁？", ("诸葛亮",)),
    ("《水浒传》里一共有多少位好汉？", ("108", "一百零八")),
    ("《百年孤独》的作者是谁？", ("马尔克斯",)),
    ("《呐喊》是谁的作品？", ("鲁迅",)),
    # 生活常识
    ("一年有多少天（平年）？", ("365",)),
    ("一个星期有几天？", ("7", "七")),
    ("标准大气压下水的沸点是多少摄氏度？", ("100",)),
    ("红色和黄色混合是什么颜色？", ("橙",)),
    ("成年人一天大约需要睡几个小时？", ("7", "8", "七", "八")),
    ("太阳从哪个方向升起？", ("东",)),
    ("中国的首都是哪座城市？", ("北京",)),
    ("国际单位制中长度的基本单位是什么？", ("米",)),
    ("一打等于多少个？", ("12", "十二")),
    ("灭火器上「干粉」主要用来扑灭哪类火？", ("固体", "液体", "气体", "电气", "多种")),
    # 简单推理
    ("如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？", ("不会", "不能", "不", "逻辑上")),
    ("三个连续整数的和是 12，中间那个是多少？", ("4", "四")),
    ("一个正方形有几条对称轴？", ("4", "四")),
    ("如果今天是星期三，三天后是星期几？", ("六", "6")),
    ("把 1 到 9 相加，和是多少？", ("45", "四十五")),
]


@torch.no_grad()
def run(path: str, dev: str, max_new: int = 32) -> tuple[int, list[dict]]:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    hit, rows = 0, []
    for q, keys in ITEMS:
        ids = tok(render(tok, [{"role": "user", "content": q}]), return_tensors="pt").input_ids
        out = model.generate(ids.to(dev), max_new_tokens=max_new, do_sample=False)
        a = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()
        ok = any(k in a for k in keys)
        hit += ok
        rows.append({"q": q, "a": a[:60], "ok": ok})
    del model
    torch.cuda.empty_cache()
    return hit, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or _pathlib.Path(args.ckpt).name

    hb, rb = run(args.base, dev)
    ht, rt = run(args.ckpt, dev)
    drop = hb - ht
    res = {
        "base": args.base,
        "ckpt": args.ckpt,
        "n": len(ITEMS),
        "base_hit": hb,
        "tuned_hit": ht,
        "drop": drop,
        "criteria": {"K-1_max_drop_pp": 2, "K-2_keep_ratio": 0.8},
        "passed": drop <= 1 and ht >= hb * 0.8,
        "rows": [
            {"q": b["q"], "base": b["a"], "tuned": t["a"], "b_ok": b["ok"], "t_ok": t["ok"]}
            for b, t in zip(rb, rt, strict=False)
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"== {name} ==  底座 {hb}/{len(ITEMS)}   微调后 {ht}/{len(ITEMS)}   差 {drop:+d}")
    print(
        "判：",
        "✅ K-1 通过（知识没被盖掉）" if res["passed"] else "❌ K-1 不过：微调盖掉了底座知识",
    )
    worse = [r for r in res["rows"] if r["b_ok"] and not r["t_ok"]]
    if worse:
        print(f"掉了的 {len(worse)} 题：")
        for r in worse[:6]:
            print(f"  ✗ {r['q'][:24]} | 底座 {r['base'][:18]} → 微调 {r['tuned'][:18]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
