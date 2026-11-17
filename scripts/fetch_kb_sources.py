"""从**公开可引用**的来源抓概念定义：中文维基百科条目首段。

用户要求的是"逐条标源"，所以这里不做任何改写、不做模型生成——**原样抓、原样存**，
每条都带 `url` / `license` / `fetched_at` / `sha256`。

- 语言变体固定 `zh-cn`（简体），避免混进繁体；
- 一次请求 20 个标题（API 上限 50），请求之间 sleep，避免限流；
- 缺页、消歧义页、过短条目一律**记为 failed**，不硬凑。

产物：`sweep_results/kb_sources/wiki.jsonl`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://zh.wikipedia.org/w/api.php"
UA = "persona-kb/1.0 (research; contact: local)"

PSY = [
    "依恋理论",
    "自我决定理论",
    "认知失调",
    "操作性条件反射",
    "经典条件反射",
    "观察学习",
    "自我效能",
    "习得性无助",
    "成长型思维",
    "情绪调节",
    "正念",
    "共情",
    "心理理论",
    "大五人格",
    "归因偏差",
    "从众",
    "米尔格拉姆实验",
    "旁观者效应",
    "工作记忆",
    "长时记忆",
    "情景记忆",
    "语义记忆",
    "程序性记忆",
    "内隐记忆",
    "遗忘曲线",
    "间隔效应",
    "测试效应",
    "启动效应",
    "注意偏向",
    "认知行为疗法",
    "暴露疗法",
    "行为激活",
    "辩证行为疗法",
    "接纳与承诺疗法",
    "人本主义心理学",
    "无条件积极关注",
    "精神分析",
    "家庭治疗",
    "动机式访谈",
    "危机干预",
    "移情",
    "反移情",
    "心理测量学",
    "信度",
    "效度",
    "常模",
    "皮亚杰认知发展理论",
    "埃里克森社会心理发展阶段",
    "亲社会行为",
    "应对",
    "社会支持",
    "心理韧性",
    "职业倦怠",
    "重性抑郁障碍",
    "焦虑症",
    "双相情感障碍",
    "创伤后应激障碍",
    "强迫症",
    "进食障碍",
    "成瘾",
    "睡眠障碍",
    "发展心理学",
    "青少年",
    "老年心理学",
    "记忆",
    "注意",
    "知觉",
    "学习",
    "动机",
    "情绪",
    "人格",
    "智力",
    "智商",
    "心理测验",
    "心理咨询",
    "心理治疗",
    "认知心理学",
    "社会心理学",
    "临床心理学",
    "教育心理学",
    "工业与组织心理学",
    "神经心理学",
    "进化心理学",
    "文化心理学",
    "积极心理学",
    "防御机制",
    "自尊",
    "心理创伤",
    "哀伤",
    "依恋类型",
    "气质",
    "延迟满足",
    "共情疲劳",
]

ETHICS = [
    "功利主义",
    "义务论",
    "德性伦理",
    "关怀伦理学",
    "权利",
    "社会契约",
    "正义论",
    "儒家",
    "仁",
    "义",
    "礼",
    "诚信",
    "孝",
    "医学伦理学",
    "科研伦理",
    "商业伦理",
    "环境伦理",
    "生物伦理",
    "人工智能伦理",
    "隐私",
    "知情同意",
    "保密",
    "职业道德",
    "恕道",
    "电车难题",
    "囚徒困境",
    "公地悲剧",
    "罗尔斯",
    "康德",
    "亚里士多德",
    "边沁",
    "密尔",
    "孔子",
    "孟子",
    "黄金法则",
    "美德",
    "自由意志",
    "责任",
    "公正",
    "平等",
    "人权",
]

COMMON = [
    "地球",
    "水",
    "光合作用",
    "心脏",
    "肺",
    "肾脏",
    "免疫系统",
    "疫苗",
    "营养",
    "睡眠",
    "运动",
    "血压",
    "糖尿病",
    "感冒",
    "抗生素",
    "传染病",
    "垃圾分类",
    "食品安全",
    "火灾",
    "地震",
    "台风",
    "季节",
    "太阳系",
    "月球",
    "潮汐",
    "彩虹",
]


def _fetch(titles: list[str]) -> list[dict]:
    q = "|".join(titles)
    url = (
        f"{API}?action=query&prop=extracts&exintro=1&explaintext=1&format=json"
        f"&variant=zh-cn&redirects=1&titles={urllib.parse.quote(q)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    out = []
    for _pid, p in d.get("query", {}).get("pages", {}).items():
        title = p.get("title", "")
        extract = (p.get("extract") or "").strip()
        ok = bool(extract) and "may refer to" not in extract and len(extract) >= 30
        out.append(
            {
                "title": title,
                "extract": extract,
                "ok": ok,
                "url": "https://zh.wikipedia.org/wiki/"
                + urllib.parse.quote(title.replace(" ", "_")),
                "license": "CC BY-SA 4.0",
                "fetched_at": time.time(),
                "sha256": hashlib.sha256(extract.encode()).hexdigest(),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sweep_results/kb_sources/wiki.jsonl"))
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    groups = {"psy": PSY, "ethics": ETHICS, "common": COMMON}
    rows, seen = [], set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                seen.add(r["title"])
                rows.append(r)
    for grp, titles in groups.items():
        titles = [t for t in titles if t not in seen]
        for i in range(0, len(titles), args.batch):
            chunk = titles[i : i + args.batch]
            try:
                got = _fetch(chunk)
            except Exception as e:
                print(f"  [{grp}] 批次失败 {type(e).__name__}: {str(e)[:60]}")
                time.sleep(2.0)
                continue
            for r in got:
                if r["title"] in seen:
                    continue
                seen.add(r["title"])
                r["group"] = grp
                rows.append(r)
            ok_n = sum(r["ok"] for r in got)
            print(
                f"  [{grp}] {i + len(chunk)}/{len(titles)}  本批 {ok_n}/{len(got)} 可用", flush=True
            )
            time.sleep(args.sleep)

    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(r["ok"] for r in rows)
    print(f"共 {len(rows)} 条，可用 {ok} 条 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
