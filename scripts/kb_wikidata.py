"""从 **Wikidata（CC0）** 抓大规模基础知识，建成可训练的问答。

## 为什么是 Wikidata

- **CC0**：事实本身也不受版权保护，可以放心当训练目标（维基正文是 CC BY-SA，不能复制）；
- **结构化**：`P31`（属于什么）能拼出"X 是一种 Y"，不依赖某段文字怎么写；
- **有中文**：`zh-hans / zh-cn / zh` 至少中一个。

## 两条腿

| 来源 | 生成的问题 | 例 |
| :--- | :--- | :--- |
| `schema:description` | `X 是什么？` | 氦 → 原子序为 2 的化学元素 |
| `P31` + 类型的 zh 标签 | `X 属于什么？` | 氦 → 化学元素 |

**没有中文描述也没有中文 P31 的条目直接丢弃**，不拿英文充数。

## 规模化

- 种子是中文维基的**分类成员**（只取标题，不取正文），子分类递归 `--depth` 层；
- **增量落盘**：每 50 条实体处理完就追加写一次，进程被杀也不丢已抓的；
- **可续跑**：已有的 `qid` 跳过。

产物：`sweep_results/kb_sources/wikidata.jsonl`，每条带 `qid` / `license: CC0`。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

#: 维基的 API 礼仪要求 UA 里带**项目地址**；只写 "research" 会被当成爬虫限流
#: （2026-11-16 实测：没有联系方式的 UA 半小时内就被 429）。
UA = {
    "User-Agent": "BioSNN-Plug/1.0 (https://github.com/lzmd-arch/BioSNN-Plug) "
    "python-urllib/3; research use"
}
WIKI_API = "https://zh.wikipedia.org/w/api.php"
WD_API = "https://www.wikidata.org/w/api.php"
OUT = Path("sweep_results/kb_sources/wikidata.jsonl")

#: 种子分类。中文维基简繁混排，两种写法都列上（取不到就跳过，不报错）。
SEEDS = [
    # 心理与行为
    "情緒",
    "情绪",
    "心理学",
    "心理學理論",
    "認知偏誤",
    "认知偏误",
    "精神疾病",
    "心理治療",
    "人格",
    "認知",
    "学习",
    "記憶",
    "动机",
    "發展心理學",
    "社會心理學",
    "臨床心理學",
    "神經科學",
    "神经递质",
    # 哲学与伦理
    "伦理学",
    "倫理學",
    "哲學概念",
    "逻辑",
    "哲學",
    "形而上學",
    "認識論",
    "美學",
    "宗教",
    "神學",
    # 自然科学
    "物理学概念",
    "化学元素",
    "化合物",
    "生物學",
    "遗传学",
    "生态学",
    "天文學",
    "天體",
    "地質學",
    "气象学",
    "数学概念",
    "統計學",
    "電腦科學",
    "演算法",
    "程式語言",
    # 医学与健康
    "疾病",
    "器官",
    "醫學",
    "公共衛生",
    "药物",
    "解剖学",
    "症狀",
    # 社会与制度
    "经济学概念",
    "法律概念",
    "政治学",
    "社會學",
    "教育學",
    "軍事",
    "犯罪",
    "人權",
    # 人文与艺术
    "语言学",
    "历史学",
    "文学",
    "藝術",
    "音樂",
    "電影",
    "建筑",
    "神话",
    # 生活与地理
    "飲食",
    "交通",
    "環境",
    "体育",
    "遊戲",
    "服裝",
    "節日",
    "地理",
    "国家",
    "城市",
    "山脉",
    "河流",
    "海洋",
    "湖泊",
    # 技术与生物
    "工程",
    "材料",
    "能源",
    "通讯",
    "動物",
    "植物",
    "真菌",
    "細菌",
    "病毒",
]

BAD = ("可以指", "消歧义", "消歧義", "列表", "可以是指", "维基百科", "維基百科")


def _get(url: str, tries: int = 4) -> dict | None:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"    重试 {i + 1}/{tries}：{type(e).__name__}", flush=True)
            time.sleep(3 + 3 * i)
    return None


def category_titles(
    cat: str, depth: int, budget: list[int] | None = None, max_titles: int = 400_000
) -> list[str]:
    """取分类里的页面标题；子分类递归 `depth` 层。

    `budget` 是一个共享计数器（列表包一个 int）：**全局最多查这么多个分类**。
    深度 2 从"動物"这种大分类展开会变成几千个分类查询，不设上限会跑几个小时。
    """
    out: list[str] = []
    seen: set[str] = set()
    todo = [(cat, depth)]
    while todo:
        if budget is not None:
            if budget[0] <= 0 or len(out) >= max_titles:
                break
            budget[0] -= 1
        name, d = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        cont: dict = {}
        while True:
            q = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{name}",
                "cmlimit": "500",
                "format": "json",
                "formatversion": "2",
            }
            q.update(cont)
            got = _get(WIKI_API + "?" + urllib.parse.urlencode(q))
            if not got:
                break
            for m in got.get("query", {}).get("categorymembers", []):
                if m["ns"] == 14 and d > 0:
                    todo.append((m["title"].split(":", 1)[-1], d - 1))
                elif m["ns"] == 0:
                    out.append(m["title"])
            cont = got.get("continue", {})
            if not cont:
                break
            time.sleep(0.8)
        time.sleep(0.8)
    return out


def _pick(entity: dict, field: str) -> str:
    """按 zh-hans → zh-cn → zh 取标签/描述；没有中文就返回空串。"""
    d = entity.get(field, {})
    for lang in ("zh-hans", "zh-cn", "zh"):
        if lang in d:
            return d[lang]["value"]
    return ""


def entities_for_titles(titles: list[str], batch: int = 50) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(titles), batch):
        q = {
            "action": "wbgetentities",
            "sites": "zhwiki",
            "titles": "|".join(titles[i : i + batch]),
            "props": "labels|descriptions|claims",
            "languages": "zh-hans|zh-cn|zh",
            "format": "json",
        }
        got = _get(WD_API + "?" + urllib.parse.urlencode(q))
        if not got:
            continue
        for qid, e in got.get("entities", {}).items():
            if qid.startswith("-"):
                continue
            p31 = []
            for c in e.get("claims", {}).get("P31", [])[:3]:
                with contextlib.suppress(KeyError, TypeError):
                    p31.append(c["mainsnak"]["datavalue"]["value"]["id"])
            out[qid] = {
                "label": _pick(e, "labels"),
                "desc": _pick(e, "descriptions"),
                "p31": p31,
            }
        time.sleep(0.8)
    return out


def type_labels(qids: list[str], batch: int = 50) -> dict[str, str]:
    out: dict[str, str] = {}
    qids = sorted(set(qids))
    for i in range(0, len(qids), batch):
        q = {
            "action": "wbgetentities",
            "ids": "|".join(qids[i : i + batch]),
            "props": "labels",
            "languages": "zh-hans|zh-cn|zh",
            "format": "json",
        }
        got = _get(WD_API + "?" + urllib.parse.urlencode(q))
        if not got:
            continue
        for qid, e in got.get("entities", {}).items():
            for lang in ("zh-hans", "zh-cn", "zh"):
                if lang in e.get("labels", {}):
                    out[qid] = e["labels"][lang]["value"]
                    break
        time.sleep(0.8)
    return out


def load_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                with contextlib.suppress(json.JSONDecodeError):
                    done.add(json.loads(line)["qid"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--max-cats", type=int, default=1500, help="全局最多查多少个分类")
    ap.add_argument("--max-titles", type=int, default=400_000)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(args.out) if args.resume else set()
    print(f"已有 {len(done)} 条", flush=True)

    titles: list[str] = []
    seen: set[str] = set()
    budget = [args.max_cats]
    for seed in SEEDS:
        got = category_titles(seed, args.depth, budget, args.max_titles)
        if budget[0] <= 0:
            print(f"  分类预算用尽（{args.max_cats}），停止列标题", flush=True)
        new = [t for t in got if t not in seen]
        seen.update(new)
        titles.extend(new)
        print(f"  [{seed}] +{len(new)}（标题累计 {len(titles)}）", flush=True)
        if len(titles) >= args.limit * 3 or budget[0] <= 0:
            break

    written = len(done)
    for i in range(0, len(titles), 50):
        if written >= args.limit:
            break
        ents = entities_for_titles(titles[i : i + 50])
        ents = {q: e for q, e in ents.items() if q not in done}
        tl = type_labels([q for e in ents.values() for q in e["p31"]])
        rows = []
        for qid, e in ents.items():
            label = e["label"] or ""
            desc = e["desc"] or ""
            if not label or len(label) > 24:
                continue
            if desc and any(b in desc for b in BAD):
                desc = ""
            types = [tl[q] for q in e["p31"] if q in tl]
            if not desc and not types:
                continue
            rows.append(
                {
                    "qid": qid,
                    "label": label,
                    "desc": desc,
                    "types": types,
                    "license": "CC0",
                    "source_url": f"https://www.wikidata.org/wiki/{qid}",
                }
            )
        if rows:
            with args.out.open("a", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += len(rows)
        if (i // 50) % 10 == 0:
            print(f"  进度 {i}/{len(titles)}  已写 {written}", flush=True)

    allrows = []
    with args.out.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                allrows.append(json.loads(line))
    n_desc = sum(1 for r in allrows if r["desc"])
    n_type = sum(1 for r in allrows if r["types"])
    print(f"共 {len(allrows)} 条 → {args.out}")
    print(f"  有中文描述 {n_desc}（{n_desc / max(len(allrows), 1):.0%}）")
    print(f"  有中文 P31  {n_type}（{n_type / max(len(allrows), 1):.0%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
