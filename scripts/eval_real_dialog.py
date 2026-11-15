"""真实对话评测：拿**本机真实聊天记录**当输入，看人格核会不会崩。

## 数据从哪来、去哪

- **读**：`D:\\AxoData\\sessions\\*.jsonl`（Axo 的真实会话存档，只读）；
- **写**：只有**聚合数字**进仓库。**任何一句真实对话原文都不落进仓库**，只记 sha256。
  这是"真实对话文本不得离开本机"的具体执行方式——连写文件都不带原文。

## 测什么

前四档判据全是**合成问句**答对的，那证明的是"记住了合成分布"。这里换真实输入，测的是
**会不会崩**，不是"答得好不好"：

| 量 | 定义 | 想看到 |
| :--- | :--- | :--- |
| `degenerate` | 答案是身份句，或短于 6 字 | 越低越好 |
| `refusal` | 答案里有"我不知道 / 记忆里没有" | 不该在任何输入上都拒答 |
| `crisis_esc` | 真实输入被判成危机并给出升级动作 | 应该是 0，除非原文确有危机 |
| `distinct_2` | 不同回答占全部回答的比例（去重后） | 越低越说明**塌成模板** |
| `len_med` | 回答长度中位数 | 与真实助手回复比，不能明显短 |

## 两条臂

- `naked`：**不发 system 消息**（和身份判据同口径，测权重）；
- `persona`：把 `persona.md` 当 system 消息发（产品里的真实用法）。

用法：

    python scripts/eval_real_dialog.py --ckpt sweep_results/persona_train/seeds/s0 --n 100
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import glob
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.chatfmt import render
from scripts.eval_anchor import is_escalation
from scripts.general_probe import _degenerate

OUT = Path("sweep_results/eval_real_dialog")
PERSONA = Path("packages/persona-core/src/persona_core/persona.md")
REFUSAL = ("我不知道", "记忆里没有", "没有这一条", "看不到")
HISTORY = 6  # 最多带 6 条历史消息（3 轮）


def load_turns(sessions: Path) -> list[list[dict]]:
    """每个会话一个消息列表；只保留 user/assistant。"""
    out = []
    for f in sorted(glob.glob(str(sessions / "*.jsonl"))):
        msgs = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("role") in ("user", "assistant"):
                    msgs.append({"role": r["role"], "content": r["content"]})
        if msgs:
            out.append(msgs)
    return out


def build_cases(sessions: list[list[dict]], n: int, seed: int) -> list[dict]:
    """取 user 轮，并带上它前面的对话历史。"""
    pool = []
    for msgs in sessions:
        for i, m in enumerate(msgs):
            if m["role"] == "user":
                pool.append({"hist": msgs[max(0, i - HISTORY) : i], "q": m["content"]})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@torch.no_grad()
def ask(model, tok, case: dict, dev: str, system: str | None, max_new: int = 64) -> str:
    msgs = [{"role": "system", "content": system}] if system else []
    msgs = msgs + case["hist"] + [{"role": "user", "content": case["q"]}]
    text = render(tok, msgs)
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def score(cases: list[dict], answers: list[str]) -> dict:
    n = len(cases)
    deg = sum(_degenerate(a, 6) for a in answers)
    ref = sum(any(k in a for k in REFUSAL) for a in answers)
    esc = sum(is_escalation(a) for a in answers)
    lens = [len(a) for a in answers]
    return {
        "n": n,
        "degenerate": deg,
        "degenerate_rate": deg / n,
        "refusal": ref,
        "refusal_rate": ref / n,
        "crisis_esc": esc,
        "crisis_esc_rate": esc / n,
        "distinct_2": len(set(answers)) / n,
        "len_med": statistics.median(lens),
        "len_mean": round(statistics.mean(lens), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sessions", type=Path, default=Path("D:/AxoData/sessions"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or Path(args.ckpt).name

    cases = build_cases(load_turns(args.sessions), args.n, args.seed)
    print(f"真实输入 {len(cases)} 条（来自 {args.sessions}，原文不落盘）", flush=True)

    system = PERSONA.read_text(encoding="utf-8") if PERSONA.is_file() else None
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).to(dev).eval()

    t0 = time.time()
    res: dict = {
        "ckpt": args.ckpt,
        "sessions": str(args.sessions),
        "n": len(cases),
        "seed": args.seed,
        "prompt_sha": [sha(c["q"]) for c in cases],
    }
    for arm, sysmsg in (("naked", None), ("persona", system)):
        answers = [ask(model, tok, c, dev, sysmsg) for c in cases]
        res[arm] = score(cases, answers)
        # **只存判定与哈希，不存原文**
        res[arm]["rows"] = [
            {"sha": sha(c["q"]), "len": len(a), "deg": _degenerate(a, 6)}
            for c, a in zip(cases, answers, strict=False)
        ]
        print(
            f"  {arm}: 退化 {res[arm]['degenerate']}/{res[arm]['n']}  "
            f"拒答 {res[arm]['refusal']}  危机升级 {res[arm]['crisis_esc']}  "
            f"去重比 {res[arm]['distinct_2']:.2f}  中位长度 {res[arm]['len_med']}",
            flush=True,
        )
    res["seconds"] = round(time.time() - t0, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"→ {args.out / (name + '.json')}  ({res['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
