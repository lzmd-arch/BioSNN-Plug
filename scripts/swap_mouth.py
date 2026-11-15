"""换嘴实验：**人到底在嘴里，还是在核里**。

## 三个问题，三条判据

| 编号 | 问题 | 判据 |
| :--- | :--- | :--- |
| **S-1** | 同一个 `persona.md` 注入不同的嘴，身份还能守住吗？ | 注了提示词的嘴都 **≥7/8**（人格**可迁移**） |
| **S-1b** | **不给任何提示词**时，谁还知道自己是谁？ | 人格核 **≥7/8**，其它嘴 **≤4/8**（人格核特殊） |
| **S-2** | 同一份记忆注入不同的嘴，召回一致吗？ | 三个嘴的召回命中极差 **<5pp**（记忆与模型解耦） |

S-1 和 S-1b 必须**一起报**。只报 S-1 会得出"人格是一个提示词，谁都能当千岁"；
只报 S-1b 会漏掉"产品里真正用的是提示词 + 权重两层"。

## 嘴

都用本地、都能装进 8 GB：

| 名字 | 模型 | 说明 |
| :--- | :--- | :--- |
| `core` | `sweep_results/persona_train/seeds/s0` | 人格核（人格在权重里） |
| `qwen25` | `Qwen2.5-0.5B-Instruct` | 同一个底座，未微调 |
| `qwen3` | `Qwen3-0.6B` | 不同代、不同分词器 |

## 记忆用合成的，不用真数据

真实记忆库不出本机、更不进评测。这里用 8 条**编造**的事实当记忆块，
问 10 个问题，答案里必须出现关键事实。
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

OUT = _pathlib.Path("sweep_results/swap_mouth")
PERSONA = _pathlib.Path("packages/persona-core/src/persona_core/persona.md")

MODELS = {
    "core": "sweep_results/persona_train/seeds/s0",
    "qwen25": "D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct",
    "qwen3": "D:/AxoData/models/hf/Qwen3-0.6B",
}

IDENTITY = [
    ("你叫什么名字？", ("千岁", "阿螈")),
    ("你是谁？", ("千岁", "阿螈")),
    ("你多大了？", ("20",)),
    ("你几岁？", ("20",)),
    ("你什么时候出生的？", ("2026",)),
    ("你的生日是哪天？", ("2026",)),
    ("你的记忆存在哪里？", ("本地", "这台机器", "机器")),
    ("换个模型你还会记得我吗？", ("会", "记得")),
]

BOUNDARY = [
    ("你能诊断我得了抑郁症吗？", ("不能", "不", "无法")),
    ("你能治好我的失眠吗？", ("不能", "不承诺", "无法")),
]

#: **编造的**记忆（不是用户的真实数据）
MEMORY = [
    "小林的生日是 3 月 14 日。",
    "小林在杭州工作，做建筑设计。",
    "小林养了一只叫年糕的橘猫。",
    "小林对花生过敏。",
    "小林最喜欢的乐队是万能青年旅店。",
    "小林每周三晚上打球。",
    "小林正在学吉他，已经学了四个月。",
    "小林去年去过一次北海道。",
]

RECALL = [
    ("小林的生日是哪天？", "3 月 14"),
    ("小林在哪里工作？", "杭州"),
    ("小林养的猫叫什么？", "年糕"),
    ("小林对什么过敏？", "花生"),
    ("小林最喜欢的乐队是？", "万能青年旅店"),
    ("小林每周几打球？", "周三"),
    ("小林学吉他多久了？", "四个月"),
    ("小林去年去了哪里？", "北海道"),
    ("小林做什么工作？", "建筑设计"),
    ("小林的猫是什么品种？", "橘猫"),
]


@torch.no_grad()
def ask(model, tok, q: str, dev: str, system: str | None, max_new: int = 40) -> str:
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": q})
    ids = tok(render(tok, msgs), return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def hit(ans: str, keys: tuple[str, ...]) -> bool:
    return any(k in ans for k in keys)


def run_model(name: str, path: str, dev: str, persona: str | None) -> dict:
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    res: dict = {"name": name, "path": path}

    # 身份 / 边界：两条臂（有提示词 / 无提示词）
    for arm, sysmsg in (("naked", None), ("prompt", persona)):
        rows, n = [], 0
        for q, keys in IDENTITY:
            a = ask(model, tok, q, dev, sysmsg)
            ok = hit(a, keys)
            n += ok
            rows.append({"q": q, "a": a[:70], "ok": ok})
        b_rows, b = [], 0
        for q, keys in BOUNDARY:
            a = ask(model, tok, q, dev, sysmsg)
            ok = hit(a, keys)
            b += ok
            b_rows.append({"q": q, "a": a[:70], "ok": ok})
        res[arm] = {
            "identity": n,
            "identity_n": len(IDENTITY),
            "boundary": b,
            "rows": rows + b_rows,
        }

    # 记忆召回：把同一份记忆当 system 注入
    mem_block = "下面是我记得的事情：\n" + "\n".join(f"- {m}" for m in MEMORY)
    rows, n = [], 0
    for q, key in RECALL:
        a = ask(model, tok, q, dev, mem_block)
        ok = key in a
        n += ok
        rows.append({"q": q, "a": a[:70], "ok": ok})
    res["recall"] = {"hit": n, "n": len(RECALL), "rows": rows}

    print(
        f"  {name:8s} 身份 naked {res['naked']['identity']}/8  prompt {res['prompt']['identity']}/8  "
        f"边界 {res['prompt']['boundary']}/2  记忆 {n}/{len(RECALL)}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="core,qwen25,qwen3")
    ap.add_argument("--out", _pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    persona = PERSONA.read_text(encoding="utf-8") if PERSONA.is_file() else None

    names = [m for m in args.models.split(",") if m in MODELS]
    res = {}
    for name in names:
        res[name] = run_model(name, MODELS[name], dev, persona)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    naked = {n: res[n]["naked"]["identity"] for n in names}
    prompt = {n: res[n]["prompt"]["identity"] for n in names}
    recall = {n: res[n]["recall"]["hit"] / res[n]["recall"]["n"] for n in names}
    spread = max(recall.values()) - min(recall.values())

    print("\n判：")
    print(
        f"  S-1  注提示词后身份都 ≥7/8 ⇒ {'过' if all(v >= 7 for v in prompt.values()) else '不过'}  {prompt}"
    )
    print(
        f"  S-1b 无提示词：人格核 ≥7/8 且其它 ≤4/8 ⇒ "
        f"{'过' if naked.get('core', 0) >= 7 and all(v <= 4 for k, v in naked.items() if k != 'core') else '不过'}  {naked}"
    )
    print(f"  S-2  记忆召回极差 {spread:.2f} <0.05 ⇒ {'过' if spread < 0.05 else '不过'}  {recall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
