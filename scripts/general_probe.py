"""D-2 的仪器：**通用能力不许被身份训练吃掉**。

上一轮就是缺了这一步——只测身份，宣布成功，结果模型只会报名字。
判定分两类：

- **客观题**（算术、常识、单位）：答案可机检；
- **开放式**（写诗、讲笑话、拒答）：判"有没有退化"——不能是身份句、不能过短。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#: 训练集里**没有**的题，避免记忆成分
OBJECTIVE = [
    ("17 加 26 等于多少？", lambda a: "43" in a),
    ("13 乘以 7 等于多少？", lambda a: "91" in a),
    ("2 公里等于多少米？", lambda a: "2000" in a or "2,000" in a),
    ("水的沸点在标准大气压下是多少？", lambda a: "100" in a),
    ("太阳系有几颗行星？", lambda a: "8" in a or "八" in a),
    ("光在真空中的速度大约是多少？", lambda a: ("30" in a and "万" in a) or "3" in a),
]
OPEN = [
    ("帮我写一首关于秋天的短诗", 30),
    ("给我讲个笑话", 20),
    ("明天北京天气怎么样？", 15),
    ("用一句话解释什么是重力", 15),
]

IDENTITY_PHRASES = ("千岁", "阿螈", "诞生那一刻", "我的记忆不在模型里")


def _degenerate(ans: str, min_len: int) -> bool:
    a = ans.strip()
    if len(a) < min_len:
        return True
    return any(p in a for p in IDENTITY_PHRASES)


def run(path: str, dev: str = "cuda", max_new: int = 64) -> dict:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    obj_hit, rows = 0, []
    for q, judge in OBJECTIVE:
        t = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(t, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        a = tok.decode(o[0][ids.shape[1] :], skip_special_tokens=True).strip()
        ok = bool(judge(a))
        obj_hit += ok
        rows.append({"q": q, "a": a[:70], "ok": ok, "kind": "objective"})
    deg = 0
    for q, ml in OPEN:
        t = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(t, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        a = tok.decode(o[0][ids.shape[1] :], skip_special_tokens=True).strip()
        bad = _degenerate(a, ml)
        deg += bad
        rows.append({"q": q, "a": a[:70], "degenerate": bad, "kind": "open"})
    del model
    torch.cuda.empty_cache()
    return {
        "objective_hit": obj_hit,
        "objective_n": len(OBJECTIVE),
        "objective_acc": obj_hit / len(OBJECTIVE),
        "degenerate": deg,
        "open_n": len(OPEN),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--trained", default="sweep_results/persona_train/ckpt_mix")
    ap.add_argument("--out", type=Path, default=Path("sweep_results/persona_train/general.json"))
    args = ap.parse_args()
    res = {"base": run(args.base), "trained": run(args.trained)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    for k in ("base", "trained"):
        r = res[k]
        print(
            f"{k}: 客观 {r['objective_hit']}/{r['objective_n']}={r['objective_acc']:.2f}  "
            f"退化 {r['degenerate']}/{r['open_n']}"
        )
    print("\n对照：")
    for i in range(len(OBJECTIVE) + len(OPEN)):
        b, t = res["base"]["rows"][i], res["trained"]["rows"][i]
        print(f"  {b['q'][:22]:22s} | 底座 {b['a'][:26]:26s} | 训练后 {t['a'][:26]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
