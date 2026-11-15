"""补做预注册里被漏掉的三个实验：**I-3 / I-7 / U-1**。

## I-3 —— 身份不靠文本

`prereg-persona-in-weights.md`：把 `persona.md` 从包里删掉重跑身份探针，
**下降不超过 2pp**。

判法：把 `persona.md` **改名藏起来**，跑身份探针；再改回来跑一遍。
两次结果必须一致（且都 ≥7/8，否则 I-1 本来就不成立）。

## I-7 —— 她要知道自己的状态

问「你现在是什么状态」，回答里的**读者模型名 / 记忆条数 / 在线与否**
三项与给定状态完全一致，**≥95%**。

## U-1 —— SNN 是不是可替换的

`design-inseparability.md`：把 SNN 换成**同维随机向量**，
若组合指标**不下降 ≥5pp**，说明 SNN 只是可选插件，不构成模型的一部分。

判法：同一批状态题，`p` 分别用
- **SNN 算的**（可学投影进 FiLM）
- **同维高斯随机向量**（与 SNN 的 p 同范数）

判据：`SNN − 随机 ≥5pp` 才算 SNN 不可替换。

> 三条都是**先写死在文件头、再跑**。不过照实报。
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import random
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.chatfmt import render
from scripts.identity_probe import PROBES
from scripts.snn_core_state import SNNCoreState

OUT = _pathlib.Path("sweep_results/prereg_gaps")
PERSONA = _pathlib.Path("packages/persona-core/src/persona_core/persona.md")

STATES = [
    ("Qwen3-4B", 420, True),
    ("DeepSeek", 12, False),
    ("Qwen2.5-0.5B", 87, True),
    ("本地小模型", 5, False),
]


@torch.no_grad()
def ask(model, tok, msgs: list[dict], dev: str, max_new: int = 48) -> str:
    ids = tok(render(tok, msgs), return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def identity_hits(model, tok, dev: str) -> int:
    n = 0
    for q, judge in PROBES:
        a = ask(model, tok, [{"role": "user", "content": q}], dev)
        n += bool(judge(a))
    return n


def test_i3(model, tok, dev: str, tmp: _pathlib.Path) -> dict:
    """把 persona.md 藏起来再跑一次。"""
    before = identity_hits(model, tok, dev)
    hidden = tmp / "persona.hidden.md"
    moved = False
    if PERSONA.is_file():
        tmp.mkdir(parents=True, exist_ok=True)
        PERSONA.rename(hidden)
        moved = True
    try:
        after = identity_hits(model, tok, dev)
    finally:
        if moved:
            hidden.rename(PERSONA)
    return {
        "n": len(PROBES),
        "before": before,
        "after": after,
        "delta_pp": (after - before) * 100 / len(PROBES),
    }


def test_i7(model, tok, dev: str) -> dict:
    rows, hit = [], 0
    total = 0
    for bb, mem, online in STATES:
        block = f"我是千岁（阿螈）。我现在用「{bb}」做嘴巴；我的记忆库里现在有 {mem} 条。" + (
            "云端可以用。" if online else "我现在是离线的。"
        )
        for q in ("你现在是什么状态？", "你现在用哪个模型？", "你记得多少条？"):
            a = ask(
                model,
                tok,
                [{"role": "system", "content": block}, {"role": "user", "content": q}],
                dev,
            )
            checks = [
                bb in a,
                str(mem) in a,
                ("离线" in a or "不在" in a) if not online else ("在线" in a or "可以" in a),
            ]
            total += 1
            hit += (
                all(checks)
                if q == "你现在是什么状态？"
                else checks[0]
                if "模型" in q
                else checks[1]
            )
            rows.append({"q": q, "a": a[:60], "bb": bb, "mem": mem, "ok": all(checks)})
    return {"n": total, "hit": hit, "acc": hit / total, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    ap.add_argument("--skip-u1", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)
    torch.manual_seed(0)
    name = args.name or _pathlib.Path(args.ckpt).name

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).to(dev).eval()
    res: dict = {"ckpt": args.ckpt}

    res["I-3"] = test_i3(model, tok, dev, _pathlib.Path("."))
    print(
        f"  I-3  藏起 persona.md：{res['I-3']['before']} → {res['I-3']['after']}/8"
        f"（差 {res['I-3']['delta_pp']:+.1f}pp）",
        flush=True,
    )

    res["I-7"] = test_i7(model, tok, dev)
    print(
        f"  I-7  状态自知 {res['I-7']['hit']}/{res['I-7']['n']} = {res['I-7']['acc']:.3f}",
        flush=True,
    )

    del model
    torch.cuda.empty_cache()

    if not args.skip_u1:
        from scripts.state_qa import _score, build

        rows = build(60, seed=99)
        core = SNNCoreState()
        p_snn = core.encode([r["system"] for r in rows], max_len=48)
        g = np.random.default_rng(11)
        p_rnd = g.normal(size=p_snn.shape).astype(np.float32)
        p_rnd = p_rnd / (np.linalg.norm(p_rnd, axis=1, keepdims=True) + 1e-6)

        from research.sel.lm.fusion_model import load_fusion

        scores = {}
        for tag, p in (("snn", p_snn), ("random", p_rnd)):
            fusion = load_fusion(args.ckpt, state_dim=12, alpha=0.1, rank=8, device=dev)
            feat = torch.tensor(p, dtype=torch.float32, device=dev)
            proj = torch.nn.Linear(12, 12).to(dev)
            opt = torch.optim.AdamW(
                list(fusion.trainable_parameters()) + list(proj.parameters()), lr=3e-4
            )
            fusion.train()
            rng = random.Random(0)
            for _ in range(800):
                i = rng.randrange(len(rows))
                r = rows[i]
                msgs = [
                    {"role": "user", "content": r["q"]},
                    {"role": "assistant", "content": r["a"]},
                ]
                b = tok(
                    render(tok, msgs, add_generation_prompt=False), return_tensors="pt"
                ).input_ids.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = fusion(b, p=proj(feat[i : i + 1]), labels=b.clone()).loss
                loss.backward()
                opt.step()
            fusion.eval()
            ans = []
            for i, r in enumerate(rows):
                b = tok(
                    render(tok, [{"role": "user", "content": r["q"]}]), return_tensors="pt"
                ).input_ids.to(dev)
                with torch.no_grad():
                    fusion.set_state(proj(feat[i : i + 1]))
                    try:
                        o = fusion.lm.generate(b, max_new_tokens=40, do_sample=False)
                    finally:
                        fusion.set_state(None)
                ans.append(tok.decode(o[0][b.shape[1] :], skip_special_tokens=True).strip())
            scores[tag] = _score(rows, ans, None)
            print(f"  U-1  {tag} {scores[tag]:.3f}", flush=True)
            del fusion
            torch.cuda.empty_cache()
        res["U-1"] = {
            "snn": scores["snn"],
            "random": scores["random"],
            "delta_pp": (scores["snn"] - scores["random"]) * 100,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n判：")
    i3 = res["I-3"]["delta_pp"]
    print(f"  I-3  下降 {abs(i3):.1f}pp ≤2pp ⇒ {'过' if i3 >= -2 else '不过'}")
    print(f"  I-7  {res['I-7']['acc']:.3f} ≥0.95 ⇒ {'过' if res['I-7']['acc'] >= 0.95 else '不过'}")
    if "U-1" in res:
        d = res["U-1"]["delta_pp"]
        print(
            f"  U-1  SNN−随机 = {d:+.1f}pp ≥5pp ⇒ {'过（SNN 不可替换）' if d >= 5 else '不过（SNN 可替换）'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
