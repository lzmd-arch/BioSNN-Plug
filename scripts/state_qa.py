"""把状态接进人格核的三件套：数据、文字通道、FiLM 通道。

四条臂，同一批状态与问题：

| 臂 | 状态从哪进 | 要不要训练 |
| :--- | :--- | :--- |
| `none` | 不进 | — |
| `text` | system 消息里的状态块 | **不训**（测底座会不会读） |
| `film` | 12 维向量 `p` → FiLM 调制每层 | 只训 20.5k 参数 |
| `both` | 文字 + `p` | 只训调制器 |

判据（I-5 的具体化）：**答案必须与状态一致**。状态在每条样本里随机，背不下来。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

BASE = "D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct"
OUT = Path("sweep_results/state_qa")
BACKBONES = ["Qwen3-4B", "DeepSeek", "Qwen2.5-0.5B", "本地小模型"]


def sample(rng: random.Random) -> dict:
    bb = rng.randrange(len(BACKBONES))
    mem = rng.randint(0, 500)
    online = rng.random() < 0.5
    turns = rng.randint(0, 40)
    qs = [
        ("你现在用哪个模型做嘴巴？", f"我现在用「{BACKBONES[bb]}」做嘴巴。"),
        ("你记得多少条？", f"我的记忆库里现在有 {mem} 条。"),
        ("你现在在线吗？", "在，云端可以用。" if online else "不在，我现在是离线状态。"),
        ("这次对话几轮了？", f"这次对话已经进行了 {turns} 轮。"),
    ]
    kind = rng.randrange(len(qs))
    q, a = qs[kind]
    block = (
        f"我是千岁（阿螈），现在 20 岁。"
        f"我现在用「{BACKBONES[bb]}」做嘴巴；我的记忆库里现在有 {mem} 条。"
        + ("云端可以用。" if online else "我现在是离线的，回答能力受限。")
        + (f"这次对话已经进行了 {turns} 轮。" if turns else "")
    )
    vec = [
        min(1.0, turns / 50.0),
        1.0 if online else 0.0,
        min(1.0, __import__("math").log1p(mem) / 12.0),
        *[1.0 if k == bb else 0.0 for k in range(len(BACKBONES))],
        0.2,
        0.4,
        1.0,
        min(1.0, __import__("math").log1p(mem) / 12.0),
        0.5,
    ]
    key = {0: BACKBONES[bb], 1: str(mem), 2: "离线" if not online else "在线", 3: str(turns)}[kind]
    return {
        "q": q,
        "a": a,
        "system": block,
        "p": vec,
        "kind": kind,
        "key": key,
        "mem": mem,
        "online": online,
        "bb": bb,
        "turns": turns,
    }


def build(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [sample(rng) for _ in range(n)]


def _ask(tok, model, rows, arm: str, dev: str, max_new: int = 40) -> list[str]:
    out = []
    for r in rows:
        msgs = []
        if arm in ("text", "both"):
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["q"]})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        p = (
            torch.tensor([r["p"]], dtype=torch.float32, device=dev)
            if arm in ("film", "both")
            else None
        )
        with torch.no_grad():
            # 生成时**设一次状态**，hook 在每次前向里都会读它
            model.set_state(p)
            try:
                o = model.lm.generate(ids, max_new_tokens=max_new)
            finally:
                model.set_state(None)
        out.append(tok.decode(o[0][ids.shape[1] :], skip_special_tokens=True).strip())
    return out


def _score(rows, answers, _qtype) -> float:
    """只看**关键事实**在不在答案里，不比较措辞。

    严格逐字比对会把"读懂了但换了说法"误判成失败——上一版就是这样只拿到 0.23。
    """
    hit = 0
    for r, a in zip(rows, answers, strict=False):
        k = r["key"]
        if r["kind"] == 2:  # 在线/离线：两侧措辞都要认
            ok = ("离线" in a or "不在" in a) if k == "离线" else ("在线" in a or "在，" in a)
        else:
            ok = k in a
        hit += ok
    return hit / max(len(rows), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", type=Path, default=OUT / "modulator.pt")
    ap.add_argument("--base", default=BASE, help="被调制的底座（默认原始 Qwen2.5）")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--p-source",
        choices=("hand", "snn", "core"),
        default="hand",
        help="hand = 手写 12 维；snn = 16 神经元小池；core = 0.5B 脉冲核心",
    )
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from research.sel.lm.fusion_model import load_fusion

    tok = AutoTokenizer.from_pretrained(args.base)
    fusion = load_fusion(args.base, state_dim=12, alpha=args.alpha, rank=args.rank)
    model = fusion

    te = build(args.eval, seed=99)
    tr_all = build(args.train, seed=1)
    if args.p_source == "core":
        from scripts.snn_core_state import SNNCoreState

        enc = SNNCoreState()
        for rows in (te, tr_all):
            # `system` 就是那段状态描述句，直接喂给 0.5B 核心
            ps = enc.encode([r["system"] for r in rows], max_len=48)
            for r, v in zip(rows, ps, strict=False):
                r["p"] = [float(x) for x in v]
        print("p 来源：0.5B 脉冲核心（不含输出树）", flush=True)
    elif args.p_source == "snn":
        from scripts.snn_state_p import SCNNState

        enc = SCNNState()
        for rows in (te, tr_all):
            ps = enc.encode(
                [
                    {"mem": r["mem"], "online": r["online"], "bb": r["bb"], "turns": r["turns"]}
                    for r in rows
                ]
            )
            for r, v in zip(rows, ps, strict=False):
                r["p"] = [float(x) for x in v]
        print("p 来源：SNN 池", flush=True)
    res = {
        "none": _score(te, _ask(tok, model, te, "none", dev), None),
        "text": _score(te, _ask(tok, model, te, "text", dev), None),
    }
    print(f"未训练：none {res['none']:.2f}  text {res['text']:.2f}", flush=True)

    if not args.skip_train:
        tr = tr_all
        opt = torch.optim.AdamW(list(fusion.trainable_parameters()), lr=args.lr)
        fusion.train()
        rng = random.Random(0)
        for step in range(1, args.steps + 1):
            r = tr[rng.randrange(len(tr))]
            # **训练时把文字随机放/不放**：否则推理时加文字对调制器是分布外输入
            # （上一轮 both 0.383 < film 0.417 就是这个原因）。
            msgs = []
            if rng.random() < 0.5:
                msgs.append({"role": "system", "content": r["system"]})
            msgs += [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}]
            text = tok.apply_chat_template(msgs, tokenize=False)
            ids = tok(text, return_tensors="pt").input_ids.to(dev)
            p = torch.tensor([r["p"]], dtype=torch.float32, device=dev)
            labels = ids.clone()
            opt.zero_grad(set_to_none=True)
            loss = fusion(ids, p=p, labels=labels).loss
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"  step {step} loss {float(loss):.4f}", flush=True)
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fusion.modulator.state_dict(), args.save)
        fusion.eval()
        res["film"] = _score(te, _ask(tok, fusion, te, "film", dev), None)
        res["both"] = _score(te, _ask(tok, fusion, te, "both", dev), None)
        print(f"训练后：film {res['film']:.2f}  both {res['both']:.2f}", flush=True)

    (OUT / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
