"""融合模型的**五臂对照**：两层融合各测一遍，再看叠加。

## 五臂

| 臂 | SNN 怎么参与 | 说明 |
| :--- | :--- | :--- |
| `llm` | 不参与 | 只有人格权重（现有 V1） |
| `film_hand` | 状态 → 手写 12 维 `p` → FiLM | 上一版的做法 |
| `film_core` | 状态 → **0.5B 脉冲核心**算出的 `p` → FiLM | 本次新增，**真 SNN 参与特征级融合** |
| `dist` | `log q + α·log p_snn` | 分布级融合（32k 子集上对齐） |
| `film+dist` | 两层同时 | 叠加 |

## 两条度量

1. **状态题**（`state_qa` 那四问）：答案的关键事实是否跟着状态变。判据是命中率。
2. **留出文本困惑度**：中文维基条目首段（**不在训练集里**，训练用的是自写定义）。
   分布级融合有没有用，最终要看它在真文本上降不降困惑度。

## 判据（先落盘）

- **F-1**：`film_core` 的状态命中率 **≥ `film_hand`**（SNN 算的 p 不差于手写的）；
- **F-2**：`dist` 在某个 α>0 上的**留出困惑度低于 α=0**（分布级融合有正贡献）；
- **F-3**：`film+dist` 的两项都**不低于**各自单臂的较好者（叠加不互相拆台）。

**任何一条不过就照实报不过**，不调 α、不换判定词。过了才写进模型卡。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from scripts.chatfmt import render
from scripts.snn_core_state import SNNCoreState, SNNFull
from scripts.state_qa import _score, build

OUT = Path("sweep_results/eval_fusion")
WIKI = Path("sweep_results/kb_sources/wiki.jsonl")
ALPHAS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20)


def heldout_texts(n: int, seed: int = 0) -> list[str]:
    """留出文本：维基首段。**训练集里只有自写定义，没有这些原文。**"""
    rows = []
    for line in WIKI.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ok") and len(r["extract"]) > 120:
            rows.append(r["extract"][:400])
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_subset() -> torch.Tensor:
    import json as _j

    p = Path("D:/Projects/Semprea-4.5B/subset_32k.json")
    if not p.is_file():
        p = Path("sweep_results/lm_semprea_v1/id_map.json")
    d = _j.loads(p.read_text(encoding="utf-8"))
    ids = d.get("ids", d)
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def lm_logprobs(model, ids: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    """(B,T) → (B,T-1,32000) 子集上的 log-softmax。"""
    out = model(ids).logits.float()
    sub = out[:, :-1, :][:, :, subset.to(out.device)]
    return torch.log_softmax(sub, dim=-1), ids[:, 1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="人格权重（融合的慢半）")
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--texts", type=int, default=24)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or Path(args.ckpt).name

    from research.sel.lm.fusion_model import load_fusion

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    fusion = load_fusion(args.ckpt, state_dim=12, alpha=0.1, rank=8, device=dev)
    snn = SNNFull()  # bf16 树

    rows = build(args.eval, seed=99)
    texts = heldout_texts(args.texts)
    subset = load_subset()
    res: dict = {"ckpt": args.ckpt, "n_state": len(rows), "n_text": len(texts)}

    # ---------- 两种 p ----------
    core = SNNCoreState()
    p_core = core.encode([r["system"] for r in rows], max_len=48)
    p_hand = [r["p"] for r in rows]

    # ---------- 训练两个调制器（只在 hand / core 各自的 p 上） ----------
    def train_mod(p_list, tag: str):
        opt = torch.optim.AdamW(list(fusion.trainable_parameters()), lr=args.lr)
        fusion.train()
        rng = random.Random(0)
        for step in range(1, args.steps + 1):
            r = rows[rng.randrange(len(rows))]
            idx = rows.index(r)
            msgs = []
            if rng.random() < 0.5:
                msgs.append({"role": "system", "content": r["system"]})
            msgs += [
                {"role": "user", "content": r["q"]},
                {"role": "assistant", "content": r["a"]},
            ]
            text = render(tok, msgs, add_generation_prompt=False)
            b = tok(text, return_tensors="pt").input_ids.to(dev)
            p = torch.tensor([p_list[idx]], dtype=torch.float32, device=dev)
            opt.zero_grad(set_to_none=True)
            loss = fusion(b, p=p, labels=b.clone()).loss
            loss.backward()
            opt.step()
            if step % 200 == 0:
                print(f"  [{tag}] step {step} loss {float(loss):.4f}", flush=True)
        fusion.eval()
        return {k: v.detach().clone() for k, v in fusion.modulator.state_dict().items()}

    sd_hand = train_mod(p_hand, "hand")
    sd_core = train_mod([list(map(float, x)) for x in p_core], "core")

    # ---------- 臂 1/2/3：状态题 ----------
    def ask_arm(p_list, sd, use_film: bool) -> float:
        if sd is not None:
            fusion.modulator.load_state_dict(sd)
        answers = []
        for i, r in enumerate(rows):
            text = render(tok, [{"role": "user", "content": r["q"]}])
            b = tok(text, return_tensors="pt").input_ids.to(dev)
            p = torch.tensor([list(map(float, p_list[i]))], dtype=torch.float32, device=dev)
            with torch.no_grad():
                fusion.set_state(p if use_film else None)
                try:
                    o = fusion.lm.generate(b, max_new_tokens=40, do_sample=False)
                finally:
                    fusion.set_state(None)
            answers.append(tok.decode(o[0][b.shape[1] :], skip_special_tokens=True).strip())
        return _score(rows, answers, None)

    res["llm"] = ask_arm(p_hand, None, False)
    print(f"  llm       {res['llm']:.3f}", flush=True)
    res["film_hand"] = ask_arm(p_hand, sd_hand, True)
    print(f"  film_hand {res['film_hand']:.3f}", flush=True)
    res["film_core"] = ask_arm([list(map(float, x)) for x in p_core], sd_core, True)
    print(f"  film_core {res['film_core']:.3f}", flush=True)

    # ---------- 臂 4/5：留出文本困惑度 ----------
    fusion.set_state(None)
    nll = {a: [] for a in ALPHAS}
    agree = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=96).input_ids.to(dev)
        if enc.shape[1] < 8:
            continue
        lq, tgt = lm_logprobs(fusion.lm, enc, subset)  # (1, T-1, 32000)
        # SNN 的逐位置分布：一次前向全算完（逐前缀重跑会慢两个数量级）
        lp_fast = snn.logprobs_per_position(t, max_len=96).to(dev)
        n = min(lp_fast.shape[0], lq.shape[1])
        lq, tgt, lp_fast = lq[:, :n], tgt[:, :n], lp_fast[:n].unsqueeze(0)
        if n == 0:
            continue
        idx = tgt.reshape(-1)
        flat_q = lq.reshape(-1, lq.shape[-1])
        flat_f = lp_fast.reshape(-1, lp_fast.shape[-1])
        for a in ALPHAS:
            fused = torch.log_softmax(flat_q + a * flat_f, -1)
            nll[a].append(float(-fused.gather(1, idx.unsqueeze(-1)).mean()))
        agree.append(float((flat_q.argmax(-1) == flat_f.argmax(-1)).float().mean()))
    res["ppl"] = {str(a): math.exp(sum(v) / max(len(v), 1)) for a, v in nll.items()}
    res["top1_agree_with_snn"] = sum(agree) / max(len(agree), 1)
    print("  困惑度：" + "  ".join(f"a={a} {res['ppl'][str(a)]:.3f}" for a in ALPHAS), flush=True)
    print(f"  LLM 与 SNN 的 top-1 一致率 {res['top1_agree_with_snn']:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    best_a = min(ALPHAS, key=lambda a: res["ppl"][str(a)])
    print("\n判：")
    print(
        f"  F-1 film_core {res['film_core']:.3f} ≥ film_hand {res['film_hand']:.3f} ⇒ "
        f"{'过' if res['film_core'] >= res['film_hand'] else '不过'}"
    )
    print(
        f"  F-2 最佳 alpha={best_a} 困惑度 {res['ppl'][str(best_a)]:.3f} vs α=0 {res['ppl']['0.0']:.3f} ⇒ "
        f"{'过' if best_a > 0 and res['ppl'][str(best_a)] < res['ppl']['0.0'] else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
