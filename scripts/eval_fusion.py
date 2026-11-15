"""融合模型的**五臂对照**：两层融合各测一遍，再看叠加。

## 五臂

| 臂 | SNN 怎么参与 | 说明 |
| :--- | :--- | :--- |
| `llm` | 不参与 | 只有人格权重（现有 V1） |
| `film_hand` | 状态 → 手写 12 维 `p` → FiLM | 上一版的做法 |
| `film_core` | 状态 → **0.5B 脉冲核心**算出的 `p`（**固定随机投影**）→ FiLM | 第一版接法 |
| `film_snn` | 状态 → **SNN 原始 15,624 维特征** → **可学投影** → FiLM | 修正版：投影跟着调制器一起训 |
| `dist` | `log q + α·log p_snn` | 分布级融合（32k 子集上对齐） |
| `film+dist` | 两层同时 | 叠加 |

## 两条度量

1. **状态题**（`state_qa` 那四问）：答案的关键事实是否跟着状态变。判据是命中率。
2. **留出文本困惑度**：中文维基条目首段（**不在训练集里**，训练用的是自写定义）。
   分布级融合有没有用，最终要看它在真文本上降不降困惑度。

## 判据（先落盘）

- **F-1**：`film_core` 的状态命中率 **≥ `film_hand`**（SNN 算的 p 不差于手写的）；
- **F-2**：`dist` 在某个 α>0 上的**留出困惑度低于 α=0**（分布级融合有正贡献）；
- **F-3**：`film+dist` 的两项都**不低于**各自单臂的较好者（叠加不互相拆台）；
- **F-4**：`film_snn`（可学投影）**≥ `film_hand`**。
  **F-1 用固定随机投影，测的是"随机投影能不能当 p"，不是"SNN 能不能当 p"。**
  随机投影保距离但不保语义——它不知道哪几个方向对应轮次/在线/条数。
  F-1 不过时的正确反应是**把投影改成可学**，不是宣布 SNN 不行。

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


def load_subset() -> tuple[torch.Tensor, torch.Tensor]:
    import json as _j

    p = Path("D:/Projects/Semprea-4.5B/subset_32k.json")
    if not p.is_file():
        p = Path("sweep_results/lm_semprea_v1/id_map.json")
    d = _j.loads(p.read_text(encoding="utf-8"))
    ids = d.get("ids", d)
    sub = torch.tensor(ids, dtype=torch.long)
    # 反查表：千问 id → 本地 id（不在子集里的是 -1）。**目标 token 必须换算**，
    # 否则 gather 会用 151k 的 id 去索引 32k 的列，直接踩爆显存边界。
    q2l = torch.full((int(sub.max()) + 2,), -1, dtype=torch.long)
    q2l[sub] = torch.arange(sub.numel())
    return sub, q2l


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=2000, help="调制器训练用的行数（与评测集分开）")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or Path(args.ckpt).name
    # **必须播种**：不播的话调制器每次初始化都不同，同一份代码两次跑出 0.233 / 0.117，
    # 那不是"融合效果不稳"，是仪器没定住。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from research.sel.lm.fusion_model import load_fusion

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    fusion = load_fusion(args.ckpt, state_dim=12, alpha=0.1, rank=8, device=dev)
    snn = SNNFull()  # bf16 树

    rows = build(args.eval, seed=99)  # 评测集 60 条
    tr_rows = build(args.train, seed=1)  # 训练集另取，**不与评测集重叠**
    texts = heldout_texts(args.texts)
    subset, q2l = load_subset()
    res: dict = {"ckpt": args.ckpt, "n_state": len(rows), "n_text": len(texts)}

    # ---------- 两种 p ----------
    core = SNNCoreState()
    p_core = core.encode([r["system"] for r in rows], max_len=48)
    p_hand = [r["p"] for r in rows]
    p_core_tr = core.encode([r["system"] for r in tr_rows], max_len=48)
    p_hand_tr = [r["p"] for r in tr_rows]

    # ---------- 训练两个调制器（只在 hand / core 各自的 p 上） ----------
    def train_mod(p_list, tag: str):
        opt = torch.optim.AdamW(list(fusion.trainable_parameters()), lr=args.lr)
        fusion.train()
        rng = random.Random(0)
        for step in range(1, args.steps + 1):
            idx = rng.randrange(len(tr_rows))
            r = tr_rows[idx]
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

    # ---------- SNN 原始特征 → **可学**投影 → p ----------
    # 固定随机投影保距离但不保"语义"：它不知道哪几个方向对应轮次/在线/条数。
    # 让 15624→12 这一层跟着调制器一起学，才是把 SNN 接进 FiLM 的正确方式。
    f_snn = snn.features([r["system"] for r in tr_rows], max_len=48).float()
    proj = torch.nn.Linear(f_snn.shape[1], 12).to(dev)
    torch.nn.init.zeros_(proj.bias)

    def train_mod_snn(tag: str):
        params = list(fusion.trainable_parameters()) + list(proj.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr)
        fusion.train()
        rng = random.Random(0)
        for step in range(1, args.steps + 1):
            idx = rng.randrange(len(tr_rows))
            r = tr_rows[idx]
            msgs = []
            if rng.random() < 0.5:
                msgs.append({"role": "system", "content": r["system"]})
            msgs += [
                {"role": "user", "content": r["q"]},
                {"role": "assistant", "content": r["a"]},
            ]
            text = render(tok, msgs, add_generation_prompt=False)
            b = tok(text, return_tensors="pt").input_ids.to(dev)
            feat = f_snn[idx : idx + 1].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = fusion(b, p=proj(feat), labels=b.clone()).loss
            loss.backward()
            opt.step()
            if step % 200 == 0:
                print(f"  [{tag}] step {step} loss {float(loss.detach()):.4f}", flush=True)
        fusion.eval()
        return {k: v.detach().clone() for k, v in fusion.modulator.state_dict().items()}

    sd_hand = train_mod(p_hand_tr, "hand")
    sd_core = train_mod([list(map(float, x)) for x in p_core_tr], "core")
    sd_snn = train_mod_snn("snn")

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

    # film_snn：SNN 原始特征经**可学投影**进 FiLM
    f_snn_eval = snn.features([r["system"] for r in rows], max_len=48).float()
    fusion.modulator.load_state_dict(sd_snn)
    answers = []
    for i, r in enumerate(rows):
        text = render(tok, [{"role": "user", "content": r["q"]}])
        b = tok(text, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            fusion.set_state(proj(f_snn_eval[i : i + 1].to(dev)))
            try:
                o = fusion.lm.generate(b, max_new_tokens=40, do_sample=False)
            finally:
                fusion.set_state(None)
        answers.append(tok.decode(o[0][b.shape[1] :], skip_special_tokens=True).strip())
    res["film_snn"] = _score(rows, answers, None)
    print(f"  film_snn  {res['film_snn']:.3f}", flush=True)

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
        tgt_local = q2l.to(tgt.device)[tgt.reshape(-1).clamp(max=q2l.numel() - 1)]
        keep = tgt_local >= 0
        if not bool(keep.any()):
            continue
        idx = tgt_local[keep]
        flat_q = lq.reshape(-1, lq.shape[-1])[keep]
        flat_f = lp_fast.reshape(-1, lp_fast.shape[-1])[keep]
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
