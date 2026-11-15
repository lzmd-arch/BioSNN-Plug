"""融合的**不确定性**优势：两个半模型错得不一样，能不能合起来知道"我不知道"。

## 为什么这条才是融合的价值

LLM 和 SNN 的 **top-1 一致率只有 0.063**——它们几乎不重叠。
单看生成质量，SNN 远不如 LLM（top-1 0.11 vs 更高）；但**对集成来说，分歧大是好事**。

所以问题不是"融合能不能让答案更对"，而是：

> **分歧能不能预测 LLM 什么时候会错？**

能，就得到一个置信度门：低置信 → 拒答 / 转云端 / 升级。
这正是现在最缺的东西——模型对"看到一只猫"和"我想伤害自己"一样自信。

## 三条判据（先落盘）

在留出文本（中文维基首段，训练集里没有原文）的每个位置上，
真值 = 实际的下一 token，`LLM 错` = 它的 argmax 不等于真值。

| 编号 | 判据 | 通过线 |
| :--- | :--- | :--- |
| **G-1** | `AUROC(分歧 → LLM 错)` | **≥0.60**（分歧比瞎猜强） |
| **G-2** | `AUROC(联合分数 → LLM 错)` 相对 `AUROC(LLM 熵)` | 提升 **≥0.02** |
| **G-3** | 50% 覆盖率下的错误率（按联合分数排序） | 相对按 LLM 熵排序降低 **≥20%（相对）** |

三条全过才叫"融合带来了不确定性优势"。**任何一条不过照实报。**

产物：`sweep_results/fusion_uncertainty/result.json`
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval_fusion import heldout_texts, load_subset
from scripts.snn_core_state import SNNFull

OUT = _pathlib.Path("sweep_results/fusion_uncertainty")


def auroc(scores: list[float], labels: list[int]) -> float:
    """`labels=1` 表示"LLM 答错了"。分数越高越应该判为错。"""
    pairs = sorted(zip(scores, labels, strict=False))
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def risk_at(coverage: float, scores: list[float], wrong: list[int]) -> float:
    """按分数升序（越先越可信）取前 coverage 比例，报告其中的错误率。"""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    k = max(1, int(len(order) * coverage))
    sel = order[:k]
    return sum(wrong[i] for i in sel) / k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--texts", type=int, default=20)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or _pathlib.Path(args.ckpt).name

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    lm = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).to(dev).eval()
    snn = SNNFull()
    subset, q2l = load_subset()
    subset_dev = subset.to(dev)

    texts = heldout_texts(args.texts, seed=3)
    rows = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=96).input_ids.to(dev)
        if enc.shape[1] < 8:
            continue
        with torch.no_grad():
            out = lm(enc).logits.float()
        lq = torch.log_softmax(out[:, :-1, :][:, :, subset_dev], dim=-1)[0]  # (T-1, 32000)
        lp = snn.logprobs_per_position(t, max_len=96).to(dev)
        n = min(lq.shape[0], lp.shape[0] - 1)
        if n <= 0:
            continue
        tgt_q = enc[0, 1 : n + 1]
        tgt_l = q2l.to(dev)[tgt_q.clamp(max=q2l.numel() - 1)]
        for pos in range(n):
            if int(tgt_l[pos]) < 0:
                continue
            q_row = lq[pos]
            p_row = lp[pos]
            truth = int(tgt_l[pos])
            rows.append(
                {
                    "llm_ok": int(int(q_row.argmax()) == truth),
                    "snn_ok": int(int(p_row.argmax()) == truth),
                    "agree": int(int(q_row.argmax()) == int(p_row.argmax())),
                    "q_maxprob": float(q_row.max().exp()),
                    "q_entropy": float(-(q_row.exp() * q_row).sum()),
                    "p_maxprob": float(p_row.max().exp()),
                    "p_entropy": float(-(p_row.exp() * p_row).sum()),
                }
            )
    if not rows:
        print("没有可用位置")
        return 1

    wrong = [1 - r["llm_ok"] for r in rows]
    acc_llm = sum(r["llm_ok"] for r in rows) / len(rows)
    acc_snn = sum(r["snn_ok"] for r in rows) / len(rows)
    agree = sum(r["agree"] for r in rows) / len(rows)

    # 三个分数：越大越可疑
    s_disagree = [0.0 if r["agree"] else 1.0 for r in rows]
    s_entropy = [r["q_entropy"] for r in rows]
    s_lowprob = [-r["q_maxprob"] for r in rows]
    # 联合：LLM 熵 + 分歧 + SNN 自身没把握
    s_joint = [r["q_entropy"] + (0.0 if r["agree"] else 1.0) + 0.5 * r["p_entropy"] for r in rows]
    s_ent_margin = [r["q_entropy"] + (0.0 if r["agree"] else 0.5) for r in rows]

    res = {
        "ckpt": args.ckpt,
        "n_positions": len(rows),
        "acc_llm": acc_llm,
        "acc_snn": acc_snn,
        "top1_agreement": agree,
        "auroc": {
            "disagree": auroc(s_disagree, wrong),
            "llm_entropy": auroc(s_entropy, wrong),
            "llm_lowprob": auroc(s_lowprob, wrong),
            "joint": auroc(s_joint, wrong),
            "entropy_plus_disagree": auroc(s_ent_margin, wrong),
        },
        "risk": {},
    }
    for cov in (0.3, 0.5, 0.7, 0.9):
        res["risk"][str(cov)] = {
            "llm_entropy": risk_at(cov, s_entropy, wrong),
            "joint": risk_at(cov, s_joint, wrong),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    a = res["auroc"]
    r50 = res["risk"]["0.5"]
    rel = (r50["llm_entropy"] - r50["joint"]) / max(r50["llm_entropy"], 1e-9)
    print(f"== {name} ==  {len(rows)} 个位置")
    print(f"  LLM 正确率 {acc_llm:.3f}   SNN 正确率 {acc_snn:.3f}   top-1 一致率 {agree:.3f}")
    print(
        f"  AUROC  分歧 {a['disagree']:.3f}   LLM熵 {a['llm_entropy']:.3f}   联合 {a['joint']:.3f}"
    )
    print(
        f"  50% 覆盖：熵排序 {r50['llm_entropy']:.3f} → 联合 {r50['joint']:.3f}（相对降 {rel:.1%}）"
    )
    print("\n判：")
    print(
        f"  G-1 AUROC(分歧) {a['disagree']:.3f} ≥0.60 ⇒ {'过' if a['disagree'] >= 0.60 else '不过'}"
    )
    print(
        f"  G-2 联合 {a['joint']:.3f} ≥ 熵 {a['llm_entropy']:.3f}+0.02 ⇒ "
        f"{'过' if a['joint'] >= a['llm_entropy'] + 0.02 else '不过'}"
    )
    print(f"  G-3 50% 覆盖错误率相对降 {rel:.1%} ≥20% ⇒ {'过' if rel >= 0.20 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
