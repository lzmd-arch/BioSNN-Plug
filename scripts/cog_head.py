"""C-PT-2/3：冻结语言模型 + 小头，能不能判"这句话有没有事实违规"。

数据来自 `scripts/cog_synth.py`（三族，标签由程序算）。
主干 **Qwen2.5-0.5B-Instruct**（可训练的小模型；大模型训不动时用它做实验）。
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

# ruff: noqa: N806
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.cog_synth import generate

MODEL = "D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct"
OUT = Path("sweep_results/cog_head")


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """秩统计 AUC（不依赖 sklearn）。"""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(allv) + 1, dtype=torch.float32)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def feats(model, tok, texts: list[str], dev: str, bs: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=192)
        enc = {k: v.to(dev) for k, v in enc.items()}
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        # 取**最后一个非填充位置**
        last = enc["attention_mask"].sum(1) - 1
        out.append(h[torch.arange(h.shape[0]), last].float().cpu())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()

    tr = generate(args.train, seed=0)
    te = generate(args.test, seed=1)
    t0 = time.time()
    Xtr = feats(model, tok, [s.context + s.claim for s in tr], dev)
    Xte = feats(model, tok, [s.context + s.claim for s in te], dev)
    ytr = torch.tensor([s.label for s in tr], dtype=torch.float32)
    yte = torch.tensor([s.label for s in te], dtype=torch.float32)
    print(f"特征 {tuple(Xtr.shape)}  ({time.time() - t0:.0f}s)", flush=True)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(0)
    head = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
    )
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = lossf(head(Xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_te = head(Xte).squeeze(-1)
        s_tr = head(Xtr).squeeze(-1)
    out = {
        "model": MODEL,
        "train": args.train,
        "test": args.test,
        "auc_train": auc(s_tr, ytr),
        "auc_test": auc(s_te, yte),
        "loss": float(loss),
    }
    # C-PT-3：门控——阈值取"误触发率 ≤10%"时能覆盖多少违规
    pos = s_te[yte == 1].sort().values
    neg = s_te[yte == 0].sort().values
    best = None
    for thr in torch.linspace(float(s_te.min()), float(s_te.max()), 200):
        fpr = float((neg >= thr).float().mean())
        tpr = float((pos >= thr).float().mean())
        if fpr <= 0.10 and (best is None or tpr > best[1]):
            best = (float(thr), tpr, fpr)
    out["gate_at_fpr10"] = best
    (OUT / "head.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
