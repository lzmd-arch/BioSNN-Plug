"""B0：同尺寸反向传播对照（判据见 `research/sel/prereg-lm-b0-backprop-control.md`）。

只回答一件事：这 3.4 万字的真语料，够不够任何一个 ~1.5M 参数的网络学到超过 bigram。
数据、分词、截断、折、数据暴露全部与 R0b 对齐；唯一变的是学习规则（局部 → BPTT）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from research.sel.lm.run_m0 import BATCH, make_batches
from research.sel.lm.run_r0 import (
    DATA,
    MAX_LEN,
    SEED,
    bigram_accuracy,
    build_vocab,
    encode,
    load_sequences,
)

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_b0"

EMB = 128
HIDDEN = 512
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
CLIP = 1.0
UPDATES = 8000


class CharGRU(nn.Module):
    def __init__(
        self, vocab: int, *, emb: int = EMB, hidden: int = HIDDEN, dropout: float = DROPOUT
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.drop(self.emb(tokens))
        output, _ = self.gru(hidden)
        return self.head(self.drop(output))


@torch.no_grad()
def evaluate(model: nn.Module, tokens: torch.Tensor, valid: torch.Tensor) -> dict:
    model.eval()
    logits = model(tokens[:, :-1])
    target = tokens[:, 1:].masked_fill(~valid[:, 1:], -100)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    mask = target.reshape(-1) != -100
    hits = int(
        (
            logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)[mask] == target.reshape(-1)[mask]
        ).sum()
    )
    counted = int(mask.sum())
    model.train()
    return {
        "ce": float(loss[mask].sum()) / max(counted, 1),
        "top1": hits / max(counted, 1),
        "tokens": counted,
    }


def run_fold(
    train_texts: list[str],
    test_texts: list[str],
    device,
    updates: int,
    *,
    max_vocab: int = 0,
    lr: float,
    eval_every: int = 0,
) -> dict:
    torch.manual_seed(SEED)
    vocab = build_vocab(train_texts, max_vocab=max_vocab)
    train = [encode(text, vocab, cap=MAX_LEN) for text in train_texts]
    test = [encode(text, vocab, cap=MAX_LEN) for text in test_texts]
    pad = vocab["<unk>"]
    model = CharGRU(len(vocab)).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED + 17)
    losses: list[float] = []
    curve: list[dict] = []
    probe_tokens, probe_valid = make_batches(test, order=list(range(len(test))), pad=pad)
    started = time.perf_counter()
    model.train()
    for step in range(updates):
        order = torch.randperm(len(train), generator=generator)[:BATCH].tolist()
        tokens, valid = make_batches(train, order=order, pad=pad)
        tokens, valid = tokens.to(device), valid.to(device)
        logits = model(tokens[:, :-1])
        target = tokens[:, 1:].masked_fill(~valid[:, 1:], -100)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        losses.append(float(loss.detach()))
        if eval_every and (step + 1) % eval_every == 0:
            probe = evaluate(model, probe_tokens.to(device), probe_valid.to(device))
            curve.append(
                {"updates": step + 1, "top1": round(probe["top1"], 4), "ce": round(probe["ce"], 4)}
            )
            print(f"    step {step + 1}: top1 {probe['top1']:.4f} ce {probe['ce']:.3f}", flush=True)
    metrics = evaluate(model, probe_tokens.to(device), probe_valid.to(device))
    base_top1, base_ce = bigram_accuracy(train, test, len(vocab))
    head = statistics.median(losses[:200])
    tail = statistics.median(losses[-200:])
    return {
        "vocab": len(vocab),
        "params": params,
        "lr": lr,
        "train_pairs": len(train),
        "test_pairs": len(test),
        "top1": round(metrics["top1"], 4),
        "ce": round(metrics["ce"], 4),
        "bigram_top1": round(base_top1, 4),
        "bigram_ce": round(base_ce, 4),
        "delta_top1": round(metrics["top1"] - base_top1, 4),
        "ce_ratio": round(tail / head, 4) if head else None,
        "tokens": metrics["tokens"],
        "elapsed_s": round(time.perf_counter() - started, 1),
        "curve": curve,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B0：同尺寸反向传播对照")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--lr", default="1e-3,3e-3")
    parser.add_argument("--sessions", default="", help="逗号分隔的留出会话；空=全部")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--tag", default="b0")
    parser.add_argument("--max-vocab", type=int, default=0, help="0=全部唯一字；给了按频率截断")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [float(item) for item in args.lr.split(",") if item.strip()]

    by_session = load_sequences(args.data)
    sessions = sorted(by_session)
    held_sessions = sessions
    if args.sessions:
        wanted = [item.strip() for item in args.sessions.split(",") if item.strip()]
        missing = [w for w in wanted if w not in sessions]
        if missing:
            raise SystemExit(f"未知会话：{missing}")
        held_sessions = wanted
    print(f"会话 {len(sessions)} 个，各折对数 {[len(by_session[s]) for s in sessions]}", flush=True)

    folds: dict[str, dict] = {}
    for held in held_sessions:
        train_texts = [text for s in sessions if s != held for text in by_session[s]]
        for lr in learning_rates:
            key = f"{held}|lr={lr:g}"
            folds[key] = run_fold(
                train_texts,
                by_session[held],
                device,
                args.updates,
                max_vocab=args.max_vocab,
                lr=lr,
                eval_every=args.eval_every,
            )
            # 一折跑完就清一次缓存：2026-11-10 在 KdConv（V=3001、长句）上，
            # 第二个 lr 因上一折的残留分配 OOM（11.2 GB）。
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row = folds[key]
            print(
                f"[{key}] params {row['params']} top1 {row['top1']} vs bigram {row['bigram_top1']} "
                f"→ Δ {row['delta_top1']:+.4f}，{row['elapsed_s']}s",
                flush=True,
            )

    best = max((row["top1"] for row in folds.values()), default=0.0)
    bigram_best = max((row["bigram_top1"] for row in folds.values()), default=0.0)
    criteria = {
        "b0_1_bp_beats_bigram": best > bigram_best,
        "best_top1": round(best, 4),
        "bigram_top1": round(bigram_best, 4),
        "best_delta": round(best - bigram_best, 4),
    }
    out = {
        "kind": "B0: same-size backprop control on the real corpus",
        "seed": SEED,
        "config": {
            "emb": EMB,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "clip": CLIP,
            "batch": BATCH,
            "max_len": MAX_LEN,
            "updates": args.updates,
            "eval_every": args.eval_every,
            "lrs": learning_rates,
            "data": str(args.data),
        },
        "folds": folds,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"最好一折 top1 {best:.4f} vs bigram {bigram_best:.4f} → Δ {best - bigram_best:+.4f}")
    print(f"[判据] b0_1={'过' if criteria['b0_1_bp_beats_bigram'] else '不过'}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
