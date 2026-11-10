"""R12 探针的执行器：**同一个单层 LIF，用 BPTT 训练**。

用法::

    uv run python -m research.sel.lm.run_bptt_lif --updates 4000 --eval-every 1000

对照组是**同架构的局部规则**（2 抽头、同语料、同预算）：4,000 步 top-1 **0.2451** / CE 4.417。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.sel.lm import run_r0 as data_utils
from research.sel.lm.bptt_lif import BPTTLIF

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_r12_bptt"
TAUS_DEFAULT = (2.0, 128.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BPTT 对照：同架构单层 LIF + 真梯度")
    parser.add_argument(
        "--data", type=Path, default=Path("D:/AxoData/sel/lm/v1/public/kdconv/dialogue.jsonl")
    )
    parser.add_argument("--sessions", default="test")
    parser.add_argument("--max-vocab", type=int, default=3000)
    parser.add_argument("--n-rec", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--readout-taus", default="2,128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="bptt192")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    by_session = data_utils.load_sequences(args.data)
    held = [s for s in args.sessions.split(",") if s]
    test_texts = [t for s in held for t in by_session[s]]
    train_texts = [t for s, v in by_session.items() if s not in held for t in v]
    print(f"会话 {len(by_session)} 个，各折对数 {[len(by_session[s]) for s in held]}", flush=True)

    vocab = data_utils.build_vocab(train_texts, max_vocab=args.max_vocab)
    pad = vocab["<unk>"]
    train = [data_utils.encode(t, vocab, cap=data_utils.MAX_LEN) for t in train_texts]
    test = [data_utils.encode(t, vocab, cap=data_utils.MAX_LEN) for t in test_texts]
    test_tokens, test_valid = data_utils.make_batches(test, order=list(range(len(test))), pad=pad)
    test_tokens, test_valid = test_tokens.to(device), test_valid.to(device)

    taus = tuple(float(x) for x in args.readout_taus.split(",") if x.strip())
    model = BPTTLIF(
        len(vocab),
        args.n_rec,
        readout_taus=taus,
        generator=torch.Generator().manual_seed(args.seed),
        device=device,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"参数 {params:,}；抽头 {taus}；lr {args.lr}；clip {args.clip}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed + 17)

    bigram_top1, bigram_ce = data_utils.bigram_accuracy(train, test, len(vocab))
    curve = []
    running = torch.zeros((), device=device)
    for step in range(1, args.updates + 1):
        order = torch.randint(0, len(train), (args.batch,), generator=generator).tolist()
        tokens, valid = data_utils.make_batches(train, order=order, pad=pad)
        tokens, valid = tokens.to(device), valid.to(device)
        losses, _ = model(tokens, valid)
        mask = valid[:, 1:].to(losses.dtype)
        loss = (losses * mask).sum() / mask.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        running = running + loss.detach()
        if args.eval_every and step % args.eval_every == 0:
            stats = model.evaluate(test_tokens, test_valid)
            curve.append(
                {"updates": step, "top1": round(stats["top1"], 4), "ce": round(stats["ce"], 4)}
            )
            print(
                f"    step {step}: test top1 {stats['top1']:.4f} ce {stats['ce']:.3f}"
                f"  (train {float(running) / args.eval_every:.3f})",
                flush=True,
            )
            running = torch.zeros((), device=device)

    stats = model.evaluate(test_tokens, test_valid)
    fold = {
        "top1": round(stats["top1"], 4),
        "ce": round(stats["ce"], 4),
        "bigram_top1": round(bigram_top1, 4),
        "bigram_ce": round(bigram_ce, 3),
        "delta_top1": round(stats["top1"] - bigram_top1, 4),
        "tokens": stats["tokens"],
        "vocab": len(vocab),
        "params": params,
        "curve": curve,
    }
    print(f"[hold-out] {json.dumps(fold, ensure_ascii=False)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(
            {
                "config": vars(args) | {"data": str(args.data), "out": str(args.out)},
                "folds": {"test": fold},
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
