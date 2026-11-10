"""D2 执行器：**只驱动 `DeepTokenLM`**，不碰 `run_r0`（那边有排队中的臂在跑）。

对照是单层的 `run_r0` 同配置：2 抽头、4,000 步 → top-1 **0.2451** / CE **4.417**。

用法::

    uv run python -m research.sel.lm.run_deep --updates 4000 --eval-every 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.sel.lm import run_r0 as data_utils
from research.sel.lm.model_deep import DeepTokenLM

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_d2"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D2：两层 LIF + 全局部规则")
    ap.add_argument(
        "--data", type=Path, default=Path("D:/AxoData/sel/lm/v1/public/kdconv/dialogue.jsonl")
    )
    ap.add_argument("--sessions", default="test")
    ap.add_argument("--max-vocab", type=int, default=3000)
    ap.add_argument("--n-rec", type=int, default=1024)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument(
        "--batch", type=int, default=data_utils.BATCH if hasattr(data_utils, "BATCH") else 16
    )
    ap.add_argument("--lr-in", type=float, default=1e-3)
    ap.add_argument("--lr-mid", type=float, default=1e-3)
    ap.add_argument(
        "--w12-scale",
        type=float,
        default=0.4,
        help="第 2 层电流的初始尺度，**必须标定**：0.1 会把第 2 层饿死（电流 0.13 ≪ 阈值 0.62，"
        "实测 z2 几乎为 0、读出完全不学）；1.0 会饱和。0.4 实测两层都在发放。",
    )
    ap.add_argument("--readout-taus", default="2,128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="d2_2tap_4k")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    by_session = data_utils.load_sequences(args.data)
    held = [s for s in args.sessions.split(",") if s]
    test_texts = [t for s in held for t in by_session[s]]
    train_texts = [t for s, v in by_session.items() if s not in held for t in v]
    vocab = data_utils.build_vocab(train_texts, max_vocab=args.max_vocab)
    pad = vocab["<unk>"]
    train = [data_utils.encode(t, vocab, cap=data_utils.MAX_LEN) for t in train_texts]
    test = [data_utils.encode(t, vocab, cap=data_utils.MAX_LEN) for t in test_texts]
    test_tokens, test_valid = data_utils.make_batches(test, order=list(range(len(test))), pad=pad)
    test_tokens, test_valid = test_tokens.to(device), test_valid.to(device)

    taus = tuple(float(x) for x in args.readout_taus.split(",") if x.strip())
    model = DeepTokenLM(
        len(vocab),
        args.n_rec,
        readout_taus=taus,
        lr_in=args.lr_in,
        lr_mid=args.lr_mid,
        w12_scale=args.w12_scale,
        generator=torch.Generator().manual_seed(args.seed),
        device=device,
    )
    params = model.w_in.numel() + model.w12.numel() + model.w_out.numel()
    print(
        f"层数 2；参数 {params:,}（w12 占 {model.w12.numel():,}）；w12_scale {args.w12_scale}",
        flush=True,
    )

    gen = torch.Generator().manual_seed(args.seed + 17)
    curve = []
    running = 0.0
    fired2 = 0.0
    for step in range(1, args.updates + 1):
        order = torch.randint(0, len(train), (args.batch,), generator=gen).tolist()
        tokens, valid = data_utils.make_batches(train, order=order, pad=pad)
        tokens, valid = tokens.to(device), valid.to(device)
        trace = model.new_trace(tokens.shape[0])
        result = model.run(tokens, trace, valid=valid)
        running += result["ce"]
        fired2 += float(trace.z2.sum())
        if args.eval_every and step % args.eval_every == 0:
            stats = model.evaluate(test_tokens, test_valid)
            curve.append(
                {"updates": step, "top1": round(stats["top1"], 4), "ce": round(stats["ce"], 4)}
            )
            print(
                f"    step {step}: test top1 {stats['top1']:.4f} ce {stats['ce']:.3f}"
                f"  (train {running / args.eval_every:.3f}, 第2层发放 {stats['firing_rate']:.4f})",
                flush=True,
            )
            running = 0.0
    bigram_top1, bigram_ce = data_utils.bigram_accuracy(train, test, len(vocab))
    stats = model.evaluate(test_tokens, test_valid)
    fold = {
        "top1": round(stats["top1"], 4),
        "ce": round(stats["ce"], 4),
        "bigram_top1": round(bigram_top1, 4),
        "bigram_ce": round(bigram_ce, 3),
        "delta_top1": round(stats["top1"] - bigram_top1, 4),
        "firing_rate": round(stats["firing_rate"], 5),
        "params": params,
        "w12_params": int(model.w12.numel()),
        "tokens": stats["tokens"],
        "curve": curve,
    }
    print(f"[hold-out] {json.dumps(fold, ensure_ascii=False)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(
            {"config": {k: str(v) for k, v in vars(args).items()}, "fold": fold},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
