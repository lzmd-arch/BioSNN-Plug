"""流式协议：**batch=1、逐 token、一次一个更新、无回放、状态不增长**。

这是局部规则**结构上**可能赢过 BPTT 的那条轴：BPTT 在同样约束下只能退化成 T=1 截断
（丢掉长程），否则必须开回放（内存 O(L·B)）。我们的规则**天生就是这个协议**。

prequential（在线）指标：**先预测、再更新**——所以报的每个损失都是"更新前"的，
不允许偷看当前位置的标签。跨对话对边界的步会被跳过（那里没有可预测的上文）。

用法::

    uv run python -m research.sel.lm.run_stream --train-tokens 400000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.sel.lm import run_r0 as data_utils
from research.sel.lm.model import TokenEProp

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_stream"


def flatten(texts: list[str], vocab: dict[str, int]) -> torch.Tensor:
    ids: list[int] = []
    for text in texts:
        ids.extend(data_utils.encode(text, vocab, cap=data_utils.MAX_LEN))
    return torch.tensor(ids, dtype=torch.long)


def stream_pass(
    model, trace, stream: torch.Tensor, *, boundary: int, update: bool, limit: int, device
) -> dict:
    """沿流走一遍；返回 prequential 的 CE / top-1（按十分位分段）。"""
    steps = min(limit, stream.numel() - 1)
    buckets = [[0.0, 0, 0] for _ in range(10)]  # ce_sum, hits, count
    total_ce = 0.0
    total_hits = 0
    total = 0
    tokens = stream.to(device)
    for index in range(steps):
        prev, target = tokens[index], tokens[index + 1]
        if int(prev) == boundary:  # 跨对话边界，无可预测上文
            continue
        loss, _ = model.step(prev.unsqueeze(0), target.unsqueeze(0), trace, update=update)
        value = float(loss)
        # 只有需要 top-1 时才多算一次 logits（评测阶段）
        bucket = min(9, index * 10 // max(steps, 1))
        buckets[bucket][0] += value
        buckets[bucket][2] += 1
        total_ce += value
        total += 1
        if not update:
            with torch.no_grad():
                logits = model._logits(model._features(trace))[:, : model.n_vocab]
            hit = int(logits.argmax(dim=-1)[0] == target)
            buckets[bucket][1] += hit
            total_hits += hit
    return {
        "ce": total_ce / max(total, 1),
        "top1": total_hits / max(total, 1),
        "tokens": total,
        "deciles": [
            {"ce": round(c / max(n, 1), 4), "top1": round(h / max(n, 1), 4), "n": n}
            for c, h, n in buckets
            if n
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="流式协议（batch=1、无回放、常数内存）")
    parser.add_argument(
        "--data", type=Path, default=Path("D:/AxoData/sel/lm/v1/public/kdconv/dialogue.jsonl")
    )
    parser.add_argument("--sessions", default="test")
    parser.add_argument("--max-vocab", type=int, default=3000)
    parser.add_argument("--n-rec", type=int, default=1024)
    parser.add_argument("--train-tokens", type=int, default=400000)
    parser.add_argument("--test-tokens", type=int, default=200000)
    parser.add_argument("--readout-taus", default="2,128")
    parser.add_argument("--trace-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="stream_local")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    by_session = data_utils.load_sequences(args.data)
    held = [s for s in args.sessions.split(",") if s]
    test_texts = [t for s in held for t in by_session[s]]
    train_texts = [t for s, v in by_session.items() if s not in held for t in v]
    vocab = data_utils.build_vocab(train_texts, max_vocab=args.max_vocab)
    boundary = vocab[chr(1)]
    train_stream = flatten(train_texts, vocab)
    test_stream = flatten(test_texts, vocab)
    print(
        f"训练流 {train_stream.numel():,} token；测试流 {test_stream.numel():,} token", flush=True
    )

    taus = tuple(float(x) for x in args.readout_taus.split(",") if x.strip())
    model = TokenEProp(
        len(vocab),
        args.n_rec,
        fan_in=data_utils.FAN_IN,
        tau=data_utils.TAU,
        tau_read=data_utils.TAU_READ,
        input_std=data_utils.INPUT_STD,
        lr_in=data_utils.LR_IN,
        lr_out=data_utils.LR_OUT,
        recurrence="none",
        readout_taus=taus,
        trace_decay=args.trace_decay,
        generator=torch.Generator().manual_seed(args.seed),
        device=device,
    )
    #: 状态是 O(1) 的：膜/脉冲/不应期 (1,N)、输入痕迹 (1,V)、多抽头 EMA (1,K,N)。
    #: **与流长度无关**——这正是 BPTT 在同样协议下做不到的地方。
    state_elems = sum(
        int(t.numel()) for t in vars(model.new_trace(1)).values() if isinstance(t, torch.Tensor)
    )
    print(f"常数状态 {state_elems:,} 个标量（与流长度无关）", flush=True)

    trace = model.new_trace(1)
    before = stream_pass(
        model,
        model.new_trace(1),
        test_stream,
        boundary=boundary,
        update=False,
        limit=args.test_tokens,
        device=device,
    )
    print(f"  流前（未训练）: ce {before['ce']:.3f} top1 {before['top1']:.4f}", flush=True)

    trace = model.new_trace(1)
    train_stats = stream_pass(
        model,
        trace,
        train_stream,
        boundary=boundary,
        update=True,
        limit=args.train_tokens,
        device=device,
    )
    print(f"  训练流走完 {train_stats['tokens']:,} 步（batch=1，每步一次更新）", flush=True)

    trace = model.new_trace(1)
    after = stream_pass(
        model,
        model.new_trace(1),
        test_stream,
        boundary=boundary,
        update=False,
        limit=args.test_tokens,
        device=device,
    )
    print(f"  流后（不更新）: ce {after['ce']:.3f} top1 {after['top1']:.4f}", flush=True)
    for i, d in enumerate(after["deciles"]):
        print(f"    第 {i} 十分位: ce {d['ce']:.3f} top1 {d['top1']:.4f} (n={d['n']})", flush=True)

    result = {
        "kind": "流式协议：batch=1、逐 token、无回放、常数状态",
        "config": {
            "n_rec": args.n_rec,
            "taus": list(taus),
            "trace_decay": args.trace_decay,
            "train_tokens": args.train_tokens,
            "test_tokens": args.test_tokens,
            "lr_in": data_utils.LR_IN,
            "lr_out": data_utils.LR_OUT,
            "seed": args.seed,
        },
        "constant_state_scalars": state_elems,
        "before": before,
        "after": after,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
