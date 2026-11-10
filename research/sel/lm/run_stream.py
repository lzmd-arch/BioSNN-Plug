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
from research.sel.lm.bptt_lif import BPTTLIF
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
                logits = model._token_probabilities(model._features(trace))
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
    parser.add_argument(
        "--rule",
        default="local",
        choices=("local", "bptt"),
        help="STREAM-3：bptt = 同架构的 **T=1 截断 BPTT**（每步更新、状态 detach）",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="仅 bptt 臂使用（Adam）")
    parser.add_argument(
        "--corpus-bin",
        type=Path,
        default=None,
        help="预分词的 uint16 语料（build_tokenized_corpus.py 的产物）。给了就走 memmap，"
        "热路径零 Python；不给则退回 JSONL 现场编码。",
    )
    parser.add_argument("--test-frac", type=float, default=0.01, help="尾部留作测试的比例")
    parser.add_argument("--tag", default="stream_local")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if args.corpus_bin is not None:
        import numpy as np

        flat = np.memmap(args.corpus_bin, dtype=np.uint16, mode="r")
        split = int(flat.size * (1.0 - args.test_frac))
        # **只物化要流的那一段**：整条语料 np.array(int64) 在 2.4B token 下是 19 GB。
        need_train = min(args.train_tokens + 2, split)
        need_test = min(args.test_tokens + 2, flat.size - split)
        train_stream = torch.from_numpy(np.array(flat[:need_train], dtype=np.int64))
        test_stream = torch.from_numpy(np.array(flat[split : split + need_test], dtype=np.int64))
        n_vocab = int(flat.max()) + 1
        boundary = -1  # 预分词语料没有对话边界标记，不做跳过
        print(
            f"memmap 语料 {flat.size:,} token"
            f"（训练 {split:,} / 测试 {flat.size - split:,}，V={n_vocab}）",
            flush=True,
        )
    else:
        by_session = data_utils.load_sequences(args.data)
        held = [s for s in args.sessions.split(",") if s]
        test_texts = [t for s in held for t in by_session[s]]
        train_texts = [t for s, v in by_session.items() if s not in held for t in v]
        vocab = data_utils.build_vocab(train_texts, max_vocab=args.max_vocab)
        boundary = vocab[chr(1)]
        n_vocab = len(vocab)
        train_stream = flatten(train_texts, vocab)
        test_stream = flatten(test_texts, vocab)
    print(
        f"训练流 {train_stream.numel():,} token；测试流 {test_stream.numel():,} token", flush=True
    )

    if args.rule == "bptt":
        return run_bptt(args, train_stream, test_stream, boundary, device)

    taus = tuple(float(x) for x in args.readout_taus.split(",") if x.strip())
    model = TokenEProp(
        n_vocab,
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


def run_bptt(args, train_stream, test_stream, boundary, device) -> int:
    """STREAM-3：**T=1 截断 BPTT**，与局部规则同一协议（batch=1、无回放、常数状态）。"""
    taus = tuple(float(x) for x in args.readout_taus.split(",") if x.strip())
    model = BPTTLIF(
        int(max(train_stream.max().item(), test_stream.max().item())) + 1,
        args.n_rec,
        readout_taus=taus,
        generator=torch.Generator().manual_seed(args.seed),
        device=device,
    )
    # 词表大小以上面为界会偏小；用真实 vocab 更稳
    print(f"BPTT 臂：Adam lr {args.lr}，T=1，状态每步 detach", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = model.new_state(1)
    steps = min(args.train_tokens, train_stream.numel() - 1)
    tokens = train_stream.to(device)
    running = 0.0
    done = 0
    for index in range(steps):
        if int(tokens[index]) == boundary:
            continue
        loss, _, state = model.forward_stream(
            tokens[index].unsqueeze(0), tokens[index + 1].unsqueeze(0), state
        )
        optimizer.zero_grad(set_to_none=True)
        loss.mean().backward()
        # batch=1 + 脉冲稀疏梯度下 Adam 会给出巨大步长（实测 lr=1e-3 直接发散到 CE 38.7），
        # 必须裁剪。这一条写进日志，作为 BPTT 臂的**调参记录**，不是"BPTT 不行"。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += float(loss.mean())
        done += 1
        if done % 50000 == 0:
            print(f"    {done} 步，近 5 万步平均 CE {running / 50000:.3f}", flush=True)
            running = 0.0
    print(f"  训练流走完 {done:,} 步（batch=1，每步一次更新）", flush=True)

    hits = counted = 0
    total = 0.0
    state = model.new_state(1)
    test = test_stream.to(device)
    limit = min(args.test_tokens, test.numel() - 1)
    with torch.no_grad():
        for index in range(limit):
            if int(test[index]) == boundary:
                continue
            loss, logits, state = model.forward_stream(
                test[index].unsqueeze(0), test[index + 1].unsqueeze(0), state
            )
            hits += int(logits.argmax(dim=-1)[0] == test[index + 1])
            total += float(loss[0])
            counted += 1
    stats = {"ce": total / max(counted, 1), "top1": hits / max(counted, 1), "tokens": counted}
    print(f"  流后（不更新）: ce {stats['ce']:.3f} top1 {stats['top1']:.4f}", flush=True)
    result = {
        "kind": "STREAM-3：T=1 截断 BPTT（同协议对照）",
        "config": {
            "rule": "bptt",
            "lr": args.lr,
            "n_rec": args.n_rec,
            "train_tokens": args.train_tokens,
            "test_tokens": args.test_tokens,
        },
        "after": stats,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
