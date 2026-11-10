"""R0：真实会话语料上的字符级 next-token（判据见 `prereg-lm-r0-real-corpus.md`）。

留一会话（六折）；每折独立建词表、独立训练、独立评测；基线是同折的字符 bigram。
**数据不出本机**：只读 `D:/AxoData/sel/lm/v1/real.jsonl`，结果只写聚合指标。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import torch

from research.sel.lm.model import TokenEProp
from research.sel.lm.run_m0 import BATCH, FAN_IN, N_REC, make_batches

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/lm/v1/real.jsonl")
OUT = REPO / "sweep_results" / "lm_eprop_r0"

UPDATES = 2000
MAX_LEN = 192
TAU = 20.0
TAU_READ = 8.0
INPUT_STD = 0.12
LR_IN = 1e-3
LR_OUT = 0.5
SEED = 0
BIGRAM_K = 0.5

#: 判据（改这里必须改预注册）。
LOSS_DROP = 0.6
WIN_DELTA = 0.03
WIN_FOLDS = 4


def load_sequences(path: Path, *, use_context: bool = False) -> dict[str, list[str]]:
    """按会话分组。``use_context`` 时把 ``context`` 字段**前置**（P2 的判别实验）。

    理由：这个模型没有注意力，上下文能起作用的唯一机制是**膜电位 priming**——
    上下文必须先到，才能影响用户那句话。分隔符与原有的对分隔符区分开。
    """
    by_session: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        text = f"\x01{row['user']}\x02{row['assistant']}\x03"
        if use_context and row.get("context"):
            text = f"\x04{row['context']}\x05" + text
        by_session[row["session"]].append(text)
    return dict(by_session)


def strip_context(text: str) -> str:
    """去掉前置的 context（chr(4)…chr(5)），只留 user / assistant 那一对。

    P2 的 bigram 基线用**剥掉上下文**的序列算——bigram 只能看前一个字符，本来就吃不到
    上下文；用剥掉的版本才与 P1 的 0.2887 完全同口径。
    """
    open_mark, close_mark = chr(4), chr(5)
    if not text.startswith(open_mark):
        return text
    end = text.find(close_mark)
    return text[end + 1 :] if end >= 0 else text


def _mask_context(tokens, valid, vocab) -> None:
    """把上下文那一段从 ``valid`` 里划掉——只对 user/assistant 计分（P2）。

    判据是每行第一个 ``chr(1)``（user 段的开头）；它之前的 token 全是前置上下文。
    """
    mark = vocab.get(chr(1))
    if mark is None:
        return
    for row in range(tokens.shape[0]):
        hits = (tokens[row] == mark).nonzero(as_tuple=True)[0]
        if hits.numel():
            valid[row, : int(hits[0]) + 1] = False


def build_candidates(train: list[list[int]], size: int, count: int) -> torch.Tensor:
    """A1：``(V, C)`` 候选表——给定上一个字，最可能的 C 个下一个字。

    用训练集 bigram 计数，**加一元频率兜底**（``1e-3 × unigram``），
    否则没见过的上下文会退化成按 id 排序的任意候选。
    """
    import numpy as np

    counts = np.zeros((size, size), dtype=np.float32)
    for sequence in train:
        for left, right in pairwise(sequence):
            counts[left, right] += 1.0
    unigram = counts.sum(axis=0)
    unigram = unigram / max(float(unigram.sum()), 1.0)
    order = np.argsort(-(counts + 1e-3 * unigram[None, :]), axis=1)[:, :count]
    return torch.tensor(order, dtype=torch.long)


def candidate_hit_rate(test: list[list[int]], table: torch.Tensor) -> float:
    """target 落在候选集里的比例（A1-1 的前置判据）。"""
    hits = total = 0
    for sequence in test:
        for left, right in pairwise(sequence):
            total += 1
            if bool((table[left] == right).any()):
                hits += 1
    return hits / max(total, 1)


def build_vocab(texts: list[str], *, max_vocab: int = 0) -> dict[str, int]:
    """词表。``max_vocab=0`` 时保持历史行为（全部唯一字、按码位排序）；
    给了上限时按**训练集频率**取前 N 个（公共语料稀有字太多，全收会撑爆读出）。"""
    if max_vocab <= 0:
        chars = sorted({ch for text in texts for ch in text})
    else:
        counts = Counter(ch for text in texts for ch in text)
        chars = [ch for ch, _ in counts.most_common(max_vocab)]
    return {token: index for index, token in enumerate(["<unk>", *chars])}


def encode(text: str, vocab: dict[str, int], *, cap: int) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(ch, unk) for ch in text][:cap]


def bigram_accuracy(
    train: list[list[int]], test: list[list[int]], size: int
) -> tuple[float, float]:
    counts: dict[int, Counter] = defaultdict(Counter)
    for sequence in train:
        for left, right in pairwise(sequence):
            counts[left][right] += 1
    hits = total = 0
    ce = 0.0
    for sequence in test:
        for left, right in pairwise(sequence):
            row = counts.get(left)
            if not row:
                total += 1
                ce += -math.log(1.0 / max(size, 2))
                continue
            denominator = sum(row.values()) + BIGRAM_K * size
            hits += int(max(row, key=row.get) == right)
            ce += -math.log((row.get(right, 0) + BIGRAM_K) / denominator)
            total += 1
    return hits / max(total, 1), ce / max(total, 1)


def run_fold(
    train_texts: list[str],
    test_texts: list[str],
    device,
    updates: int,
    eval_every: int = 0,
    *,
    max_vocab: int = 0,
    use_context: bool = False,
    candidate_count: int = 0,
    readout_rank: int | None = None,
    tau_groups: int = 1,
    beta: float = 0.0,
    tau: float = TAU,
    readout_width: int | None = None,
    readout_taus: tuple[float, ...] | None = None,
) -> dict:
    torch.manual_seed(SEED)
    vocab = build_vocab(train_texts, max_vocab=max_vocab)
    train = [encode(text, vocab, cap=MAX_LEN) for text in train_texts]
    test = [encode(text, vocab, cap=MAX_LEN) for text in test_texts]
    # P2：bigram 基线用**剥掉上下文**的序列（它只能看前一个字符，本来就吃不到上下文）
    if use_context:
        train_plain = [encode(strip_context(text), vocab, cap=MAX_LEN) for text in train_texts]
        test_plain = [encode(strip_context(text), vocab, cap=MAX_LEN) for text in test_texts]
    else:
        train_plain, test_plain = train, test
    pad = vocab["<unk>"]
    candidates = None
    hit = None
    if candidate_count > 0:
        candidates = build_candidates(train, len(vocab), candidate_count)
        hit = round(candidate_hit_rate(test, candidates), 4)
    model = TokenEProp(
        len(vocab),
        N_REC,
        fan_in=FAN_IN,
        tau=tau,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=LR_IN,
        lr_out=LR_OUT,
        recurrence="none",
        readout_width=readout_width,
        readout_taus=readout_taus,
        candidates=candidates,
        readout_rank=readout_rank,
        tau_groups=tau_groups,
        beta=beta,
        generator=torch.Generator().manual_seed(SEED),
        device=device,
    )
    generator = torch.Generator().manual_seed(SEED + 17)
    losses: list[float] = []
    curve: list[dict] = []
    test_tokens, test_valid = make_batches(test, order=list(range(len(test))), pad=pad)
    if use_context:
        _mask_context(test_tokens, test_valid, vocab)
    for step in range(updates):
        order = torch.randperm(len(train), generator=generator)[:BATCH].tolist()
        tokens, valid = make_batches(train, order=order, pad=pad)
        if use_context:
            _mask_context(tokens, valid, vocab)
        result = model.run(
            tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
        )
        losses.append(result["ce"])
        if eval_every and (step + 1) % eval_every == 0:
            # 周期评测必须用**全新状态**，否则会把训练状态带进测试
            probe = model.evaluate(
                test_tokens.to(device),
                model.new_trace(test_tokens.shape[0]),
                valid=test_valid.to(device),
            )
            curve.append(
                {"updates": step + 1, "top1": round(probe["top1"], 4), "ce": round(probe["ce"], 4)}
            )
            print(
                f"    step {step + 1}: test top1 {probe['top1']:.4f} ce {probe['ce']:.3f}",
                flush=True,
            )
    tokens, valid = test_tokens, test_valid
    metrics = model.evaluate(
        tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
    )
    base_top1, base_ce = bigram_accuracy(train_plain, test_plain, len(vocab))
    head = statistics.median(losses[:200])
    tail = statistics.median(losses[-200:])
    return {
        "vocab": len(vocab),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "top1": round(metrics["top1"], 4),
        "ce": round(metrics["ce"], 4),
        "bigram_top1": round(base_top1, 4),
        "bigram_ce": round(base_ce, 4),
        "delta_top1": round(metrics["top1"] - base_top1, 4),
        "ce_ratio": round(tail / head, 4) if head else None,
        "tokens": metrics["tokens"],
        "top1_by_pos": metrics.get("top1_by_pos"),
        "candidate_hit": hit,
        "tau": tau,
        "readout_dims": model.readout_width,
        "firing_rate": round(metrics["firing_rate"], 5),
        "curve": curve,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R0：真语料")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--tag", default="r0")
    parser.add_argument("--sessions", default="", help="逗号分隔的留出会话；空=全部")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--tau", type=float, default=TAU, help="膜时间常数（R5 的变量）")
    parser.add_argument("--max-vocab", type=int, default=0, help="0=全部唯一字；给了按频率截断")
    parser.add_argument("--use-context", action="store_true", help="P2：把 context 字段前置")
    parser.add_argument(
        "--candidates", type=int, default=0, help="A1：候选集大小 C；0=全量 softmax"
    )
    parser.add_argument(
        "--readout-width", type=int, default=0, help="0=读全部神经元；给了就是 R6q 的变量"
    )
    parser.add_argument(
        "--readout-taus", default="", help="逗号分隔；给了就是 R8（多时间尺度读出）"
    )
    parser.add_argument("--readout-rank", type=int, default=0, help="A3：低秩读出 r；0=稠密")
    parser.add_argument("--tau-groups", type=int, default=1, help="G2：异质膜 τ 的组数；1=同质")
    parser.add_argument("--beta", type=float, default=0.0, help="G1：ALIF 适应强度 β；0=纯 LIF")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taus = tuple(float(item) for item in args.readout_taus.split(",") if item.strip()) or None

    by_session = load_sequences(args.data, use_context=args.use_context)
    sessions = sorted(by_session)
    held_sessions = sessions
    if args.sessions:
        wanted = [item.strip() for item in args.sessions.split(",") if item.strip()]
        missing = [w for w in wanted if w not in sessions]
        if missing:
            raise SystemExit(f"未知会话：{missing}")
        held_sessions = wanted  # 只限制跑哪几折；训练仍用全部会话
    print(f"会话 {len(sessions)} 个，各折对数 {[len(by_session[s]) for s in sessions]}", flush=True)

    folds: dict[str, dict] = {}
    for held in held_sessions:
        train_texts = [text for s in sessions if s != held for text in by_session[s]]
        folds[held] = run_fold(
            train_texts,
            by_session[held],
            device,
            args.updates,
            args.eval_every,
            max_vocab=args.max_vocab,
            use_context=args.use_context,
            candidate_count=args.candidates,
            readout_rank=args.readout_rank or None,
            tau_groups=args.tau_groups,
            beta=args.beta,
            tau=args.tau,
            readout_width=args.readout_width or None,
            readout_taus=taus,
        )
        print(f"[hold-out {held}] {folds[held]}", flush=True)

    deltas = [fold["delta_top1"] for fold in folds.values()]
    ce_deltas = [fold["bigram_ce"] - fold["ce"] for fold in folds.values()]
    mean_delta = statistics.mean(deltas)
    criteria = {
        "r0_0_loss_falls": all(
            fold["ce_ratio"] is not None and fold["ce_ratio"] < LOSS_DROP for fold in folds.values()
        ),
        "r0_1_beats_bigram": mean_delta >= WIN_DELTA,
        "r0_2_most_folds_win": sum(1 for delta in deltas if delta > 0) >= WIN_FOLDS,
        "r0_3_ce_better": statistics.mean(ce_deltas) > 0,
        "mean_delta_top1": round(mean_delta, 4),
        "folds_winning": sum(1 for delta in deltas if delta > 0),
        "mean_top1": round(statistics.mean(f["top1"] for f in folds.values()), 4),
        "mean_bigram_top1": round(statistics.mean(f["bigram_top1"] for f in folds.values()), 4),
    }
    out = {
        "kind": "R0: feedforward local-rule LM on a real conversation corpus",
        "seed": SEED,
        "config": {
            "n_rec": N_REC,
            "fan_in": FAN_IN,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "updates": args.updates,
            "eval_every": args.eval_every,
            "lr_in": LR_IN,
            "lr_out": LR_OUT,
            "tau": args.tau,
            "max_vocab": args.max_vocab,
            "use_context": bool(args.use_context),
            "candidates": args.candidates,
            "readout_rank": args.readout_rank or None,
            "tau_groups": args.tau_groups,
            "beta": args.beta,
            "tau_read": TAU_READ,
            "input_std": INPUT_STD,
            "seed": SEED,
            "data": str(args.data),
            "readout_width": args.readout_width or None,
            "readout_taus": list(taus) if taus else None,
        },
        "folds": folds,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"top1 均值 {criteria['mean_top1']:.4f} vs bigram {criteria['mean_bigram_top1']:.4f}；"
        f"赢的折 {criteria['folds_winning']}/{len(folds)}；均值差 {mean_delta:+.4f}"
    )
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
