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

from research.sel.lm.hierarchical import build_huffman_paths
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


class LengthBucketedSampler:
    """按长度分桶的批采样：**同批长度相近**（填充从 93–204% 降到 ~1%），但桶内与桶序都打乱。

    只是**抽样顺序**变了：同一个 epoch 里每条样本仍然被看一次，损失口径不变。
    """

    def __init__(self, lengths, batch: int, generator, *, pool_factor: int = 10):
        self.lengths = list(lengths)
        self.batch = int(batch)
        self.generator = generator
        self.pool = max(self.batch, self.batch * int(pool_factor))
        self.queue: list[int] = []

    def _refill(self) -> None:
        order = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        order.sort(key=lambda index: self.lengths[index])  # 长度升序
        pools = [order[i : i + self.pool] for i in range(0, len(order), self.pool)]
        pool_order = torch.randperm(len(pools), generator=self.generator).tolist()
        self.queue = []
        for pool_index in pool_order:
            pool = pools[pool_index]
            perm = torch.randperm(len(pool), generator=self.generator).tolist()
            self.queue.extend(pool[j] for j in perm)

    def next(self) -> list[int]:
        if len(self.queue) < self.batch:
            self._refill()
        out = self.queue[: self.batch]
        self.queue = self.queue[self.batch :]
        return out


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
    embed_rank: int | None = None,
    bucket_lengths: bool = False,
    tau_groups: int = 1,
    beta: float = 0.0,
    tau: float = TAU,
    readout_width: int | None = None,
    readout_taus: tuple[float, ...] | None = None,
    pad_vocab: bool = True,
    bf16_weights: bool = False,
    weight_dtype: str = "fp32",
    mirror_dtype: str = "",
    bf16_input: bool = False,
    norm_groups: int = 1,
    trace_decay: float = 0.0,
    n_rec: int = N_REC,
    lr_in: float = LR_IN,
    learn_hidden: bool = True,
    hierarchical: bool = False,
    tree_signal_scale: float = 1.0,
    seed: int = SEED,
) -> dict:
    torch.manual_seed(seed)
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
    paths = None
    if hierarchical:
        # 树的频次只从**训练集**统计：用测试集的频次建树等于把答案漏进结构里
        flat = torch.tensor([t for seq in train for t in seq], dtype=torch.long)
        counts = torch.bincount(flat, minlength=len(vocab)).to(torch.float32)
        paths = build_huffman_paths(counts)
    model = TokenEProp(
        len(vocab),
        n_rec,
        fan_in=FAN_IN,
        tau=tau,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=lr_in,
        lr_out=LR_OUT,
        recurrence="none",
        readout_width=readout_width,
        readout_taus=readout_taus,
        candidates=candidates,
        tree=paths,
        tree_signal_scale=tree_signal_scale,
        pad_vocab=pad_vocab,
        bf16_weights=bf16_weights,
        weight_dtype=weight_dtype,
        mirror_dtype=mirror_dtype,
        bf16_input=bf16_input,
        norm_groups=norm_groups,
        trace_decay=trace_decay,
        readout_rank=readout_rank,
        embed_rank=embed_rank,
        tau_groups=tau_groups,
        beta=beta,
        generator=torch.Generator().manual_seed(seed),
        device=device,
    )
    generator = torch.Generator().manual_seed(seed + 17)
    sampler = (
        LengthBucketedSampler([len(sequence) for sequence in train], BATCH, generator)
        if bucket_lengths
        else None
    )
    losses: list[float] = []
    curve: list[dict] = []
    test_tokens, test_valid = make_batches(test, order=list(range(len(test))), pad=pad)
    if use_context:
        _mask_context(test_tokens, test_valid, vocab)
    for step in range(updates):
        order = (
            sampler.next()
            if sampler is not None
            else torch.randperm(len(train), generator=generator)[:BATCH].tolist()
        )
        tokens, valid = make_batches(train, order=order, pad=pad)
        if use_context:
            _mask_context(tokens, valid, vocab)
        result = model.run(
            tokens.to(device),
            model.new_trace(tokens.shape[0]),
            valid=valid.to(device),
            learn_hidden=learn_hidden,
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
                f"    step {step + 1}: test top1 {probe['top1']:.4f}"
                f" top5 {probe.get('top5', 0):.4f} top10 {probe.get('top10', 0):.4f}"
                f" ce {probe['ce']:.3f}",
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
        "top5": round(metrics.get("top5", 0.0), 4),
        "top10": round(metrics.get("top10", 0.0), 4),
        "mean_rank_cost": round(metrics.get("mean_rank_cost", 0.0), 4),
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sessions", default="", help="逗号分隔的留出会话；空=全部")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--tau", type=float, default=TAU, help="膜时间常数（R5 的变量）")
    parser.add_argument("--n-rec", type=int, default=N_REC, help="神经元数 N（R3 的变量）")
    parser.add_argument("--lr-in", type=float, default=LR_IN, help="S7 对照臂：输入权重学习率")
    parser.add_argument(
        "--no-learn-hidden",
        action="store_true",
        help="S11 判决臂：冻结 w_in。若性能不变 ⇒ 隐层学习无贡献，针对隐层的辅助目标没有增益空间",
    )
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
    parser.add_argument("--embed-rank", type=int, default=0, help="A2：低秩输入 d；0=稠密")
    parser.add_argument("--bucket-lengths", action="store_true", help="按长度分桶采样（省填充）")
    parser.add_argument(
        "--tree-signal-scale", type=float, default=1.0, help="H-1c：树的隐层信号再乘这个系数"
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="H-1：读出换成 Huffman 树的路径概率（局部更新不变）；评测仍走全词表 top-k",
    )
    parser.add_argument("--no-pad-vocab", action="store_true", help="S1：关掉词表补零（默认开）")
    parser.add_argument("--bf16-weights", action="store_true", help="S2：读出权重存 bf16")
    parser.add_argument(
        "--weight-dtype",
        default="fp32",
        choices=("fp32", "bf16", "fp16"),
        help="S2：读出权重存储精度（fp16 的尾数比 bf16 多 2 位）",
    )
    parser.add_argument(
        "--mirror-dtype",
        default="",
        choices=("", "bf16", "fp16"),
        help="S2'：主权重留 fp32，另存一份低精度镜像供矩阵乘读",
    )
    parser.add_argument("--bf16-input", action="store_true", help="S2b：输入权重 w_in 存 bf16")
    parser.add_argument(
        "--trace-decay",
        type=float,
        default=0.0,
        help="S7：资格痕迹衰减 β（与膜常数解耦）；0=跟 alpha（历史行为）。β=0.99 对应窗口 ~100 步",
    )
    parser.add_argument(
        "--norm-groups",
        type=int,
        default=1,
        help="S5：读出的 NLMS 归一化分几组；1=全局标量（历史行为）",
    )
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
            embed_rank=args.embed_rank or None,
            bucket_lengths=args.bucket_lengths,
            tau_groups=args.tau_groups,
            pad_vocab=not args.no_pad_vocab,
            bf16_weights=args.bf16_weights,
            weight_dtype=args.weight_dtype,
            mirror_dtype=args.mirror_dtype,
            bf16_input=args.bf16_input,
            norm_groups=args.norm_groups,
            trace_decay=args.trace_decay,
            n_rec=args.n_rec,
            lr_in=args.lr_in,
            learn_hidden=not args.no_learn_hidden,
            hierarchical=bool(args.hierarchical),
            tree_signal_scale=args.tree_signal_scale,
            seed=args.seed,
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
            "n_rec": args.n_rec,
            "fan_in": FAN_IN,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "updates": args.updates,
            "eval_every": args.eval_every,
            "lr_in": args.lr_in,
            "lr_out": LR_OUT,
            "tau": args.tau,
            "max_vocab": args.max_vocab,
            "use_context": bool(args.use_context),
            "candidates": args.candidates,
            "readout_rank": args.readout_rank or None,
            "embed_rank": args.embed_rank or None,
            "bucket_lengths": bool(args.bucket_lengths),
            "tau_groups": args.tau_groups,
            "pad_vocab": not args.no_pad_vocab,
            "bf16_weights": args.bf16_weights,
            "weight_dtype": args.weight_dtype,
            "mirror_dtype": args.mirror_dtype,
            "norm_groups": args.norm_groups,
            "trace_decay": args.trace_decay,
            "learn_hidden": not args.no_learn_hidden,
            "hierarchical": bool(args.hierarchical),
            "tree_signal_scale": args.tree_signal_scale,
            "bf16_input": args.bf16_input,
            "beta": args.beta,
            "tau_read": TAU_READ,
            "input_std": INPUT_STD,
            "seed": args.seed,
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
