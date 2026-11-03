"""P4 学生：字符级语言模型，GRU / ALIF / TI-LIF 三臂，CE-only。

口径见 `prereg_p4.md` 与 `prereg_p4_amend1.md`：本批只做序列级 CE，**不做 KL**
（教师 token 级、学生字符级，对齐层没有），所以报告里不出现"自蒸馏"。

判据（写死）：P4-1 held-out top-1 ≥ bigram(0.6457) + 10pp；P4-2 不低于 GRU −5pp；
P4-3 TI-LIF(D=3) 的 CE ≤ 0.90× ALIF(D=1)（5 种子中位数）。

用法::

    python research/sel/student.py --data D:/AxoData/sel/pairs/v1_amend3/pairs.jsonl \
        --arm tilif --seed 1 --steps 2000 --out D:/AxoData/sel/p4/pilot
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

BOS, EOS, UNK = "<bos>", "<eos>", "<unk>"


# ── 数据 ────────────────────────────────────────────────────────────────────


def load_splits(path: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        splits.setdefault(row["split"], []).append(row["text"])
    return splits


def build_vocab(texts: list[str]) -> dict[str, int]:
    chars = sorted({ch for text in texts for ch in text})
    return {token: index for index, token in enumerate([BOS, EOS, UNK, *chars])}


def encode(text: str, vocab: dict[str, int], *, max_len: int) -> list[int]:
    unk = vocab[UNK]
    ids = [vocab[BOS], *(vocab.get(ch, unk) for ch in text), vocab[EOS]]
    return ids[: max_len + 1]


def batches(encoded: list[list[int]], batch_size: int, *, seed: int, pad_id: int):
    order = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(seed)).tolist()
    width = max(len(seq) for seq in encoded)
    for start in range(0, len(order), batch_size):
        chunk = [encoded[i] for i in order[start : start + batch_size]]
        inputs = torch.full((len(chunk), width - 1), pad_id, dtype=torch.long)
        targets = torch.full((len(chunk), width - 1), -100, dtype=torch.long)
        for row, seq in enumerate(chunk):
            length = len(seq) - 1
            inputs[row, :length] = torch.tensor(seq[:-1])
            targets[row, :length] = torch.tensor(seq[1:])
        yield inputs, targets


# ── 模型 ────────────────────────────────────────────────────────────────────


class _Spike(torch.autograd.Function):
    """硬前向、快 sigmoid 反向的替代梯度。"""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        return grad / (1.0 + 5.0 * x.abs()) ** 2


class GRULM(nn.Module):
    def __init__(self, vocab: int, emb: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embed(x))
        return self.readout(hidden)


class SpikingLM(nn.Module):
    """512 个自适应阈值神经元的一层循环；``signed=False`` 是 ALIF，``True`` 是 TI-LIF。"""

    def __init__(
        self,
        vocab: int,
        *,
        emb: int = 64,
        hidden: int = 512,
        signed: bool = False,
        depth: int = 1,
        tau: float = 20.0,
        tau_adaptation: float = 500.0,
        threshold: float = 0.62,
        beta: float = 0.07,
    ) -> None:
        super().__init__()
        if not signed and depth != 1:
            raise ValueError("ALIF 分支的 depth 必须是 1")
        self.embed = nn.Embedding(vocab, emb)
        self.input = nn.Linear(emb, hidden)
        self.recurrent = nn.Linear(hidden, hidden, bias=False)
        self.readout = nn.Linear(hidden, vocab)
        self.signed = signed
        self.depth = depth
        self.alpha = math.exp(-1.0 / tau)
        self.rho = math.exp(-1.0 / tau_adaptation)
        self.threshold = threshold
        self.beta = beta
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = self.input(self.embed(x))
        batch, steps, _ = current.shape
        v = current.new_zeros(batch, self.hidden)
        a = current.new_zeros(batch, self.hidden)
        s = current.new_zeros(batch, self.hidden)
        logits = []
        for t in range(steps):
            drive = current[:, t] + self.recurrent(s)
            a = self.rho * a + (s.abs() if self.signed else s)
            v = self.alpha * v + drive - s * self.threshold
            eff = self.threshold + self.beta * a
            if self.signed:
                raw = torch.clamp(v / eff, -self.depth, self.depth)
                quant = torch.round(raw)
                gated = torch.where(v.abs() >= eff, quant, torch.zeros_like(quant))
                s = raw + (gated - raw).detach()
            else:
                s = _Spike.apply(v - eff)
            logits.append(self.readout(s))
        return torch.stack(logits, dim=1)


# ── 训练与评测 ──────────────────────────────────────────────────────────────


def make_model(arm: str, vocab: int) -> nn.Module:
    if arm == "gru":
        return GRULM(vocab)
    if arm == "alif":
        return SpikingLM(vocab, signed=False)
    if arm == "tilif":
        return SpikingLM(vocab, signed=True, depth=3)
    raise ValueError(f"未知臂：{arm}")


@torch.no_grad()
def evaluate(
    model: nn.Module, encoded: list[list[int]], *, max_len: int, pad_id: int
) -> tuple[float, float]:
    model.eval()
    hits = total = 0
    ce_sum = 0.0
    width = max(len(seq) for seq in encoded)
    for start in range(0, len(encoded), 32):
        chunk = encoded[start : start + 32]
        inputs = torch.full((len(chunk), width - 1), pad_id, dtype=torch.long)
        targets = torch.full((len(chunk), width - 1), -100, dtype=torch.long)
        for row, seq in enumerate(chunk):
            length = len(seq) - 1
            inputs[row, :length] = torch.tensor(seq[:-1])
            targets[row, :length] = torch.tensor(seq[1:])
        device = next(model.parameters()).device
        targets = targets.to(device)
        logits = model(inputs.to(device))
        mask = targets != -100
        hits += int(((logits.argmax(-1) == targets) & mask).sum())
        total += int(mask.sum())
        ce_sum += float(
            nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
        ) * int(mask.sum())
    model.train()
    return hits / max(1, total), ce_sum / max(1, total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--arm", choices=("gru", "alif", "tilif"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    splits = load_splits(args.data)
    vocab = build_vocab(splits["train"])
    train = [encode(text, vocab, max_len=args.max_len) for text in splits["train"]]
    valid = [encode(text, vocab, max_len=args.max_len) for text in splits.get("valid", [])]
    test = [encode(text, vocab, max_len=args.max_len) for text in splits.get("test", [])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.arm, len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_valid, best_test, best_test_ce = 0.0, 0.0, float("inf")
    started = time.time()
    step = 0
    while step < args.steps:
        for inputs, targets in batches(train, args.batch, seed=args.seed + step, pad_id=vocab[EOS]):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.eval_every == 0 or step >= args.steps:
                valid_top1, valid_ce = evaluate(
                    model, valid, max_len=args.max_len, pad_id=vocab[EOS]
                )
                if valid_top1 > best_valid:
                    best_valid = valid_top1
                    best_test, best_test_ce = evaluate(
                        model, test, max_len=args.max_len, pad_id=vocab[EOS]
                    )
                print(
                    f"{args.arm} seed={args.seed} step={step} loss={loss.item():.3f} "
                    f"valid={valid_top1:.4f} best_test={best_test:.4f}",
                    flush=True,
                )
            if step >= args.steps:
                break

    report = {
        "arm": args.arm,
        "seed": args.seed,
        "steps": step,
        "vocab": len(vocab),
        "best_valid_top1": round(best_valid, 4),
        "test_top1": round(best_test, 4),
        "test_ce": round(best_test_ce, 4),
        "seconds": round(time.time() - started, 1),
        "device": str(device),
    }
    if device.type == "cuda":
        report["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print(json.dumps(report, ensure_ascii=False))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.arm}_seed{args.seed}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
