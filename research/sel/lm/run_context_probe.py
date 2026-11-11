"""延迟回忆探针：**量出模型能记住多少步**（`design-context-mechanisms.md` 的第 0 步）。

序列 ``[A][s][填充 N 步][Q][s]``：预测 ``Q`` 的下一个 token 时，必须从 N 步之前取回 ``s``。
指标是「答案位置 top-1」对 N 的曲线——**50% 交叉点就是记忆长度**。

判据（写死在设计文档里）：

- **P-0**：基线在 N=1 上 ≥ 0.90，且交叉点落在 τ_read 附近（±1 档）——否则先修探针；
- **P-1**：任何机制要过「交叉点至少翻倍」；
- **P-2**：且 N ≤ 8 的点掉 ≤ 1pp。

用法::

    uv run python -m research.sel.lm.run_context_probe --updates 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.sel.lm.model import TokenEProp

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_context"

N_SYMBOL = 32
MARK_A, MARK_Q, PAD = 32, 33, 34
VOCAB = 35
TRAIN_N = (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 256)
TEST_N = (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 256, 384, 512)


def build(symbols: torch.Tensor, delays: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, L)`` 的 token 与「答案位置」的 bool 掩码。"""
    batch = symbols.numel()
    width = int(delays.max()) + 4
    tokens = torch.full((batch, width), PAD, dtype=torch.long)
    tokens[:, 0] = MARK_A
    tokens[:, 1] = symbols
    for row in range(batch):
        q = int(delays[row]) + 2
        tokens[row, q] = MARK_Q
        tokens[row, q + 1] = symbols[row]
    answer = torch.zeros(batch, width, dtype=torch.bool)
    answer[torch.arange(batch), delays + 3] = True
    return tokens, answer


@torch.no_grad()
def accuracy(model, tokens: torch.Tensor, answer: torch.Tensor, device) -> float:
    trace = model.new_trace(tokens.shape[0])
    tokens = tokens.to(device)
    answer = answer.to(device)
    hits = counted = 0
    for t in range(tokens.shape[1] - 1):
        model.step(tokens[:, t], tokens[:, t + 1], trace, update=False)
        rows = answer[:, t + 1]
        if bool(rows.any()):
            probs = model._token_probabilities(model._features(trace))
            pred = probs[rows].argmax(dim=-1)
            hits += int((pred == tokens[:, t + 1][rows]).sum())
            counted += int(rows.sum())
    return hits / max(counted, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="延迟回忆探针：量记忆长度")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-rec", type=int, default=512)
    ap.add_argument("--readout-taus", default="2,128")
    ap.add_argument("--test-per-n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="probe_base")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(args.seed)
    taus = tuple(float(x) for x in args.readout_taus.split(",") if x.strip())
    model = TokenEProp(
        VOCAB,
        args.n_rec,
        recurrence="none",
        readout_taus=taus,
        lr_in=1e-3,
        lr_out=0.5,
        tau=20.0,
        input_std=0.12,
        generator=torch.Generator().manual_seed(args.seed),
        device=device,
    )
    print(f"探针：V={VOCAB} n_rec={args.n_rec} taus={taus} 更新 {args.updates}", flush=True)

    running = 0.0
    for step in range(1, args.updates + 1):
        delays = torch.tensor(TRAIN_N)[torch.randint(0, len(TRAIN_N), (args.batch,), generator=gen)]
        symbols = torch.randint(0, N_SYMBOL, (args.batch,), generator=gen)
        tokens, answer = build(symbols, delays)
        # **状态与梯度必须分开**：填充位要更新膜电位（否则网络不发放），但不该产生梯度
        # （否则读出被 259 个"预测填充"淹没，实测整体 CE 0.065 而答案位置全错）。
        trace = model.new_trace(tokens.shape[0])
        rows = tokens.shape[0]
        tokens = tokens.to(device)
        answer_dev = answer.to(device)
        # **循环里不做 .item() / bool(...)**：每步一次 GPU→CPU 同步会把一次更新拖到
        # 几十毫秒（实测 500 步卡了 20 分钟）。全部张量累加，循环结束只取一次。
        acc = torch.zeros((), device=device)
        for t in range(tokens.shape[1] - 1):
            learn = answer_dev[:, t + 1]
            loss, _ = model.step(
                tokens[:, t],
                tokens[:, t + 1],
                trace,
                update=True,
                all_valid=True,
                learn_mask=learn,
            )
            acc = acc + (loss * learn.to(loss.dtype)).sum()
        running += float(acc) / max(rows, 1)
        if step % 500 == 0:
            print(f"  {step} 步：近 500 步答案位 CE {running / 500:.3f}", flush=True)
            running = 0.0

    curve = []
    for delay in TEST_N:
        symbols = torch.randint(0, N_SYMBOL, (args.test_per_n,), generator=gen)
        tokens, answer = build(symbols, torch.full((args.test_per_n,), delay))
        curve.append({"delay": delay, "top1": round(accuracy(model, tokens, answer, device), 4)})
        print(f"  N={delay:4d}: top1 {curve[-1]['top1']:.4f}", flush=True)

    first = next((c for c in curve if c["top1"] >= 0.9), None)
    cross = next((c["delay"] for c in curve if c["top1"] < 0.5), None)
    print(
        f"[判据] N=1 是否 ≥0.90：{first is not None and first['delay'] == 1}；50% 交叉点：{cross}",
        flush=True,
    )
    result = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "curve": curve,
        "p0_short_ok": bool(first is not None and first["delay"] == 1),
        "cross_50": cross,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {args.out / (args.tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
