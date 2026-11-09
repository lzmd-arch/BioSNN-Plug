"""在**训练后**的权重上量梯度余弦：局部规则离真梯度有多远，随训练相位怎么变。

只跑单抽头读出（`tau_read=8`，即 R0b / R8 基线那一档），原因是
`sweep_results/lm_eprop_gradcos/` 的三块分解是在这个前向上测的；多时间尺度读出的
逐抽头反向滤波要另写一份，等这一轮的结论出来再说。

流程：真语料大折 → 按 R0b 的配置训练 → 在 2,000 与 8,000 步各量一次余弦
（e-prop / F1 修正 / F2 反向 EMA 三种方向，对整段 BPTT 的真梯度）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.sel.lm.grad_cosine import DiffLM, bptt_delta, cosine, eprop_delta
from research.sel.lm.model import TokenEProp
from research.sel.lm.run_m0 import BATCH, FAN_IN, make_batches
from research.sel.lm.run_r0 import (
    DATA,
    INPUT_STD,
    LR_IN,
    LR_OUT,
    MAX_LEN,
    SEED,
    TAU,
    TAU_READ,
    build_vocab,
    encode,
    load_sequences,
)

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_gradcos"
CHECKPOINTS = (2000, 8000)


def measure(model: TokenEProp, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
    """在一个 batch 上算三种方向对真梯度的余弦。"""
    common = {
        "tau": TAU,
        "tau_read": TAU_READ,
        "threshold": model.threshold,
        "dampening": model.dampening_factor,
        "n_refractory": float(model.n_refractory),
    }
    w_in = model.w_in.detach()
    w_out = model.w_out.detach()
    plain = eprop_delta(tokens, targets, w_in, w_out, fixed_mask=False, **common)
    fixed = eprop_delta(tokens, targets, w_in, w_out, fixed_mask=True, **common)
    corrected = eprop_delta(
        tokens, targets, w_in, w_out, fixed_mask=True, backward_ema=True, **common
    )
    mirror = DiffLM(w_in, w_out, fixed_mask=True, **common)
    grad = bptt_delta(mirror, tokens, targets, tokens.shape[1])
    return {
        "eprop": round(cosine(plain, grad), 4),
        "f1_fix": round(cosine(fixed, grad), 4),
        "f1_f2": round(cosine(corrected, grad), 4),
        "grad_norm": round(float(grad.norm()), 6),
        "eprop_norm": round(float(plain.norm()), 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练后权重的梯度余弦")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--sessions", default="20261018-154235-861faf")
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="gradcos_trained")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_session = load_sequences(args.data)
    sessions = sorted(by_session)
    held = args.sessions.split(",")[0]
    train_texts = [text for s in sessions if s != held for text in by_session[s]]
    test_texts = by_session[held]
    torch.manual_seed(SEED)
    vocab = build_vocab(train_texts)
    train = [encode(text, vocab, cap=MAX_LEN) for text in train_texts]
    test = [encode(text, vocab, cap=MAX_LEN) for text in test_texts]
    pad = vocab["<unk>"]
    model = TokenEProp(
        len(vocab),
        1024,
        fan_in=FAN_IN,
        tau=TAU,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=LR_IN,
        lr_out=LR_OUT,
        recurrence="none",
        generator=torch.Generator().manual_seed(SEED),
        device=device,
    )
    generator = torch.Generator().manual_seed(SEED + 17)
    # 固定的测量 batch：从测试集取前 16 条，全程序列
    probe_tokens, probe_valid = make_batches(
        test, order=list(range(min(BATCH, len(test)))), pad=pad
    )
    probe_tokens = probe_tokens.to(device)
    probe_valid = probe_valid.to(device)
    probe_targets = probe_tokens.clone()
    probe_targets[:, 0] = -100
    probe_targets = probe_targets.masked_fill(~probe_valid, -100)

    report: dict = {"config": {"updates": args.updates, "sessions": held}, "checkpoints": {}}
    step = 0
    marks = sorted(c for c in CHECKPOINTS if c <= args.updates)
    while step < args.updates:
        stop = min(marks[0] if marks else args.updates, args.updates)
        while step < stop:
            order = torch.randperm(len(train), generator=generator)[:BATCH].tolist()
            tokens, valid = make_batches(train, order=order, pad=pad)
            model.run(tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device))
            step += 1
        result = measure(model, probe_tokens, probe_targets)
        result["updates"] = step
        report["checkpoints"][str(step)] = result
        print(f"step {step}: {result}", flush=True)
        if marks:
            marks.pop(0)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
