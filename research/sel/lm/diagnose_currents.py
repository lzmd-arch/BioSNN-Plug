"""诊断：三臂的电流分量谁在主导（S0 的事后解释，**非预注册**）。

只测初始化后的前若干步——训练会改变权重，但"随机循环的电流比输入大多少"是**初始化尺度**决定的，
那正是解释 M0/M0b/S0 差异的地方。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from research.sel.lm.model import TokenEProp
from research.sel.lm.run_m0 import BATCH, FAN_IN, N_REC, SEED, encode_split, make_batches
from research.sel.student import build_vocab, load_splits

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_s0" / "diagnostic_currents.json"
STEPS = 120


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(Path("D:/AxoData/sel/pairs/v1_amend3/pairs.jsonl"))
    vocab = build_vocab(splits["train"])
    train = encode_split(splits["train"], vocab)
    tokens, _ = make_batches(train, order=list(range(BATCH)), pad=len(vocab) - 1)
    tokens = tokens.to(device)

    report: dict[str, dict] = {}
    for mode in ("none", "random", "ssm"):
        model = TokenEProp(
            len(vocab),
            N_REC,
            fan_in=FAN_IN,
            recurrence=mode,
            generator=torch.Generator().manual_seed(SEED),
            device=device,
        )
        trace = model.new_trace(tokens.shape[0])
        direct, recurrent, ssm = [], [], []
        for t in range(min(STEPS, tokens.shape[1] - 1)):
            token = tokens[:, t]
            direct.append(
                float(model.w_in[token].abs().mean())
                if mode != "ssm"
                else float(model.D[token].abs().mean())
            )
            if mode == "random":
                recurrent.append(float(model._recurrent_current(trace.z).abs().mean()))
            if mode == "ssm":
                h = model.poles * trace.h + model.E[token]
                ssm.append(float((h @ model.C.T).abs().mean()))
            model.step(token, tokens[:, t + 1], trace, update=False)
        report[mode] = {
            "direct_current_mean_abs": round(statistics.median(direct), 4),
            "recurrent_current_mean_abs": round(statistics.median(recurrent), 4)
            if recurrent
            else None,
            "ssm_current_mean_abs": round(statistics.median(ssm), 4) if ssm else None,
        }
        if mode == "random":
            report[mode]["recurrent_over_direct"] = round(
                report[mode]["recurrent_current_mean_abs"]
                / report[mode]["direct_current_mean_abs"],
                3,
            )
        if mode == "ssm":
            report[mode]["ssm_over_direct"] = round(
                report[mode]["ssm_current_mean_abs"] / report[mode]["direct_current_mean_abs"], 3
            )
        print(mode, report[mode], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"steps": STEPS, "seed": SEED, "arms": report}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
