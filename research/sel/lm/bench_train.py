"""训练吞吐基线：batch × N × V 的 tokens/s 与峰值显存（T1）。

**为什么要有这个脚本。** 之前所有"能不能训 1B"的推算都建立在猜的吞吐上。
这个脚本给出唯一可信的锚点：**同样的 `TokenEProp.run`（含权重更新），在给定
(batch, N, V, 抽头数) 下每秒处理多少 token、峰值显存多少**。

用法::

    uv run python -m research.sel.lm.bench_train                      # 默认扫描
    uv run python -m research.sel.lm.bench_train --preset 1b --updates 100
    uv run python -m research.sel.lm.bench_train --batches 16,64,256 --n-recs 1024,4096

**测的是真训练路径**：每步都调 `model.run(..., valid=...)`，与 `run_r0` 完全同一条代码，
所以数字可以直接外推。

**不要与别的 GPU 任务并跑**——那会把吞吐测成别人的。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.sel.lm.model import TokenEProp

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sweep_results" / "lm_eprop_bench"

TAUS_DEFAULT = (2.0, 8.0, 32.0, 128.0)


def run_case(
    *,
    n_rec: int,
    vocab: int,
    batch: int,
    steps: int,
    updates: int,
    taus: tuple[float, ...],
    device: torch.device,
    readout_rank: int | None = None,
    embed_rank: int | None = None,
    amp: bool = False,
) -> dict:
    """跑一个配置，返回 tokens/s、每步耗时与峰值显存。"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = TokenEProp(
        vocab,
        n_rec,
        fan_in=32,
        tau=20.0,
        tau_read=8.0,
        input_std=0.12,
        lr_in=1e-3,
        lr_out=0.5,
        recurrence="none",
        readout_taus=taus,
        readout_rank=readout_rank,
        embed_rank=embed_rank,
        amp=amp,
        generator=torch.Generator().manual_seed(0),
        device=device,
    )
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, vocab, (batch, steps), generator=gen).to(device)
    valid = torch.ones(batch, steps, dtype=torch.bool, device=device)
    width = model.readout_width
    if model.w_b is None:
        params = (
            model.w_in.numel() if model.w_in is not None else model.emb.numel() + model.proj.numel()
        ) + width * vocab
    else:
        params = (
            (
                model.w_in.numel()
                if model.w_in is not None
                else model.emb.numel() + model.proj.numel()
            )
            + int(model.w_a.numel())
            + int(model.w_b.numel())
        )

    # 预热一次（分配器/内核选择），不计时
    model.run(tokens, model.new_trace(batch), valid=valid)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(updates):
        model.run(tokens, model.new_trace(batch), valid=valid)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    processed = updates * batch * (steps - 1)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    del model, tokens, valid
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "n_rec": n_rec,
        "vocab": vocab,
        "batch": batch,
        "steps": steps,
        "updates": updates,
        "taus": len(taus),
        "width": width,
        "readout_rank": readout_rank,
        "params": params,
        "tokens": processed,
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s": round(processed / max(elapsed, 1e-9), 1),
        "ms_per_update": round(1000.0 * elapsed / max(updates, 1), 2),
        "peak_mib": round(peak, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练吞吐基线")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="bench")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", default="16,64,256")
    parser.add_argument("--n-recs", default="1024,4096")
    parser.add_argument("--vocabs", default="3001")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--readout-rank", type=int, default=0, help="A3：低秩读出 r；0=稠密")
    parser.add_argument("--embed-rank", type=int, default=0, help="A2：低秩输入 d；0=稠密")
    parser.add_argument("--amp", action="store_true", help="T3：矩阵乘用 bf16")
    parser.add_argument(
        "--preset", choices=("", "1b"), default="", help="1b = N=40000 / V=5000 / batch 256"
    )
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.preset == "1b":
        cases = [(40000, 5000, 256)]
    else:
        batches = [int(x) for x in args.batches.split(",") if x.strip()]
        n_recs = [int(x) for x in args.n_recs.split(",") if x.strip()]
        vocabs = [int(x) for x in args.vocabs.split(",") if x.strip()]
        cases = [(n, v, b) for n in n_recs for v in vocabs for b in batches]

    results = []
    for n_rec, vocab, batch in cases:
        try:
            row = run_case(
                n_rec=n_rec,
                vocab=vocab,
                batch=batch,
                steps=args.steps,
                updates=args.updates,
                taus=TAUS_DEFAULT,
                device=device,
                readout_rank=args.readout_rank or None,
                embed_rank=args.embed_rank or None,
                amp=args.amp,
            )
        except torch.cuda.OutOfMemoryError:
            row = {"n_rec": n_rec, "vocab": vocab, "batch": batch, "oom": True}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps({"device": str(device), "cases": results}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
