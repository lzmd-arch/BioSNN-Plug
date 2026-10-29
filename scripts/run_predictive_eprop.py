"""Predictive E-prop 正弦生成任务：预注册判据见 sweep_results/predictive_eprop_prereg.md。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.common.provenance import collect  # noqa: E402
from research.eprop.predictive import PredictiveEProp  # noqa: E402

OUT = REPO / "sweep_results" / "predictive_eprop"


def sine(parameters: dict, steps: int, batch: int = 1) -> torch.Tensor:
    """按参数生成正弦，``(steps, batch, 1)``。"""
    t = torch.arange(steps, dtype=torch.float32)
    wave = (
        parameters["amplitude"]
        * torch.sin(2 * math.pi * t / parameters["period"] + parameters["phase"])
        + parameters["offset"]
    )
    return wave.reshape(steps, 1, 1).expand(steps, batch, 1).contiguous()


def sample_parameters(generator: torch.Generator) -> dict:
    return {
        "amplitude": float(torch.empty(1).uniform_(0.5, 1.0, generator=generator)),
        "period": float(torch.empty(1).uniform_(200.0, 400.0, generator=generator)),
        "phase": float(torch.empty(1).uniform_(0.0, 2 * math.pi, generator=generator)),
        "offset": 0.0,
    }


def dtw(a: np.ndarray, b: np.ndarray, band: int = 100) -> float:
    """Sakoe-Chiba 带约束 DTW，按 (len(a)+len(b)) 归一，便于跨长度比较。"""
    n, m = len(a), len(b)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        low, high = max(1, i - band), min(m, i + band)
        for j in range(low, high + 1):
            current[j] = abs(a[i - 1] - b[j - 1]) + min(
                previous[j], current[j - 1], previous[j - 1]
            )
        previous = current
    return float(previous[m] / (n + m))


def free_running_metrics(model: PredictiveEProp, parameters: dict, steps: int = 2000) -> dict:
    """自由运行 2000 ms，取后 1000 ms 与目标正弦比较。"""
    rollout = model.simulate(steps=steps)
    generated = rollout.prediction[:, 0, 0].detach().numpy()[-1000:]
    target = sine(parameters, steps)[:, 0, 0].numpy()[-1000:]
    target_std = float(np.std(target))
    return {
        "dtw": dtw(generated, target),
        "amplitude_ratio": float(np.std(generated) / target_std) if target_std > 0 else 0.0,
        "mean_spike_rate_hz": float(rollout.spikes.mean().item()) * 1000.0,
    }


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    generator = torch.Generator().manual_seed(seed)
    parameters = sample_parameters(generator)
    evaluation = sine(parameters, args.eval_steps)

    model = PredictiveEProp(
        n_signal=1,
        n_rec=args.n_rec,
        beta=args.beta,
        gain=args.gain,
        learning_rate_rec=args.learning_rate_rec,
        learning_rate_out=args.learning_rate_out,
        lambda_reg=args.lambda_reg,
        generator=torch.Generator().manual_seed(seed),
    )
    baseline = PredictiveEProp(
        n_signal=1,
        n_rec=args.n_rec,
        beta=args.beta,
        gain=args.gain,
        generator=torch.Generator().manual_seed(seed),
    )
    initial = model.reconstruction_mse(evaluation)
    baseline_metrics = free_running_metrics(baseline, parameters, args.eval_steps)

    started = time.perf_counter()
    losses = []
    for epoch in range(args.epochs):
        trial_generator = torch.Generator().manual_seed(seed * 100_003 + epoch)
        signals = [sine(sample_parameters(trial_generator), args.steps) for _ in range(args.trials)]
        batch = torch.cat(signals, dim=1)
        progress = epoch / max(1, args.epochs)
        if args.curriculum_fraction > 0 and progress >= 1.0 - args.curriculum_fraction:
            tail = (progress - (1.0 - args.curriculum_fraction)) / args.curriculum_fraction
            error_gain = max(0.0, 1.0 - tail)
        else:
            error_gain = 1.0
        # 误差回注丢弃：一部分 trial 完全关掉误差输入，逼网络靠自身反馈与前向预测的
        # 误差学习——否则它会学到「抄误差通道里的目标」这条自由运行时失效的捷径。
        error_gain = (
            0.0
            if (
                args.error_dropout
                and torch.rand(1, generator=trial_generator).item() < args.error_dropout
            )
            else 1.0
        )
        losses.append(model.update(batch, chunk=args.chunk, error_gain=error_gain))
        if (epoch + 1) % args.log_every == 0:
            print(
                f"seed {seed} epoch {epoch + 1}/{args.epochs}: mse {losses[-1]:.4f}, "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )

    final = model.reconstruction_mse(evaluation)
    metrics = free_running_metrics(model, parameters, args.eval_steps)
    criteria = {
        "reconstruction": final <= 0.1 * initial,
        "amplitude": 0.5 <= metrics["amplitude_ratio"] <= 2.0,
        "free_running_dtw": metrics["dtw"] <= 0.5 * baseline_metrics["dtw"],
    }
    if args.save_models:
        torch.save(
            {"w_rec": model.cell.w_rec.detach(), "w_out": model.w_out.detach()},
            args.out_dir / f"seed{seed}_model.pt",
        )
    return {
        "seed": seed,
        "parameters": parameters,
        "initial_mse": initial,
        "final_mse": final,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "free_running": metrics,
        "baseline_free_running": baseline_metrics,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "elapsed_s": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predictive E-prop 正弦生成")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-rec", type=int, default=300)
    parser.add_argument("--beta", type=float, default=0.07)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate-rec", type=float, default=1e-4)
    parser.add_argument("--learning-rate-out", type=float, default=1e-2)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--chunk", type=int, default=10, help="每多少毫秒更新一次权重")
    parser.add_argument(
        "--save-models", action="store_true", help="把每种子权重存到 out-dir（默认不存）"
    )
    parser.add_argument(
        "--curriculum-fraction",
        type=float,
        default=0.0,
        help="最后这个比例的 epoch 把误差回注增益线性降到 0（0=关闭）",
    )
    parser.add_argument(
        "--error-dropout", type=float, default=0.0, help="整条 trial 关闭误差输入的概率"
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s]
    results = [run_seed(seed, args) for seed in seeds]
    summary = {
        "kind": "Predictive E-prop sine generation; our adaptation, not a replication",
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "repro": collect(
            "Predictive E-prop：正弦生成",
            seeds=",".join(str(seed) for seed in seeds),
            notes=["按原文描述的改编；公式未逐字取得", "更新每 chunk 步一次，痕迹跨更新保持"],
        ).render_block(),
        "seeds": results,
        "ready": all(result["pass"] for result in results),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({r["seed"]: r["criteria"] for r in results}, ensure_ascii=False, indent=2))
    print("ready:", summary["ready"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
