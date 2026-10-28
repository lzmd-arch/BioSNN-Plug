"""液态 LIF 的 sMNIST/e-prop runner（预注册见 sweep_results/liquid_lif_prereg.md）。

复用 W2 的 ``train_sequential.train``，只换 learner；训练逻辑、数据编码、lr/epoch
全部沿用 W2。正式 5 种子 250 epoch 约 30 GPU 小时，先跑 ``--smoke``。
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

from research.common.device import describe_device, select_device
from research.common.seeding import parse_seeds
from research.eprop.liquid import LiquidEPropLearner
from research.eprop.train_sequential import build_parser, train

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "liquid_lif"


def build_parser_for_liquid() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "液态 LIF：sMNIST/e-prop（250 epoch × 5 种子）"
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--a", type=float, default=1.0 / 20.0)
    parser.add_argument("--b", type=float, default=0.15)
    parser.add_argument("--rate-decay", type=float, default=0.9)
    parser.add_argument("--tau-min", type=float, default=5.0)
    parser.add_argument("--tau-max", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser_for_liquid().parse_args(argv)
    device = select_device(args.device)
    seeds = parse_seeds(args.seeds)
    print(f"液态 LIF：{len(seeds)} 个种子，设备 {describe_device(device).render()}")
    records = []
    started = time.perf_counter()
    for seed in seeds:
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        run_args.beta = 0.0  # 液态探针用 LIF；不让 W2 的 ALIF 默认值混进记录
        history: list[dict[str, float]] = []
        accuracy, active, n_classes = train(
            run_args,
            device,
            history,
            learner_cls=LiquidEPropLearner,
            learner_kwargs={
                "a": args.a,
                "b": args.b,
                "rate_decay": args.rate_decay,
                "tau_min": args.tau_min,
                "tau_max": args.tau_max,
            },
        )
        record = {
            "seed": seed,
            "test_accuracy": accuracy,
            "active_fraction": active,
            "n_classes": n_classes,
            "curve": history,
            "config": {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "n_rec": args.n_rec,
                "batch_size": args.batch_size,
                "learning_rate_in": args.learning_rate_in,
                "learning_rate_rec": args.learning_rate_rec,
                "learning_rate_out": args.learning_rate_out,
                "a": args.a,
                "b": args.b,
                "rate_decay": args.rate_decay,
                "tau_min": args.tau_min,
                "tau_max": args.tau_max,
                "smoke": args.smoke,
            },
        }
        records.append(record)
        if not args.smoke:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"seed{seed}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(
            f"种子 {seed}: test {accuracy:.4f} active {active:.4f}",
            flush=True,
        )
    if len(records) > 1:
        print(
            f"均值 test {statistics.fmean(r['test_accuracy'] for r in records):.4f} "
            f"active {statistics.fmean(r['active_fraction'] for r in records):.4f} "
            f"（{time.perf_counter() - started:.0f} s）"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
