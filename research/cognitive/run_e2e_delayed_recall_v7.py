"""阶段四端到端 v7：双 bank + bank W 单独置信度。

预注册见 ``sweep_results/stage4_e2e_v7_prereg.md``。与 v6 dual 相同的 56 维状态，
但 WM 读出、成员、世界模型和 OOD 置信度都只在 bank W（前 28 维）上算。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v4 import _dpar_scores, _top1
from research.cognitive.run_e2e_delayed_recall_v6 import _dataset, _features
from research.cognitive.working_memory import LinearRLSReadout, fit_ridge
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v7"

N_DIMS = 4
BANK = 28
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
STATE_CONTROL_MAX = 0.35
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80


def _safe_r2(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = ((predicted - target) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    varying = total > 1e-9
    return float(np.mean(1.0 - residual[varying] / total[varying]))


def run(seeds: list[int], *, out: Path, test_episodes: int = 500) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = _dataset(2000, seed=seed)
        calibration = _dataset(test_episodes, seed=seed + 30_000)
        test = _dataset(test_episodes, seed=seed + 10_000)
        ood = _dataset(test_episodes, seed=seed + 20_000, ood=True)

        train_wm = train[2][:, :BANK]
        train_wm_previous = train[3][:, :BANK]
        calibration_wm = calibration[2][:, :BANK]
        calibration_wm_previous = calibration[3][:, :BANK]
        test_wm = test[2][:, :BANK]
        test_wm_previous = test[3][:, :BANK]
        ood_wm = ood[2][:, :BANK]
        ood_wm_previous = ood[3][:, :BANK]

        readout = LinearRLSReadout(BANK, train[4].shape[1], delta=1000.0)
        readout.fit(train_wm, train[4], epochs=EPOCHS, rng=np.random.default_rng(seed))
        members = []
        for member_index in range(MEMBERS):
            rng = np.random.default_rng(seed + 100 + member_index)
            index = rng.choice(train_wm.shape[0], size=train_wm.shape[0], replace=True)
            member = LinearRLSReadout(BANK, train[4].shape[1], delta=1000.0)
            member.fit(train_wm[index], train[4][index], epochs=EPOCHS, rng=rng)
            members.append(member)
        world = fit_ridge(train_wm_previous, train_wm)

        calibration_features = _features(
            calibration_wm,
            calibration_wm_previous,
            calibration[4],
            readout=readout,
            members=members,
            world=world,
        )
        test_features = _features(
            test_wm,
            test_wm_previous,
            test[4],
            readout=readout,
            members=members,
            world=world,
        )
        ood_features = _features(
            ood_wm,
            ood_wm_previous,
            ood[4],
            readout=readout,
            members=members,
            world=world,
        )

        predicted = readout.predict(test_wm)
        r2 = [
            _safe_r2(
                predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
                test[4][:, i * N_DIMS : (i + 1) * N_DIMS],
            )
            for i in range(3)
        ]

        def zscore(values, reference):
            return (values - reference.mean()) / max(reference.std(), 1e-8)

        calibration_confidence = -zscore(
            calibration_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(calibration_features["surprise"], calibration_features["surprise"])
        test_confidence = -zscore(
            test_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(test_features["surprise"], calibration_features["surprise"])
        ood_confidence = -zscore(
            ood_features["disagreement"], calibration_features["disagreement"]
        ) - zscore(ood_features["surprise"], calibration_features["surprise"])
        threshold = float(np.quantile(calibration_confidence, 1.0 - COVERAGE))
        answered = test_confidence >= threshold
        ood_answered = ood_confidence >= threshold
        selective_mse = (
            float(np.mean(test_features["errors"][answered])) if np.any(answered) else float("nan")
        )
        auc = auroc(
            np.concatenate([test_confidence, ood_confidence]),
            np.concatenate([np.ones_like(test_confidence), np.zeros_like(ood_confidence)]),
        )

        dpar = _dpar_scores(test[5], test[6], test[7], seed=seed + 400_000)
        dpar_best = max(dpar["pseudo"], dpar["rawnn"])

        train_values = np.stack(
            [values[pair] for values, pair in zip(train[6], train[7], strict=True)]
        )
        state_readout = LinearRLSReadout(train[2].shape[1], N_DIMS, delta=1000.0)
        state_readout.fit(
            train[2], train_values, epochs=EPOCHS, rng=np.random.default_rng(seed + 500_000)
        )
        state_predictions = state_readout.predict(test[2])
        state_top1 = float(
            np.mean(
                [
                    _top1(prediction, values) == pair
                    for prediction, values, pair in zip(
                        state_predictions, test[6], test[7], strict=True
                    )
                ]
            )
        )

        rows.append(
            {
                "seed": seed,
                "r2_a": r2[0],
                "r2_b": r2[1],
                "r2_q": r2[2],
                "r2_mean": float(np.mean(r2)),
                "selective_mse": selective_mse,
                "ood_auroc": auc,
                "ood_abstain": float(np.mean(~ood_answered)),
                "dpar_pseudo": dpar["pseudo"],
                "dpar_rawnn": dpar["rawnn"],
                "dpar_hebbian": dpar["hebbian"],
                "dpar_best": dpar_best,
                "state_control_top1": state_top1,
            }
        )
        print(
            f"seed {seed}: R2 {np.round(r2, 3).tolist()} | sel MSE {selective_mse:.3f} | "
            f"OOD {auc:.3f}/{np.mean(~ood_answered):.3f} | DPAR "
            f"{dpar['pseudo']:.3f}/{dpar['rawnn']:.3f}/{dpar['hebbian']:.3f} | "
            f"state {state_top1:.3f}",
            flush=True,
        )

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    gate = {
        "r2_mean": mean("r2_mean"),
        "selective_mse": mean("selective_mse"),
        "ood_auroc": mean("ood_auroc"),
        "ood_abstain": mean("ood_abstain"),
        "dpar_best": mean("dpar_best"),
        "state_control_top1": mean("state_control_top1"),
        "r2_ok": mean("r2_mean") >= R2_MIN,
        "selective_ok": mean("selective_mse") <= SELECTIVE_MSE_MAX,
        "ood_auroc_ok": mean("ood_auroc") >= OOD_AUROC_MIN,
        "ood_abstain_ok": mean("ood_abstain") >= OOD_ABSTAIN_MIN,
        "dpar_ok": mean("dpar_best") >= DPAR_MIN,
        "state_control_ok": mean("state_control_top1") <= STATE_CONTROL_MAX,
    }
    gate["e2e_v7_ready"] = all(
        gate[key]
        for key in (
            "r2_ok",
            "selective_ok",
            "ood_auroc_ok",
            "ood_abstain_ok",
            "dpar_ok",
            "state_control_ok",
        )
    )
    summary = {
        "experiment": "阶段四端到端 v7：双 bank + bank W 单独置信度",
        "config": {
            "seeds": seeds,
            "pairs": 4,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "bank": BANK,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "阶段四端到端 v7",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "dual：bank W 只收 A/B/Q，bank D 只收键值/查询；拼接 56 维",
                "WM 读出/成员/世界模型/OOD 置信度只读 bank W；状态控制用 56 维",
                "OOD 为 A/B ×3；阈值只在 calibration（seed+30000）上取",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v7")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        out=args.out,
        test_episodes=20 if args.smoke else args.test_episodes,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
