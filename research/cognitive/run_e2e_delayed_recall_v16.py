"""阶段四端到端 v16：输入尺度门补齐 18 格。

预注册见 ``sweep_results/stage4_e2e_v16_prereg.md``。门为 input_norm（A/B 线索的 L2 范数）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_e2e_delayed_recall_v4 import _dpar_scores, _top1
from research.cognitive.run_e2e_delayed_recall_v8 import _feature_set
from research.cognitive.run_e2e_delayed_recall_v11 import _safe_r2
from research.cognitive.run_metacognitive_gate_v4 import _evaluate
from research.cognitive.working_memory import (
    TAUS,
    LinearRLSReadout,
    encode_timeline,
    fit_ridge,
)
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e_v16"

N_DIMS = 4
MEMBERS = 5
EPOCHS = 20
COVERAGE = 0.80
COVERAGE_LOW = 0.78
COVERAGE_HIGH = 0.82

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
DPAR_MIN = 0.80
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80

PAIRS = (4, 8)
SCALES = (2.0, 3.0, 5.0)
NOISES = (0.1, 0.3, 0.5)
CONDITIONS = tuple((pairs, scale, noise) for pairs in PAIRS for scale in SCALES for noise in NOISES)


def _episode(
    rng: np.random.Generator,
    *,
    pairs: int,
    scale: float,
    cue_noise: float,
    ood: bool,
) -> tuple[np.ndarray, ...]:
    cue_a = rng.normal(size=N_DIMS)
    cue_b = rng.normal(size=N_DIMS)
    if ood:
        cue_a = cue_a * scale
        cue_b = cue_b * scale
    else:
        cue_a = cue_a + rng.normal(scale=cue_noise, size=N_DIMS)
        cue_b = cue_b + rng.normal(scale=cue_noise, size=N_DIMS)
    choose_b = bool(rng.integers(0, 2))
    query = np.array([0.0, 1.0, 0.0, 0.0] if choose_b else [1.0, 0.0, 0.0, 0.0])
    keys = rng.normal(size=(pairs, N_DIMS))
    values = rng.normal(size=(pairs, N_DIMS))
    pair = int(rng.integers(0, pairs))

    wm_cues = [(0, 4, cue_a), (15, 19, cue_b), (30, 34, query)]
    dpar_cues: list[tuple[int, int, np.ndarray]] = []
    for index in range(pairs):
        dpar_cues.append((40 + 10 * index, 44 + 10 * index, keys[index]))
        dpar_cues.append((45 + 10 * index, 49 + 10 * index, values[index]))
    dpar_cues.append((40 + 10 * pairs, 44 + 10 * pairs, keys[pair]))
    final_step = 40 + 10 * pairs + 5

    taus = (*TAUS, 400) if pairs == 8 else TAUS
    wm_state = encode_timeline(wm_cues, rng=rng, test_step=final_step, taus=taus)
    wm_previous = encode_timeline(wm_cues, rng=rng, test_step=final_step - 1, taus=taus)
    dpar_state = encode_timeline(dpar_cues, rng=rng, test_step=final_step)
    dpar_previous = encode_timeline(dpar_cues, rng=rng, test_step=final_step - 1)
    input_norm = float(max(np.linalg.norm(cue_a), np.linalg.norm(cue_b)))
    target = np.concatenate([cue_a, cue_b, query])
    return wm_state, wm_previous, dpar_state, dpar_previous, target, keys, values, input_norm, pair


def _dataset(
    episodes: int,
    *,
    seed: int,
    pairs: int,
    scale: float,
    cue_noise: float,
    ood: bool = False,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    fields = [[] for _ in range(8)]
    pair_values = []
    for _ in range(episodes):
        *values, pair = _episode(rng, pairs=pairs, scale=scale, cue_noise=cue_noise, ood=ood)
        for target, value in zip(fields, values, strict=True):
            target.append(value)
        pair_values.append(pair)
    return (*tuple(np.stack(field) for field in fields), np.asarray(pair_values))


def _checks(metrics: dict[str, float], pairs: int) -> dict[str, bool]:
    return {
        "r2_ok": metrics["r2_mean"] >= R2_MIN,
        "selective_ok": metrics["selective_mse"] <= SELECTIVE_MSE_MAX,
        "dpar_ok": metrics["dpar_best"] >= DPAR_MIN,
        "state_control_ok": metrics["state_control_top1"] <= 1.0 / pairs + 0.10,
        "ood_auroc_ok": metrics["ood_auroc"] >= OOD_AUROC_MIN,
        "ood_abstain_ok": metrics["ood_abstain"] >= OOD_ABSTAIN_MIN,
        "coverage_ok": COVERAGE_LOW <= metrics["test_coverage"] <= COVERAGE_HIGH,
    }


def _input_gate(
    calibration: np.ndarray, test: np.ndarray, ood: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = float(np.mean(calibration))
    scale = max(float(np.std(calibration)), 1e-8)

    def confidence(values: np.ndarray) -> np.ndarray:
        return -(values - mean) / scale

    return confidence(calibration), confidence(test), confidence(ood)


def run(
    seeds: list[int],
    *,
    out: Path,
    test_episodes: int = 500,
    conditions: tuple[tuple[int, float, float], ...] = CONDITIONS,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for pairs, scale, cue_noise in conditions:
        name = f"K{pairs}-s{scale:g}-n{cue_noise:g}"
        rows = []
        for seed in seeds:
            train = _dataset(2000, seed=seed, pairs=pairs, scale=scale, cue_noise=cue_noise)
            calibration = _dataset(
                test_episodes,
                seed=seed + 30_000,
                pairs=pairs,
                scale=scale,
                cue_noise=cue_noise,
            )
            test = _dataset(
                test_episodes,
                seed=seed + 10_000,
                pairs=pairs,
                scale=scale,
                cue_noise=cue_noise,
            )
            ood = _dataset(
                test_episodes,
                seed=seed + 20_000,
                pairs=pairs,
                scale=scale,
                cue_noise=cue_noise,
                ood=True,
            )
            readout = LinearRLSReadout(train[0].shape[1], train[4].shape[1], delta=1000.0)
            readout.fit(train[0], train[4], epochs=EPOCHS, rng=np.random.default_rng(seed))
            members = []
            for member_index in range(MEMBERS):
                rng = np.random.default_rng(seed + 100 + member_index)
                index = rng.choice(train[0].shape[0], size=train[0].shape[0], replace=True)
                member = LinearRLSReadout(train[0].shape[1], train[4].shape[1], delta=1000.0)
                member.fit(train[0][index], train[4][index], epochs=EPOCHS, rng=rng)
                members.append(member)
            world = fit_ridge(train[1], train[0])

            calibration_features = _feature_set(
                calibration[0],
                calibration[1],
                calibration[4],
                readout=readout,
                members=members,
                world=world,
            )
            test_features = _feature_set(
                test[0], test[1], test[4], readout=readout, members=members, world=world
            )
            ood_features = _feature_set(
                ood[0], ood[1], ood[4], readout=readout, members=members, world=world
            )
            confidence = _input_gate(calibration[7], test[7], ood[7])
            metrics = _evaluate(
                calibration_features, test_features, ood_features, confidence=confidence
            )
            predicted = readout.predict(test[0])
            r2 = [
                _safe_r2(
                    predicted[:, i * N_DIMS : (i + 1) * N_DIMS],
                    test[4][:, i * N_DIMS : (i + 1) * N_DIMS],
                )
                for i in range(3)
            ]
            dpar = _dpar_scores(test[5], test[6], test[8], seed=seed + 400_000)
            train_values = np.stack(
                [values[pair] for values, pair in zip(train[6], train[8], strict=True)]
            )
            train_control = np.concatenate([train[0], train[2]], axis=1)
            test_control = np.concatenate([test[0], test[2]], axis=1)
            state_readout = LinearRLSReadout(train_control.shape[1], N_DIMS, delta=1000.0)
            state_readout.fit(
                train_control,
                train_values,
                epochs=EPOCHS,
                rng=np.random.default_rng(seed + 500_000),
            )
            state_predictions = state_readout.predict(test_control)
            state_top1 = float(
                np.mean(
                    [
                        _top1(prediction, values) == pair
                        for prediction, values, pair in zip(
                            state_predictions, test[6], test[8], strict=True
                        )
                    ]
                )
            )
            rows.append(
                {
                    "seed": seed,
                    "r2_mean": float(np.mean(r2)),
                    "selective_mse": metrics["selective_mse"],
                    "test_coverage": metrics["test_coverage"],
                    "ood_auroc": metrics["ood_auroc"],
                    "ood_abstain": metrics["ood_abstain"],
                    "dpar_best": max(dpar["pseudo"], dpar["rawnn"]),
                    "state_control_top1": state_top1,
                }
            )
        mean = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "r2_mean",
                "selective_mse",
                "test_coverage",
                "ood_auroc",
                "ood_abstain",
                "dpar_best",
                "state_control_top1",
            )
        }
        checks = _checks(mean, pairs)
        results[name] = {
            "pairs": pairs,
            "scale": scale,
            "cue_noise": cue_noise,
            "per_seed": rows,
            "mean": mean,
            "checks": checks,
            "ready": all(checks.values()),
        }
        print(
            f"{name}: R² {mean['r2_mean']:.3f} MSE {mean['selective_mse']:.3f} "
            f"cov {mean['test_coverage']:.3f} abstain {mean['ood_abstain']:.3f} "
            f"AUROC {mean['ood_auroc']:.3f} DPAR {mean['dpar_best']:.3f} "
            f"state {mean['state_control_top1']:.3f} ready={results[name]['ready']}",
            flush=True,
        )

    summary = {
        "experiment": "阶段四端到端 v16：输入尺度门 18 格",
        "config": {
            "seeds": seeds,
            "test_episodes": test_episodes,
            "coverage": COVERAGE,
            "conditions": [f"K{p}-s{s:g}-n{n:g}" for p, s, n in conditions],
            "gate": "input_norm",
            "k8_arm": "longtau",
        },
        "conditions": results,
        "v16_ready": all(result["ready"] for result in results.values()),
        "repro": collect(
            "阶段四端到端 v16",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "K=4/8 × s=2/3/5 × sigma=0.1/0.3/0.5；input_norm 门",
                "K=4 bank W 28 维；K=8 longtau 32 维（tau=400）",
                "OOD 尺度 s、in-dist 噪声 sigma；阈值只在 calibration 上取",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"v16_ready": summary["v16_ready"]}))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四端到端 v16")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(
        parse_seeds(args.seeds),
        out=args.out,
        test_episodes=20 if args.smoke else args.test_episodes,
        conditions=(CONDITIONS[0], CONDITIONS[-1]) if args.smoke else CONDITIONS,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
