"""阶段四端到端：延迟线索问答 + OOD 弃权（预注册见 sweep_results/stage4_e2e_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.episodic_memory import RawNearestNeighbor
from research.cognitive.working_memory import (
    LinearRLSReadout,
    average_r2,
    encode_timeline,
    fit_ridge,
    predict,
)
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_e2e"

N_DIMS = 4
CUE_WINDOWS = ((0, 4), (15, 19), (30, 34))
TEST_STEP = 60
MEMBERS = 5
EPOCHS = 20
NOISE = 0.0
COVERAGE = 0.80

R2_MIN = 0.90
SELECTIVE_MSE_MAX = 0.10
OOD_AUROC_MIN = 0.75
OOD_ABSTAIN_MIN = 0.80
EPISODIC_MSE_MAX = 0.10


def _episode(rng: np.random.Generator, *, ood: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cue_a = rng.normal(size=N_DIMS)
    cue_b = rng.normal(size=N_DIMS)
    if ood:
        cue_a = cue_a * 3.0
        cue_b = cue_b * 3.0
    choose_b = bool(rng.integers(0, 2))
    query = np.array([0.0, 1.0, 0.0, 0.0] if choose_b else [1.0, 0.0, 0.0, 0.0])
    cues = [
        (0, 4, cue_a),
        (15, 19, cue_b),
        (30, 34, query),
    ]
    state_59 = encode_timeline(cues, rng=rng, test_step=59)
    state_60 = encode_timeline(cues, rng=rng, test_step=60)
    target = np.concatenate([cue_a, cue_b, query])
    return state_60, state_59, target


def _dataset(
    episodes: int, *, seed: int, ood: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states, previous, targets = [], [], []
    for _ in range(episodes):
        state, previous_state, target = _episode(rng, ood=ood)
        states.append(state)
        previous.append(previous_state)
        targets.append(target)
    return np.stack(states), np.stack(previous), np.stack(targets)


def _r2(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    return [
        average_r2(
            predicted[:, i * N_DIMS : (i + 1) * N_DIMS], target[:, i * N_DIMS : (i + 1) * N_DIMS]
        )
        for i in range(3)
    ]


def _answer(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    choose_b = target[:, 8] < 0.5  # Q 的 one-hot：A 在前，B 在后
    clues = np.where(choose_b[:, None], predicted[:, 4:8], predicted[:, 0:4])
    truth = np.where(choose_b[:, None], target[:, 4:8], target[:, 0:4])
    return ((clues - truth) ** 2).mean(axis=1)


def run(seeds: list[int], *, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_states, train_previous, train_targets = _dataset(2000, seed=seed)
        test_states, test_previous, test_targets = _dataset(500, seed=seed + 10_000)
        ood_states, ood_previous, ood_targets = _dataset(500, seed=seed + 20_000, ood=True)

        readout = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
        readout.fit(train_states, train_targets, epochs=EPOCHS, rng=np.random.default_rng(seed))
        predicted = readout.predict(test_states)
        ood_predicted = readout.predict(ood_states)
        r2 = _r2(predicted, test_targets)
        errors = _answer(predicted, test_targets)
        ood_errors = _answer(ood_predicted, ood_targets)

        members = []
        for member_index in range(MEMBERS):
            rng = np.random.default_rng(seed + 100 + member_index)
            index = rng.choice(train_states.shape[0], size=train_states.shape[0], replace=True)
            member = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
            member.fit(train_states[index], train_targets[index], epochs=EPOCHS, rng=rng)
            members.append(member)
        test_predictions = np.stack([member.predict(test_states) for member in members])
        ood_predictions = np.stack([member.predict(ood_states) for member in members])
        disagreement = test_predictions.std(axis=0).mean(axis=1)
        ood_disagreement = ood_predictions.std(axis=0).mean(axis=1)

        world = fit_ridge(train_previous, train_states)
        surprise = ((predict(test_previous, world) - test_states) ** 2).mean(axis=1)
        ood_surprise = ((predict(ood_previous, world) - ood_states) ** 2).mean(axis=1)

        def zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
            return (values - reference.mean()) / max(reference.std(), 1e-8)

        confidence = -zscore(disagreement, disagreement) - zscore(surprise, surprise)
        ood_confidence = -zscore(ood_disagreement, disagreement) - zscore(ood_surprise, surprise)
        threshold = float(np.quantile(confidence, 1.0 - COVERAGE))
        answered = confidence >= threshold
        ood_answered = ood_confidence >= threshold
        selective_mse = float(np.mean(errors[answered])) if np.any(answered) else float("nan")
        ood_abstain = float(np.mean(~ood_answered))
        # 正类 = in-dist（置信度越高越像 in-dist）。
        auc = auroc(
            np.concatenate([confidence, ood_confidence]),
            np.concatenate([np.ones_like(confidence), np.zeros_like(ood_confidence)]),
        )

        episodic = RawNearestNeighbor()
        episodic.store(train_states, train_targets[:, :8])
        retrieved = episodic.recall(test_states)
        episodic_mse = float(np.mean((retrieved - test_targets[:, :8]) ** 2))
        rows.append(
            {
                "seed": seed,
                "r2_a": r2[0],
                "r2_b": r2[1],
                "r2_q": r2[2],
                "r2_mean": float(np.mean(r2)),
                "selective_mse": selective_mse,
                "ood_auroc": auc,
                "ood_abstain": ood_abstain,
                "ood_answer_mse": float(np.mean(ood_errors)),
                "episodic_mse": episodic_mse,
            }
        )
        print(
            f"seed {seed}: R2 {np.round(r2, 3).tolist()} | sel MSE {selective_mse:.3f} | "
            f"OOD AUROC {auc:.3f} abstain {ood_abstain:.3f} | epi MSE {episodic_mse:.3f}",
            flush=True,
        )

    mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
    gate = {
        "r2_mean": mean("r2_mean"),
        "selective_mse": mean("selective_mse"),
        "ood_auroc": mean("ood_auroc"),
        "ood_abstain": mean("ood_abstain"),
        "episodic_mse": mean("episodic_mse"),
        "r2_ok": mean("r2_mean") >= R2_MIN,
        "selective_ok": mean("selective_mse") <= SELECTIVE_MSE_MAX,
        "ood_auroc_ok": mean("ood_auroc") >= OOD_AUROC_MIN,
        "ood_abstain_ok": mean("ood_abstain") >= OOD_ABSTAIN_MIN,
        "episodic_ok": mean("episodic_mse") <= EPISODIC_MSE_MAX,
    }
    gate["e2e_ready"] = all(
        gate[key]
        for key in ("r2_ok", "selective_ok", "ood_auroc_ok", "ood_abstain_ok", "episodic_ok")
    )
    summary = {
        "experiment": "阶段四端到端：延迟线索问答 + OOD 弃权",
        "config": {
            "seeds": seeds,
            "coverage": COVERAGE,
            "members": MEMBERS,
            "epochs": EPOCHS,
            "ood_scale": 3.0,
        },
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "阶段四端到端：延迟线索问答 + OOD 弃权",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "A/B/Q 三线索；OOD 为 A/B ×3；训练只用 in-dist",
                "门控 = 成员分歧 + 世界模型一步惊喜；阈值取 in-dist 80% 覆盖",
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
    parser = argparse.ArgumentParser(description="阶段四端到端任务")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    run(parse_seeds(args.seeds), out=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
