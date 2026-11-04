"""评测流上的任务边界检测 v3：两种统计量并用（判据见 `..._v3_prereg.md`）。

v1 的原始均值差与 v2 的白化均值偏移**盲区互补**（量过：1000/4000 只有 v1 看得见，
误报只有 v2 能清零）。v3 让两者各自按自己的**流级**阈值归一化，取最大值报警，
并用三个额外种子检验它不是拟合到某一条流。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.continual.data import load_cifar100
from research.continual.detect_task_boundaries_v2 import _whitener

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "continual_v3_boundary_detection_v3"
DATA_ROOT = REPO_ROOT / "data" / "cifar100"
CACHE = REPO_ROOT / "data" / "_features"

SEED = 0
EXTRA_SEEDS = (1, 2, 3)
TASKS = 10
BLOCK = 1000
WINDOW = 64
STEP = 10
PEAK_RADIUS = 50
TOLERANCE = 100
NULL_STREAMS = 50
NULL_QUANTILE = 99.0

#: 判据（改这里必须改预注册）。
MAX_NULL_ALARMS = 0.2
MIN_F1 = 0.80
MIN_ROUTING = 0.80
CONTROL_MAX_F1 = 0.30


def _statistics(
    stream: np.ndarray, factor: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (位置, s1 原始, s2 白化)。窗口均值用累积和算。"""
    positions = np.arange(WINDOW, stream.shape[0] - WINDOW, STEP)
    cumulative = np.concatenate([np.zeros((1, stream.shape[1])), np.cumsum(stream, axis=0)])
    after = (cumulative[positions + WINDOW] - cumulative[positions]) / WINDOW
    before = (cumulative[positions] - cumulative[positions - WINDOW]) / WINDOW
    difference = after - before
    raw = np.linalg.norm(difference, axis=1)
    whitened = torch.linalg.solve_triangular(
        factor, torch.tensor(difference, dtype=torch.float64).T, upper=False
    )
    return positions, raw, whitened.norm(dim=0).numpy()


def _score(raw: np.ndarray, whitened: np.ndarray, tau: tuple[float, float]) -> np.ndarray:
    return np.maximum(raw / tau[0], whitened / tau[1])


def _peaks(positions: np.ndarray, score: np.ndarray) -> list[int]:
    peaks: list[int] = []
    span = PEAK_RADIUS // STEP
    for index, value in enumerate(score):
        if value <= 1.0:
            continue
        low, high = max(0, index - span), min(len(score), index + span + 1)
        if value >= score[low:high].max():
            peak = int(positions[index])
            if not peaks or peak - peaks[-1] > PEAK_RADIUS:
                peaks.append(peak)
    return peaks


def _f1(detected: list[int], truth: list[int], tolerance: int) -> float:
    matched: set[int] = set()
    hits = 0
    for peak in detected:
        candidates = [t for t in truth if abs(t - peak) <= tolerance and t not in matched]
        if candidates:
            matched.add(min(candidates, key=lambda t: abs(t - peak)))
            hits += 1
    precision = hits / len(detected) if detected else 0.0
    recall = len(matched) / len(truth) if truth else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _routing(peaks: list[int], owner: np.ndarray) -> float:
    edges = [0, *peaks, owner.shape[0]]
    correct = 0
    for index in range(min(TASKS, len(edges) - 1)):
        segment = owner[edges[index] : edges[index + 1]]
        correct += int((segment == index).sum())
    return correct / owner.shape[0]


def _null_streams(train_x: np.ndarray, train_task: np.ndarray, rng: np.random.Generator):
    for index in range(NULL_STREAMS):
        pool = train_x[train_task == index % TASKS]
        yield np.concatenate(
            [pool[rng.integers(0, pool.shape[0], size=BLOCK)] for _ in range(TASKS)]
        )


def _test_stream(test_x: np.ndarray, test_task: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    blocks, owners = [], []
    for task in range(TASKS):
        pool = test_x[test_task == task]
        blocks.append(pool[rng.permutation(pool.shape[0])])
        owners.append(np.full(pool.shape[0], task))
    return np.concatenate(blocks), np.concatenate(owners)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="任务边界检测 v3")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="boundary_v3")
    args = parser.parse_args(argv)

    train, test = load_cifar100(DATA_ROOT)
    train_x = np.load(CACHE / "resnet18_train.npz")["features"].astype(np.float64)
    test_x = np.load(CACHE / "resnet18_test.npz")["features"].astype(np.float64)
    train_task = np.asarray(train.y, dtype=np.int64) // 10
    test_task = np.asarray(test.y, dtype=np.int64) // 10
    factor = _whitener(train_x, train_task)

    # 各自的流级阈值：无边界流上 max 的 99 分位
    calibration = np.random.default_rng(SEED)
    raw_maxima, white_maxima = [], []
    for stream in _null_streams(train_x, train_task, calibration):
        _, raw, whitened = _statistics(stream, factor)
        raw_maxima.append(float(raw.max()))
        white_maxima.append(float(whitened.max()))
    tau = (
        float(np.percentile(raw_maxima, NULL_QUANTILE)),
        float(np.percentile(white_maxima, NULL_QUANTILE)),
    )

    # E-0：同一构造的另 50 条无边界流，数每条流报几次
    replay = np.random.default_rng(SEED + 1)
    null_alarms = []
    for stream in _null_streams(train_x, train_task, replay):
        positions, raw, whitened = _statistics(stream, factor)
        null_alarms.append(len(_peaks(positions, _score(raw, whitened, tau))))

    truth = [BLOCK * (task + 1) for task in range(TASKS - 1)]

    def evaluate(seed: int) -> dict:
        stream, owner = _test_stream(test_x, test_task, seed)
        positions, raw, whitened = _statistics(stream, factor)
        peaks = _peaks(positions, _score(raw, whitened, tau))
        return {
            "seed": seed,
            "peaks": peaks,
            "detected": len(peaks),
            "f1": round(_f1(peaks, truth, TOLERANCE), 4),
            "routing": round(_routing(peaks, owner), 4),
        }

    main_run = evaluate(SEED)
    extras = [evaluate(seed) for seed in EXTRA_SEEDS]

    control_stream, _ = _test_stream(test_x, test_task, SEED)
    control = np.random.default_rng(SEED + 99).permutation(control_stream.shape[0])
    positions, raw, whitened = _statistics(control_stream[control], factor)
    control_f1 = _f1(_peaks(positions, _score(raw, whitened, tau)), truth, TOLERANCE)

    criteria = {
        "e0_stream_fp_controlled": float(np.mean(null_alarms)) <= MAX_NULL_ALARMS,
        "e1_segment_count": main_run["detected"] == TASKS - 1,
        "e2_boundary_f1": main_run["f1"] >= MIN_F1,
        "e3_routing": main_run["routing"] >= MIN_ROUTING,
        "e4_control_quiet": control_f1 < CONTROL_MAX_F1,
        "e5_holds_across_seeds": all(
            row["detected"] == TASKS - 1 and row["f1"] >= MIN_F1 for row in extras
        ),
        "null_alarms_mean": round(float(np.mean(null_alarms)), 3),
        "control_f1": round(control_f1, 4),
    }
    out = {
        "kind": "CL evaluation-stream task-boundary detection v3 (two statistics, stream-level thresholds)",
        "seed": SEED,
        "extra_seeds": list(EXTRA_SEEDS),
        "window": WINDOW,
        "step": STEP,
        "tolerance": TOLERANCE,
        "tau_raw": round(tau[0], 4),
        "tau_whitened": round(tau[1], 4),
        "null_alarms": null_alarms,
        "main": main_run,
        "extras": extras,
        "truth": truth,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[阈值] raw {tau[0]:.4f} / whitened {tau[1]:.4f}")
    print(f"[E-0] 无边界流平均检出 {np.mean(null_alarms):.2f} 次")
    print(f"[主] {main_run['detected']} 个：{main_run['peaks']}（真值 {truth}）F1 {main_run['f1']}")
    for row in extras:
        print(f"[seed {row['seed']}] {row['detected']} 个 F1 {row['f1']} 路由 {row['routing']}")
    print(f"[E-4] 随机流对照 F1 {control_f1:.4f}")
    print(
        " | ".join(
            f"[{name[:2].upper()}] {'过' if value else '不过'}"
            for name, value in criteria.items()
            if isinstance(value, bool)
        )
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
