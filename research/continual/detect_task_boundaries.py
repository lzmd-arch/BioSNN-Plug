"""评测流上的任务边界检测（判据见 `sweep_results/continual_v3_boundary_detection_prereg.md`）。

**不是逐样本任务推断。** 这里利用的是评测流按任务分块的结构：问"任务什么时候换的"，
而不是"这一张图属于哪个任务"。单张图、无上下文的场景下它给不出任务 id。

做法：滑动窗口的均值突变检测，阈值由**训练集**上同任务内部的窗口距离零分布定，
测试集标签一次都不用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.continual.data import load_cifar100

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "continual_v3_boundary_detection"
DATA_ROOT = REPO_ROOT / "data" / "cifar100"
CACHE = REPO_ROOT / "data" / "_features"

SEED = 0
TASKS = 10
WINDOW = 64
STEP = 10
PEAK_RADIUS = 50
TOLERANCE = 100
QUANTILE = 99.5
NULL_DRAWS = 2000
#: 判据（改这里必须改预注册）。
MIN_F1 = 0.80
MIN_ROUTING = 0.80
CONTROL_MAX_F1 = 0.30


def _stream_stats(stream: np.ndarray, statistic_positions: np.ndarray) -> np.ndarray:
    """每个候选位置的前后窗口均值距离。"""
    values = np.empty(len(statistic_positions))
    for index, position in enumerate(statistic_positions):
        before = stream[position - WINDOW : position].mean(axis=0)
        after = stream[position : position + WINDOW].mean(axis=0)
        values[index] = np.linalg.norm(before - after)
    return values


def _null_threshold(train_x: np.ndarray, train_task: np.ndarray) -> tuple[float, np.ndarray]:
    """零分布：同一任务内部随机取两段不相交窗口。只用训练集。"""
    rng = np.random.default_rng(SEED)
    draws = []
    for task in range(TASKS):
        pool = train_x[train_task == task]
        for _ in range(NULL_DRAWS // TASKS):
            start = rng.integers(0, pool.shape[0] - 2 * WINDOW)
            first = pool[start : start + WINDOW].mean(axis=0)
            second = pool[start + WINDOW : start + 2 * WINDOW].mean(axis=0)
            draws.append(np.linalg.norm(first - second))
    draws = np.asarray(draws)
    return float(np.percentile(draws, QUANTILE)), draws


def _detect(stream: np.ndarray, threshold: float) -> list[int]:
    positions = np.arange(WINDOW, stream.shape[0] - WINDOW, STEP)
    values = _stream_stats(stream, positions)
    peaks: list[int] = []
    for index, value in enumerate(values):
        if value <= threshold:
            continue
        low = max(0, index - PEAK_RADIUS // STEP)
        high = min(len(values), index + PEAK_RADIUS // STEP + 1)
        if value >= values[low:high].max():
            peak = int(positions[index])
            if not peaks or peak - peaks[-1] > PEAK_RADIUS:
                peaks.append(peak)
    return peaks


def _f1(detected: list[int], truth: list[int], tolerance: int) -> tuple[float, int, int]:
    matched_truth: set[int] = set()
    matched_detected = 0
    for peak in detected:
        candidates = [t for t in truth if abs(t - peak) <= tolerance and t not in matched_truth]
        if candidates:
            matched_truth.add(min(candidates, key=lambda t: abs(t - peak)))
            matched_detected += 1
    precision = matched_detected / len(detected) if detected else 0.0
    recall = len(matched_truth) / len(truth) if truth else 0.0
    if precision + recall == 0.0:
        return 0.0, matched_detected, len(detected)
    return 2 * precision * recall / (precision + recall), matched_detected, len(detected)


def _routing_accuracy(peaks: list[int], stream_task: np.ndarray) -> float:
    """按到达顺序把第 k 段判给第 k 个任务——不用任何测试标签。"""
    edges = [0, *peaks, stream_task.shape[0]]
    correct = 0
    for index in range(len(edges) - 1):
        segment = stream_task[edges[index] : edges[index + 1]]
        if index < TASKS:
            correct += int((segment == index).sum())
    return correct / stream_task.shape[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测流上的任务边界检测")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="boundary_detection")
    args = parser.parse_args(argv)

    train, test = load_cifar100(DATA_ROOT)
    train_x = np.load(CACHE / "resnet18_train.npz")["features"].astype(np.float64)
    test_x = np.load(CACHE / "resnet18_test.npz")["features"].astype(np.float64)
    train_task = np.asarray(train.y, dtype=np.int64) // 10
    test_task = np.asarray(test.y, dtype=np.int64) // 10

    threshold, null_draws = _null_threshold(train_x, train_task)

    rng = np.random.default_rng(SEED)
    blocks = []
    for task in range(TASKS):
        pool = test_x[test_task == task]
        blocks.append(pool[rng.permutation(pool.shape[0])])
    stream = np.concatenate(blocks, axis=0)
    stream_task = np.concatenate([np.full(len(block), task) for task, block in enumerate(blocks)])
    truth = [1000 * (task + 1) for task in range(TASKS - 1)]

    peaks = _detect(stream, threshold)
    f1, matched, detected = _f1(peaks, truth, TOLERANCE)
    routing = _routing_accuracy(peaks, stream_task)

    control = test_x[rng.permutation(test_x.shape[0])]
    control_peaks = _detect(control, threshold)
    control_f1, _, _ = _f1(control_peaks, truth, TOLERANCE)

    criteria = {
        "d0_segment_count": len(peaks) == TASKS - 1,
        "d1_boundary_f1": f1 >= MIN_F1,
        "d2_routing": routing >= MIN_ROUTING,
        "d3_control_quiet": control_f1 < CONTROL_MAX_F1,
        "f1": round(f1, 4),
        "routing": round(routing, 4),
        "control_f1": round(control_f1, 4),
        "detected": len(peaks),
    }
    out = {
        "kind": "CL evaluation-stream task-boundary detection",
        "seed": SEED,
        "window": WINDOW,
        "step": STEP,
        "tolerance": TOLERANCE,
        "quantile": QUANTILE,
        "threshold": round(threshold, 4),
        "null_median": round(float(np.median(null_draws)), 4),
        "peaks": peaks,
        "truth": truth,
        "matched": matched,
        "detected": detected,
        "control_peaks": control_peaks,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[阈值] {threshold:.4f}（零分布中位 {np.median(null_draws):.4f}）")
    print(f"[检出] {len(peaks)} 个边界：{peaks}（真值 {truth}）")
    print(f"[对照] 随机流检出 {len(control_peaks)} 个，F1 {control_f1:.4f}")
    print(
        f"[D-0] 段数 {'过' if criteria['d0_segment_count'] else '不过'} | "
        f"[D-1] F1 {f1:.4f} >= {MIN_F1} -> {'过' if criteria['d1_boundary_f1'] else '不过'} | "
        f"[D-2] 路由 {routing:.4f} >= {MIN_ROUTING} -> {'过' if criteria['d2_routing'] else '不过'} | "
        f"[D-3] 对照 {control_f1:.4f} < {CONTROL_MAX_F1} -> "
        f"{'过' if criteria['d3_control_quiet'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
