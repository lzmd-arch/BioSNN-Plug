"""评测流上的任务边界检测 v2（判据见 `sweep_results/continual_v3_boundary_detection_v2_prereg.md`）。

与 v1 的差别只有两处（见预注册）：

1. 统计量从"均值差的欧氏长度"换成"在训练集同任务内散布下白化后的均值偏移"；
2. 阈值从"单个位置的 99.5 分位"换成"**每条流最大统计量**的 99 分位"。

其余（W=64、步长 10、±50 峰半径、±100 容差、流的构造）与 v1 完全相同。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.continual.data import load_cifar100

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "continual_v3_boundary_detection_v2"
DATA_ROOT = REPO_ROOT / "data" / "cifar100"
CACHE = REPO_ROOT / "data" / "_features"

SEED = 0
TASKS = 10
BLOCK = 1000
WINDOW = 64
STEP = 10
PEAK_RADIUS = 50
TOLERANCE = 100
SHRINKAGE = 0.1
EPS = 1e-6
NULL_STREAMS = 50
NULL_QUANTILE = 99.0

#: 判据（改这里必须改预注册）。
MAX_NULL_ALARMS = 1.0
MIN_F1 = 0.80
MIN_ROUTING = 0.80
CONTROL_MAX_F1 = 0.30


def _whitener(train_x: np.ndarray, train_task: np.ndarray) -> torch.Tensor:
    """训练集上同任务内散布的合并估计 → 下三角因子（Σ = L Lᵀ）。"""
    scatter = np.zeros((train_x.shape[1], train_x.shape[1]))
    count = 0
    for task in range(TASKS):
        block = train_x[train_task == task]
        centered = block - block.mean(axis=0, keepdims=True)
        scatter += centered.T @ centered
        count += block.shape[0] - 1
    scatter /= count
    diagonal = np.diag(np.diag(scatter))
    covariance = (1.0 - SHRINKAGE) * scatter + SHRINKAGE * diagonal
    covariance = covariance + EPS * np.eye(covariance.shape[0])
    return torch.linalg.cholesky(torch.tensor(covariance, dtype=torch.float64))


def _statistics(stream: np.ndarray, factor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """返回 (位置, 白化统计量)。窗口均值用累积和算，避免逐位置切片。"""
    positions = np.arange(WINDOW, stream.shape[0] - WINDOW, STEP)
    cumulative = np.concatenate([np.zeros((1, stream.shape[1])), np.cumsum(stream, axis=0)])
    after = (cumulative[positions + WINDOW] - cumulative[positions]) / WINDOW
    before = (cumulative[positions] - cumulative[positions - WINDOW]) / WINDOW
    difference = after - before
    whitened = torch.linalg.solve_triangular(
        factor, torch.tensor(difference, dtype=torch.float64).T, upper=False
    )
    return positions, whitened.norm(dim=0).numpy()


def _peaks(positions: np.ndarray, values: np.ndarray, threshold: float) -> list[int]:
    peaks: list[int] = []
    span = PEAK_RADIUS // STEP
    for index, value in enumerate(values):
        if value <= threshold:
            continue
        low, high = max(0, index - span), min(len(values), index + span + 1)
        if value >= values[low:high].max():
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


def _routing(peaks: list[int], stream_task: np.ndarray) -> float:
    edges = [0, *peaks, stream_task.shape[0]]
    correct = 0
    for index in range(len(edges) - 1):
        if index >= TASKS:
            break
        segment = stream_task[edges[index] : edges[index + 1]]
        correct += int((segment == index).sum())
    return correct / stream_task.shape[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="任务边界检测 v2")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="boundary_v2")
    args = parser.parse_args(argv)

    train, test = load_cifar100(DATA_ROOT)
    train_x = np.load(CACHE / "resnet18_train.npz")["features"].astype(np.float64)
    test_x = np.load(CACHE / "resnet18_test.npz")["features"].astype(np.float64)
    train_task = np.asarray(train.y, dtype=np.int64) // 10
    test_task = np.asarray(test.y, dtype=np.int64) // 10
    factor = _whitener(train_x, train_task)

    # 阈值：每条无边界流的最大统计量的 99 分位（训练集，单任务，10 块独立重抽样）
    rng = np.random.default_rng(SEED)
    null_maxima = []
    null_alarms = []
    for index in range(NULL_STREAMS):
        pool = train_x[train_task == index % TASKS]
        chunks = [pool[rng.integers(0, pool.shape[0], size=BLOCK)] for _ in range(TASKS)]
        stream = np.concatenate(chunks)
        _, values = _statistics(stream, factor)
        null_maxima.append(float(values.max()))
    threshold = float(np.percentile(null_maxima, NULL_QUANTILE))

    # 阈值定好之后，重跑同一批无边界流，数一下每条流会报几次（E-0）
    replay = np.random.default_rng(SEED + 1)
    null_alarms = []
    for index in range(NULL_STREAMS):
        pool = train_x[train_task == index % TASKS]
        chunks = [pool[replay.integers(0, pool.shape[0], size=BLOCK)] for _ in range(TASKS)]
        positions, values = _statistics(np.concatenate(chunks), factor)
        null_alarms.append(len(_peaks(positions, values, threshold)))

    # 真实流：与 v1 完全相同的构造（seed 0）
    stream_rng = np.random.default_rng(SEED)
    blocks, owners = [], []
    for task in range(TASKS):
        pool = test_x[test_task == task]
        blocks.append(pool[stream_rng.permutation(pool.shape[0])])
        owners.append(np.full(pool.shape[0], task))
    stream = np.concatenate(blocks)
    owner = np.concatenate(owners)
    truth = [BLOCK * (task + 1) for task in range(TASKS - 1)]

    positions, values = _statistics(stream, factor)
    peaks = _peaks(positions, values, threshold)
    f1 = _f1(peaks, truth, TOLERANCE)
    routing = _routing(peaks, owner)

    control_stream = test_x[stream_rng.permutation(test_x.shape[0])]
    control_positions, control_values = _statistics(control_stream, factor)
    control_f1 = _f1(_peaks(control_positions, control_values, threshold), truth, TOLERANCE)

    true_values = []
    for boundary in truth:
        index = int(np.argmin(np.abs(positions - boundary)))
        true_values.append(round(float(values[index]), 4))
    criteria = {
        "e0_stream_fp_controlled": float(np.mean(null_alarms)) <= MAX_NULL_ALARMS,
        "e1_segment_count": len(peaks) == TASKS - 1,
        "e2_boundary_f1": f1 >= MIN_F1,
        "e3_routing": routing >= MIN_ROUTING,
        "e4_control_quiet": control_f1 < CONTROL_MAX_F1,
        "f1": round(f1, 4),
        "routing": round(routing, 4),
        "control_f1": round(control_f1, 4),
        "detected": len(peaks),
        "null_alarms_mean": round(float(np.mean(null_alarms)), 2),
    }
    out = {
        "kind": "CL evaluation-stream task-boundary detection v2 (whitened, stream-level threshold)",
        "seed": SEED,
        "window": WINDOW,
        "step": STEP,
        "tolerance": TOLERANCE,
        "shrinkage": SHRINKAGE,
        "null_streams": NULL_STREAMS,
        "null_quantile": NULL_QUANTILE,
        "threshold": round(threshold, 4),
        "null_maxima_range": [round(min(null_maxima), 4), round(max(null_maxima), 4)],
        "null_alarms": null_alarms,
        "peaks": peaks,
        "truth": truth,
        "true_boundary_statistics": true_values,
        "control_peaks": len(_peaks(control_positions, control_values, threshold)),
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[阈值] {threshold:.4f}（无边界流最大值区间 {out['null_maxima_range']}）")
    print(f"[E-0] 无边界流平均检出 {np.mean(null_alarms):.2f} 次")
    print(f"[检出] {len(peaks)} 个：{peaks}（真值 {truth}）")
    print(f"[真边界统计量] {true_values}")
    print(
        f"[E-1] {'过' if criteria['e1_segment_count'] else '不过'} | "
        f"[E-2] F1 {f1:.4f} -> {'过' if criteria['e2_boundary_f1'] else '不过'} | "
        f"[E-3] 路由 {routing:.4f} -> {'过' if criteria['e3_routing'] else '不过'} | "
        f"[E-4] 对照 {control_f1:.4f} -> {'过' if criteria['e4_control_quiet'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
