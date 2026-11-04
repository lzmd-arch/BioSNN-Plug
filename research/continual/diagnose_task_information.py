"""任务信息天花板诊断（判据见 `sweep_results/continual_v3_task_information_prereg.md`）。

**诊断，不是方法。** 五个探针只回答一件事：冻结 ResNet18 特征里还剩下多少任务身份。
C/D/E 用全部任务联合训练，在 CIL 里做不到——它们是**标尺**，不是可部署的路由器。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.continual.data import load_cifar100
from research.continual.frontends import BACKBONES, image_features

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "continual_v3_task_information"
DATA_ROOT = REPO_ROOT / "data" / "cifar100"
CACHE = REPO_ROOT / "data" / "_features"

SEED = 0
BACKBONE = "resnet18"
SHRINKAGE = 0.1
EPS = 1e-6
KNN_K = 5
TASKS = 10
MLP_HIDDEN = 256
MLP_EPOCHS = 30
LOGREG_EPOCHS = 300
BATCH = 512

#: 判据（改这里必须改预注册）。
B_REPRODUCE = (0.528, 0.03)
CEILING_PASS = 0.80
CEILING_FLOOR = 0.65


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True).clip(min=1e-6)
    return (train - mean) / scale, (test - mean) / scale


def probe_prototype(train_x, train_task, test_x, test_task) -> float:
    """A：最近任务原型（欧氏）。"""
    prototypes = np.stack([train_x[train_task == task].mean(axis=0) for task in range(TASKS)])
    distances = (
        (test_x**2).sum(axis=1)[:, None]
        + (prototypes**2).sum(axis=1)[None, :]
        - 2.0 * test_x @ prototypes.T
    )
    return float((distances.argmin(axis=1) == test_task).mean())


def probe_full_covariance(train_x, train_y, train_task, test_x, test_task) -> float:
    """B：逐任务全协方差 Mahalanobis（shrinkage 0.1），与旧臂同一口径。"""
    means, inverses = [], []
    for task in range(TASKS):
        classes = np.unique(train_y[train_task == task])
        class_means = [train_x[train_y == label].mean(axis=0) for label in classes]
        centered = [train_x[train_y == label] - class_means[i] for i, label in enumerate(classes)]
        count = sum(row.shape[0] for row in centered)
        scatter = sum(row.T @ row for row in centered) / max(count - len(classes), 1)
        diagonal = np.diag(np.diag(scatter))
        covariance = (1.0 - SHRINKAGE) * scatter + SHRINKAGE * diagonal
        covariance = covariance + EPS * np.eye(covariance.shape[0])
        means.append(np.mean(class_means, axis=0))
        inverses.append(np.linalg.inv(covariance))
    distances = np.empty((test_x.shape[0], TASKS))
    for index, (mean, inverse) in enumerate(zip(means, inverses, strict=True)):
        diff = test_x - mean
        distances[:, index] = np.einsum("ni,ni->n", diff @ inverse, diff)
    return float((distances.argmin(axis=1) == test_task).mean())


def probe_knn(train_x, train_task, test_x, test_task) -> float:
    """D：k=5 余弦最近邻。"""
    train_norm = train_x / np.linalg.norm(train_x, axis=1, keepdims=True).clip(min=1e-6)
    test_norm = test_x / np.linalg.norm(test_x, axis=1, keepdims=True).clip(min=1e-6)
    correct = 0
    for start in range(0, test_norm.shape[0], 512):
        block = test_norm[start : start + 512] @ train_norm.T
        nearest = np.argpartition(-block, KNN_K, axis=1)[:, :KNN_K]
        votes = train_task[nearest]
        predicted = np.array([np.bincount(row, minlength=TASKS).argmax() for row in votes])
        correct += int((predicted == test_task[start : start + 512]).sum())
    return correct / test_task.shape[0]


def _train_head(model, train_x, train_task, test_x, test_task, *, epochs, lr, device) -> float:
    torch.manual_seed(SEED)
    x = torch.tensor(train_x, dtype=torch.float32, device=device)
    y = torch.tensor(train_task, dtype=torch.long, device=device)
    xt = torch.tensor(test_x, dtype=torch.float32, device=device)
    yt = torch.tensor(test_task, dtype=torch.long, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], BATCH):
            index = order[start : start + BATCH]
            loss = nn.functional.cross_entropy(model(x[index]), y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return float((model(xt).argmax(dim=-1) == yt).float().mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="任务信息天花板诊断")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="task_information")
    parser.add_argument("--backbone", default=BACKBONE, choices=sorted(BACKBONES))
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train, test = load_cifar100(DATA_ROOT)
    CACHE.mkdir(parents=True, exist_ok=True)
    train_x = image_features(
        train.x, backbone=args.backbone, cache=CACHE / f"{args.backbone}_train.npz", device=device
    ).astype(np.float64)
    test_x = image_features(
        test.x, backbone=args.backbone, cache=CACHE / f"{args.backbone}_test.npz", device=device
    ).astype(np.float64)
    train_y = np.asarray(train.y, dtype=np.int64)
    test_y = np.asarray(test.y, dtype=np.int64)
    train_task = train_y // 10
    test_task = test_y // 10
    train_std, test_std = _standardize(train_x, test_x)

    results = {
        "a_prototype": probe_prototype(train_x, train_task, test_x, test_task),
        "b_full_covariance": probe_full_covariance(train_x, train_y, train_task, test_x, test_task),
        "c_logistic": _train_head(
            nn.Linear(train_std.shape[1], TASKS),
            train_std,
            train_task,
            test_std,
            test_task,
            epochs=LOGREG_EPOCHS,
            lr=2e-3,
            device=device,
        ),
        "d_knn": probe_knn(train_x, train_task, test_x, test_task),
        "f_fine_class": _train_head(
            nn.Sequential(
                nn.Linear(train_std.shape[1], MLP_HIDDEN), nn.ReLU(), nn.Linear(MLP_HIDDEN, 100)
            ),
            train_std,
            train_y,
            test_std,
            test_y,
            epochs=MLP_EPOCHS,
            lr=1e-3,
            device=device,
        ),
        "e_mlp": _train_head(
            nn.Sequential(
                nn.Linear(train_std.shape[1], MLP_HIDDEN), nn.ReLU(), nn.Linear(MLP_HIDDEN, TASKS)
            ),
            train_std,
            train_task,
            test_std,
            test_task,
            epochs=MLP_EPOCHS,
            lr=1e-3,
            device=device,
        ),
    }
    best = max(results["c_logistic"], results["d_knn"], results["e_mlp"])
    criteria = {
        "g0_reproduces": (
            args.backbone != "resnet18"
            or abs(results["b_full_covariance"] - B_REPRODUCE[0]) <= B_REPRODUCE[1]
        ),
        "g1_information_present": best >= CEILING_PASS,
        "g2_information_absent": best < CEILING_FLOOR,
        "g3_middle": CEILING_FLOOR <= best < CEILING_PASS,
        "best_nonlocal": round(best, 4),
    }
    out = {
        "kind": "CL task-information ceiling diagnosis",
        "seed": SEED,
        "backbone": args.backbone,
        "n_train": int(train_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "probes": {key: round(value, 4) for key, value in results.items()},
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[探针] {out['probes']}")
    print(
        f"[G-0] B={results['b_full_covariance']:.4f} vs {B_REPRODUCE[0]}+-{B_REPRODUCE[1]} -> "
        f"{'过' if criteria['g0_reproduces'] else '不过'} | "
        f"[G-1] 非局部最好 {best:.4f} >= {CEILING_PASS} -> "
        f"{'成立' if criteria['g1_information_present'] else '不成立'} | "
        f"[G-2] < {CEILING_FLOOR} -> {'成立' if criteria['g2_information_absent'] else '不成立'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
