"""A2 诊断：8k 管线的天花板在读出、表示还是学习规则（判据见 prereg-a2-diagnosis.md）。

三个探针，同一个训好的网络、同样的数据：

- P1 在线读出（现状）；
- P2 离线岭回归，最后一步的滤波发放；
- P3 离线岭回归，全部 28 个时间步拼起来。

P2 用对偶形式（n ≪ d），P3 也是——特征维度 28×8192 太大，不能直接解正规方程。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.brainregions.run_a1 import (
    BATCH,
    EPOCHS,
    FAN_IN,
    N_REC,
    uniform_regions,
)
from research.eprop import exp_convolve, load_smnist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a2"

SEED = 0
N_FIT = 1000
N_TEST = 1000
LAMBDA = 1e-3
D_PASS = 0.05
P1_EXPECT = 0.762
P1_TOLERANCE = 0.01


def train_model(device: str):
    torch.manual_seed(SEED)
    train = load_smnist("data", split="train", seed=SEED)
    test = load_smnist("data", split="test", seed=SEED)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    from research.brainregions.model import RegionEProp

    model = RegionEProp(
        int(inputs.shape[2]),
        N_REC,
        n_out,
        fan_in=FAN_IN,
        regions=uniform_regions(),
        intra_fraction=0.0,
        generator=torch.Generator().manual_seed(SEED),
    ).to(device)
    generator = torch.Generator().manual_seed(SEED + 991)
    for _ in range(EPOCHS):
        permutation = torch.randperm(inputs.shape[1], generator=generator).to(device)
        for start in range(0, inputs.shape[1], BATCH):
            index = permutation[start : start + BATCH]
            model.update(inputs[:, index], targets[index])
    return model, train, test


def features(model, dataset, offset: int, count: int, device: str):
    """返回 (最后一步特征, 全时序特征, 标签)，两者都 (n, ·)。"""
    last, full, labels = [], [], []
    for start in range(offset, offset + count, 100):
        stop = min(start + 100, offset + count)
        inputs, batch_labels = dataset.batch(torch.arange(start, stop))
        with torch.no_grad():
            spikes, _ = model.simulate(inputs.to(device))
            filtered = exp_convolve(spikes, model.decay_out)
        last.append(filtered[-1].cpu())
        full.append(filtered.permute(1, 0, 2).reshape(stop - start, -1).cpu())
        labels.append(batch_labels)
    return torch.cat(last), torch.cat(full), torch.cat(labels)


def ridge_accuracy(x_train, y_train, x_test, y_test, n_classes: int) -> float:
    """对偶形式岭回归；λ 按核的对角均值标定，不扫参。"""
    gram = x_train @ x_train.T
    scale = float(gram.diagonal().mean())
    target = torch.zeros(len(y_train), n_classes)
    target[torch.arange(len(y_train)), y_train] = 1.0
    alpha = torch.linalg.solve(gram + LAMBDA * scale * torch.eye(len(y_train)), target)
    predictions = (x_test @ x_train.T @ alpha).argmax(dim=-1)
    return float((predictions == y_test).float().mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A2 诊断：读出 / 表示 / 学习规则")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a2")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, train, test = train_model(device)
    n_classes = int(model.w_out.shape[1])
    last_train, full_train, y_train = features(model, train, 0, N_FIT, device)
    last_test, full_test, y_test = features(model, test, 0, N_TEST, device)

    p1 = float(((last_test.to(device) @ model.w_out).argmax(dim=-1).cpu() == y_test).float().mean())
    p2 = ridge_accuracy(last_train, y_train, last_test, y_test, n_classes)
    p3 = ridge_accuracy(full_train, y_train, full_test, y_test, n_classes)

    criteria = {
        "d0_pipeline_reproduces": abs(p1 - P1_EXPECT) <= P1_TOLERANCE,
        "d1_learning_rule_is_bottleneck": p2 - p1 >= D_PASS,
        "d2_last_step_readout_is_bottleneck": p3 - p2 >= D_PASS,
    }
    criteria["d3_representation_is_bottleneck"] = not (
        criteria["d1_learning_rule_is_bottleneck"] or criteria["d2_last_step_readout_is_bottleneck"]
    )
    out = {
        "kind": "line A v2: readout / representation / learning-rule diagnosis",
        "seed": SEED,
        "n_rec": N_REC,
        "n_fit": N_FIT,
        "n_test": N_TEST,
        "lambda": LAMBDA,
        "p1_online": round(p1, 4),
        "p2_ridge_last_step": round(p2, 4),
        "p3_ridge_all_steps": round(p3, 4),
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"P1 在线读出 {p1:.4f} | P2 岭回归(最后一步) {p2:.4f} | P3 岭回归(全时序) {p3:.4f}")
    print(
        f"[D-0] 前置 {'过' if criteria['d0_pipeline_reproduces'] else '不过'} | "
        f"[D-1] P2−P1 {p2 - p1:+.3f} | [D-2] P3−P2 {p3 - p2:+.3f} → "
        + (
            "学习规则是瓶颈"
            if criteria["d1_learning_rule_is_bottleneck"]
            else "只取最后一步是瓶颈"
            if criteria["d2_last_step_readout_is_bottleneck"]
            else "表示本身是瓶颈"
        )
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
