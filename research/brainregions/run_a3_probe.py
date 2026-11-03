"""A3 的事后探针：15.8pp 是"表示变了"还是"解码器变强了"（**非预注册**）。

预注册 `prereg-a3-multitap.md` 写死了"若 A3-1 大幅超过 9.7pp，要回头查"。本脚本就是那一步。
**它是事后诊断，不进判据表**：结论只用来解释 A3，不能反过来当成新判据的证据。

做法：用 `k=28` 把网络重训一遍（同种子、同预算），然后在这个网络上做 A2 的两个岭回归探针——

- Q2 最后一步特征 → 与 A2 的 P2（k=1 网络上 0.772）比，看**单帧表示**有没有变好；
- Q3 全 28 步特征 → 与 A2 的 P3（k=1 网络上 0.869）比，看**时序表示**有没有被训练出来。

Q1 顺带做一次**可复现性检查**：同种子重训应当逐位复现 0.917。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.brainregions.run_a1 import BATCH, EPOCHS
from research.brainregions.run_a2 import LAMBDA, N_FIT, N_TEST, features, ridge_accuracy
from research.brainregions.run_a3 import SEED, TAPS, build, evaluate
from research.eprop import load_smnist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a3"

N_FIT_BIG = 4000


def train_model(taps: int, seed: int, device: str):
    torch.manual_seed(seed)
    train = load_smnist("data", split="train", seed=seed)
    test = load_smnist("data", split="test", seed=seed)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    model = build(int(inputs.shape[2]), n_out, taps).to(device)
    generator = torch.Generator().manual_seed(seed + 991)
    for _ in range(EPOCHS):
        permutation = torch.randperm(inputs.shape[1], generator=generator).to(device)
        for start in range(0, inputs.shape[1], BATCH):
            index = permutation[start : start + BATCH]
            model.update(inputs[:, index], targets[index])
    return model, train, test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A3 事后探针（非预注册）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a3_probe")
    parser.add_argument("--taps", type=int, default=max(TAPS))
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, train, test = train_model(args.taps, SEED, device)
    n_classes = int(model.w_out.shape[1])
    online = evaluate(model, test, device)["accuracy"]

    last_train, full_train, y_train = features(model, train, 0, N_FIT, device)
    last_test, full_test, y_test = features(model, test, 0, N_TEST, device)
    _, full_train_big, y_train_big = features(model, train, 0, N_FIT_BIG, device)

    q2 = ridge_accuracy(last_train, y_train, last_test, y_test, n_classes)
    q3 = ridge_accuracy(full_train, y_train, full_test, y_test, n_classes)
    q3_big = ridge_accuracy(full_train_big, y_train_big, full_test, y_test, n_classes)

    out = {
        "kind": "line A v3 post-hoc probe (NOT preregistered)",
        "taps": args.taps,
        "seed": SEED,
        "n_fit": N_FIT,
        "n_fit_big": N_FIT_BIG,
        "lambda": LAMBDA,
        "q1_online": round(online, 4),
        "q2_ridge_last_step": round(q2, 4),
        "q3_ridge_all_steps": round(q3, 4),
        "q3_ridge_all_steps_big_fit": round(q3_big, 4),
        "reference_a2_on_taps1_net": {"p1": 0.759, "p2": 0.772, "p3": 0.869},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"[Q1] 在线 {online:.4f}（应为 0.917）| [Q2] 单帧岭回归 {q2:.4f}（k=1 网络 0.772）| "
        f"[Q3] 全时序岭回归 {q3:.4f}（k=1 网络 0.869）| [Q3+] 4,000 样本拟合 {q3_big:.4f}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
