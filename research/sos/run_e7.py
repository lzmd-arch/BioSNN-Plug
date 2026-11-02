"""E7：序列模型 + 更大语料（判据见 prereg-e7.md）。

表格 softmax 换成单层 GRU：它能维持隐藏状态，所以"我在贴哪面墙"这类记忆不再是外挂的
特征，而是模型自己学的。语料从 100 张迷宫扩到 2000 张。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from random import Random

import numpy as np
import torch
from torch import nn

from .grid import ACTIONS, DEFAULT_SLIP, make_maze, step
from .run_e2_v2 import mission
from .run_e3 import dfs_trace, opt_trace

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "sos_e7"

SIZES = (9, 11, 13)
TRAIN_MAZES = 2000
CONFIRM_MAZES = 200
EPOCHS = 100
LEARNING_RATE = 3e-3
HIDDEN = 128
BATCH = 128
SEEDS = (0, 1, 2, 3, 4)
NONE = 4

OBS_DIM = 2 + 4 + 5 + 5

E7_TRAIN_MIN = 0.50
E7_H1_MIN = 0.05
E7_STEP_RATIO = 1.50


class SequencePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(OBS_DIM, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, len(ACTIONS))

    def forward(self, obs, hidden=None):
        out, hidden = self.gru(obs, hidden)
        return self.head(out), hidden


def obs_vector(grid, cell, heading: int, prev_action: int) -> list[float]:
    size = grid.size
    vector = [
        (grid.goal[0] - cell[0]) / (size - 1),
        (grid.goal[1] - cell[1]) / (size - 1),
    ]
    vector += [1.0 if grid.blocked((cell[0] + dx, cell[1] + dy)) else 0.0 for dx, dy in ACTIONS]
    heading_onehot = [0.0] * 5
    heading_onehot[heading] = 1.0
    prev_onehot = [0.0] * 5
    prev_onehot[prev_action] = 1.0
    return vector + heading_onehot + prev_onehot


def sequence(grid, trace, chosen) -> tuple[np.ndarray, np.ndarray]:
    rows, actions = [], []
    for position in chosen:
        cell, action = trace[position]
        heading = trace[position - 1][1] if position > 0 else NONE
        prev = trace[position - 1][1] if position > 0 else NONE
        rows.append(obs_vector(grid, cell, heading, prev))
        actions.append(action)
    return np.asarray(rows, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def build_corpus(arm: str, grids, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = Random(seed)
    corpus = []
    for index, grid in enumerate(grids):
        base = opt_trace(grid)
        if not base:
            continue
        trace = base if arm == "opt" else dfs_trace(grid, seed * 100_003 + index)
        if not trace:
            continue
        chosen = (
            list(range(len(trace)))
            if len(trace) <= len(base)
            else rng.sample(range(len(trace)), len(base))
        )
        corpus.append(sequence(grid, trace, chosen))
    return corpus


def train(arm: str, grids, seed: int, device) -> SequencePolicy:
    torch.manual_seed(seed)
    model = SequencePolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    corpus = build_corpus(arm, grids, seed)
    rng = Random(seed)
    for _ in range(EPOCHS):
        order = list(range(len(corpus)))
        rng.shuffle(order)
        for start in range(0, len(order), BATCH):
            batch = [corpus[index] for index in order[start : start + BATCH]]
            maxlen = max(len(obs) for obs, _ in batch)
            obs = np.zeros((len(batch), maxlen, OBS_DIM), dtype=np.float32)
            targets = np.zeros((len(batch), maxlen), dtype=np.int64)
            mask = np.zeros((len(batch), maxlen), dtype=np.float32)
            for row, (sequence_obs, sequence_actions) in enumerate(batch):
                length = len(sequence_obs)
                obs[row, :length] = sequence_obs
                targets[row, :length] = sequence_actions
                mask[row, :length] = 1.0
            obs_t = torch.from_numpy(obs).to(device)
            target_t = torch.from_numpy(targets).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            logits, _ = model(obs_t)
            losses = loss_fn(logits.reshape(-1, len(ACTIONS)), target_t.reshape(-1))
            loss = (losses * mask_t.reshape(-1)).sum() / mask_t.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model: SequencePolicy, grids, seed: int) -> dict:
    rng = Random(seed + 90_000)
    wins, steps = 0, []
    for grid in grids:
        cell, heading, previous = grid.start, NONE, NONE
        hidden = None
        for index in range(mission(grid)):
            obs = torch.tensor([[obs_vector(grid, cell, heading, previous)]], dtype=torch.float32)
            logits, hidden = model(obs, hidden)
            action = int(torch.argmax(logits[0, -1]))
            cell = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
            heading = previous = action
            if cell == grid.goal:
                wins += 1
                steps.append(index + 1)
                break
    return {"success": wins / len(grids), "steps": statistics.fmean(steps) if steps else 0.0}


def mazes(offset: int, count: int):
    return [make_maze(offset + index, size=SIZES[index % len(SIZES)]) for index in range(count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E7（序列模型 + 大语料）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e7")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_grids = mazes(0, TRAIN_MAZES)
    confirm_grids = mazes(1000, CONFIRM_MAZES)

    results: dict[str, dict] = {}
    for arm in ("opt", "search"):
        train_rows, confirm_rows = [], []
        for seed in SEEDS:
            model = train(arm, train_grids, seed, device)
            train_rows.append(evaluate(model.to("cpu"), train_grids[:200], seed))
            confirm_rows.append(evaluate(model, confirm_grids, seed))
        results[arm] = {
            "train": {"success": statistics.median(row["success"] for row in train_rows)},
            "confirm": {
                "success": statistics.median(row["success"] for row in confirm_rows),
                "steps": statistics.median(row["steps"] for row in confirm_rows),
            },
            "rows": {"train": train_rows, "confirm": confirm_rows},
        }

    opt, search = results["opt"], results["search"]
    deltas = [
        s["success"] - o["success"]
        for o, s in zip(opt["rows"]["confirm"], search["rows"]["confirm"], strict=True)
    ]
    median = statistics.median(deltas)
    rng = Random(11)
    boot = []
    for _ in range(10_000):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        boot.append(statistics.median(sample))
    boot.sort()
    low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]

    criteria = {
        "e7_0_both_learn": min(opt["train"]["success"], search["train"]["success"]) >= E7_TRAIN_MIN,
        "e7_1_process_beats_result": median >= E7_H1_MIN and low > 0,
        "e7_2_not_step_inflation": search["confirm"]["steps"]
        <= E7_STEP_RATIO * max(opt["confirm"]["steps"], 1e-9),
    }
    out = {
        "kind": "spiking sos e7 (sequence model, large corpus)",
        "device": device,
        "epochs": EPOCHS,
        "hidden": HIDDEN,
        "train_mazes": TRAIN_MAZES,
        "confirm_mazes": CONFIRM_MAZES,
        "arms": results,
        "paired_deltas": deltas,
        "delta_median": median,
        "delta_ci95": [low, high],
        "criteria": criteria,
        "h1_prime_supported": bool(all(criteria.values())),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    for arm in ("opt", "search"):
        print(
            f"[{arm}] 训练池 {results[arm]['train']['success']:.3f} | "
            f"confirm {results[arm]['confirm']['success']:.3f} | "
            f"步数 {results[arm]['confirm']['steps']:.1f}"
        )
    print(
        f"[E7] 配对差中位 {median:+.3f}，95% 区间 [{low:+.3f}, {high:+.3f}] | "
        f"E7-0 {'过' if criteria['e7_0_both_learn'] else '不过'} | "
        f"E7-1 {'成立' if criteria['e7_1_process_beats_result'] else '不成立'} | "
        f"E7-2 {'过' if criteria['e7_2_not_step_inflation'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
