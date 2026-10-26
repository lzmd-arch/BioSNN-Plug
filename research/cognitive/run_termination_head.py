"""P3 终止头 v1 的执行入口（预注册见 sweep_results/termination_head_v1_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_world_model_closed_loop import collect_episodes
from research.cognitive.termination_head import TerminationHead
from research.cognitive.world_model import auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "termination_head_v1"
AUROC_MIN = 0.75
NOISE = 0.1


def run(seeds: list[int], *, episodes: int, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    data = collect_episodes(episodes, seed=0, noise=NOISE)
    if len(data) < 10:
        raise ValueError("至少需要 10 个回合才能切分。")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(data))
        split = max(1, int(0.75 * len(data)))
        train_episodes = [data[i] for i in order[:split]]
        test_episodes = [data[i] for i in order[split:]]
        train_states = np.concatenate([episode.states for episode in train_episodes])
        train_actions = np.concatenate([episode.actions for episode in train_episodes])
        train_terminated = np.concatenate([episode.terminated for episode in train_episodes])
        test_states = np.concatenate([episode.states for episode in test_episodes])
        test_actions = np.concatenate([episode.actions for episode in test_episodes])
        test_terminated = np.concatenate([episode.terminated for episode in test_episodes])

        head = TerminationHead()
        head.fit(train_states, train_actions, train_terminated)
        scores = head.predict_proba(test_states, test_actions)
        rate = float(np.mean(test_terminated))
        value = (
            auroc(scores, test_terminated.astype(int))
            if rate > 0 and 0 < np.sum(test_terminated) < test_terminated.size
            else None
        )
        rows.append(
            {
                "seed": seed,
                "auroc": value,
                "test_termination_rate": rate,
                "test_positives": int(np.sum(test_terminated)),
                "test_steps": int(test_terminated.size),
                "train_positives": int(np.sum(train_terminated)),
            }
        )
        print(
            f"seed {seed}: AUROC {value} | test positives {rows[-1]['test_positives']}/"
            f"{rows[-1]['test_steps']}",
            flush=True,
        )

    valid = [row["auroc"] for row in rows if row["auroc"] is not None]
    gate = {
        "auroc_mean": float(np.mean(valid)) if valid else None,
        "valid_splits": len(valid),
        "ready": bool(valid) and float(np.mean(valid)) >= AUROC_MIN,
    }
    summary = {
        "experiment": "世界模型 P3 终止头 v1",
        "config": {"policy_noise": NOISE, "episodes": episodes, "seeds": seeds},
        "per_seed": rows,
        "gate": gate,
        "repro": collect(
            "世界模型 P3 终止头 v1",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "类平衡 L2 逻辑回归（IRLS），特征图与 ridge quadratic 相同",
                "按回合 75/25 切分；测试回合不参与拟合",
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
    parser = argparse.ArgumentParser(description="P3 终止头 v1")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(parse_seeds(args.seeds), episodes=10 if args.smoke else args.episodes, out=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
