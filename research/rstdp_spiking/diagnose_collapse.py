"""D5：训练末期是策略坍缩还是输出层沉默（判据见 sweep_results/rstdp_spiking_collapse_prereg.md）。

只加记录、不改规则：开 ``entropy_log``、每 10 回合评测、每 25 回合在极角 ±0.2 上采窗口探针。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.rstdp_spiking.cartpole import train

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_collapse"


def _verdict(rows: list[dict]) -> dict:
    collapsed = sum(1 for row in rows if row["final_entropy"] < 0.10)
    silent = sum(1 for row in rows if row["silent_fraction"] >= 0.5)
    if collapsed >= 7 and silent < 3:
        label = "collapse"
    elif silent >= 7:
        label = "silence"
    elif collapsed < 3 and silent < 3:
        label = "neither"
    else:
        label = "mixed"
    return {"collapsed_seeds": collapsed, "silent_seeds": silent, "verdict": label}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D5 坍缩/沉默诊断")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    rows = []
    for seed in seeds:
        result = train(
            seed=seed,
            episodes=args.episodes,
            eval_every=10,
            eval_episodes=10,
            n_features=64,
            sigma=1.0,
            steps=25,
            learning_rate=1e-3,
            baseline_decay=0.05,
            critic_kind="pop",
            critic_rate=0.1,
            gamma=0.99,
            pop_post="gradient",
            warmup=10,
            device=torch.device("cpu"),
            trace_rule="count-aligned",
            entropy_log=True,
            probe_every=25,
        )
        final_entropy = statistics.fmean(result.entropies[-100:]) if result.entropies else 0.0
        silent = sum(
            1 for probe in result.probes if probe["spikes_pos"] == 0 and probe["spikes_neg"] == 0
        )
        silent_fraction = silent / len(result.probes) if result.probes else 0.0
        peak = max(result.eval_scores, key=lambda point: point["mean_steps"])
        rows.append(
            {
                "seed": seed,
                "final_entropy": final_entropy,
                "entropy_at_100": result.entropies[99] if len(result.entropies) > 99 else None,
                "silent_fraction": silent_fraction,
                "peak_episode": peak["episode"],
                "peak_steps": peak["mean_steps"],
                "last_steps": result.eval_scores[-1]["mean_steps"],
                "probes": list(result.probes),
                "eval_scores": result.eval_scores,
                "entropies": list(result.entropies),
            }
        )
        print(
            f"seed {seed}: final entropy {final_entropy:.3f}, silent {silent_fraction:.2f}, "
            f"peak {peak['mean_steps']:.1f}@{peak['episode']}, last "
            f"{result.eval_scores[-1]['mean_steps']:.1f}",
            flush=True,
        )

    verdict = _verdict(rows)
    summary = {
        "kind": "diagnostic: collapse vs silence at end of training",
        "seeds": rows,
        "verdict": verdict,
        "median_peak": statistics.median([row["peak_steps"] for row in rows]),
        "median_last": statistics.median([row["last_steps"] for row in rows]),
        "repro": collect(
            "W3 脉冲化：D5 坍缩/沉默诊断",
            seeds=",".join(str(seed) for seed in seeds),
            notes=["记录每回合熵、每 10 回合评测、每 25 回合窗口探针", "不改规则、不调超参"],
            device=torch.device("cpu"),
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


def _parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part]


if __name__ == "__main__":
    raise SystemExit(main())
