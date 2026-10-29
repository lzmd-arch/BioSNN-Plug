"""D7：峰值之后的退化归因（判据见 sweep_results/rstdp_spiking_degradation_prereg.md）。

5 个种子、每 10 回合评测与探针、每回合熵与 |δ|。对每个种子判定下降与哪个量同步：
熵再坍缩 / 列范数漂移 / Critic 漂移 / 多因 / 无法归因。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.rstdp_spiking.cartpole import train

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_degradation"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _signature(seed_row: dict) -> dict:
    evaluations = seed_row["eval_scores"]
    if len(evaluations) < 2:
        return {"degraded": False, "reason": "no_evaluation"}
    peak = max(evaluations, key=lambda point: point["mean_steps"])
    last = evaluations[-1]
    degraded = last["mean_steps"] < 0.7 * peak["mean_steps"]
    if not degraded:
        return {"degraded": False, "reason": "no_degradation"}

    # 退化起点：峰值之后第一次跌破 0.7×峰值 的评测
    onset = next(
        point
        for point in evaluations
        if point["episode"] > peak["episode"] and point["mean_steps"] < 0.7 * peak["mean_steps"]
    )
    peak_episode, onset_episode = peak["episode"], onset["episode"]

    entropies = seed_row["entropies"]
    before_onset = entropies[max(0, onset_episode - 21) : max(1, onset_episode - 1)]
    entropy_recollapse = _mean(before_onset) < 0.05

    def probe_at(episode: int) -> dict | None:
        candidates = [p for p in seed_row["probes"] if p["episode"] <= episode]
        return candidates[-1] if candidates else None

    peak_probe, onset_probe = probe_at(peak_episode), probe_at(onset_episode)
    norm_drift = False
    if peak_probe and onset_probe:
        ratios = [
            onset / peak if peak > 0 else 1.0
            for onset, peak in zip(onset_probe["column_l1"], peak_probe["column_l1"], strict=False)
        ]
        norm_drift = any(ratio > 2.0 or ratio < 0.5 for ratio in ratios)

    critic = seed_row["critic_absdelta"]
    peak_delta = _mean(critic[max(0, peak_episode - 10) : peak_episode])
    onset_delta = _mean(critic[max(0, onset_episode - 10) : onset_episode])
    critic_drift = peak_delta > 0 and onset_delta > 2.0 * peak_delta

    signatures = [
        name
        for name, hit in (
            ("entropy_recollapse", entropy_recollapse),
            ("norm_drift", norm_drift),
            ("critic_drift", critic_drift),
        )
        if hit
    ]
    if not signatures:
        reason = "unattributed"
    elif len(signatures) == 1:
        reason = signatures[0]
    else:
        reason = "mixed:" + ",".join(signatures)
    return {
        "degraded": True,
        "peak_episode": peak_episode,
        "peak_steps": peak["mean_steps"],
        "onset_episode": onset_episode,
        "last_steps": last["mean_steps"],
        "entropy_before_onset": _mean(before_onset),
        "peak_column_l1": peak_probe["column_l1"] if peak_probe else None,
        "onset_column_l1": onset_probe["column_l1"] if onset_probe else None,
        "peak_absdelta": peak_delta,
        "onset_absdelta": onset_delta,
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D7 退化归因")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--logit-scale-start", type=float, default=0.1)
    parser.add_argument("--logit-scale-end", type=float, default=0.02)
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
            spiking_temperature=True,
            logit_scale_start=args.logit_scale_start,
            logit_scale_end=args.logit_scale_end,
            entropy_log=True,
            probe_every=10,
        )
        row = {
            "seed": seed,
            "eval_scores": result.eval_scores,
            "entropies": list(result.entropies),
            "critic_absdelta": list(result.critic_absdelta),
            "probes": list(result.probes),
        }
        row["signature"] = _signature(row)
        rows.append(row)
        print(f"seed {seed}: {row['signature']['reason']}", flush=True)

    degraded = [row for row in rows if row["signature"]["degraded"]]
    reasons = [row["signature"]["reason"] for row in degraded]
    counts = {reason: reasons.count(reason) for reason in set(reasons)}
    single = [
        reason
        for reason, count in counts.items()
        if count >= 3 and reason not in {"unattributed"} and not reason.startswith("mixed")
    ]
    summary = {
        "kind": "diagnostic: attribution of post-peak degradation",
        "seeds": rows,
        "degraded_seeds": len(degraded),
        "reason_counts": counts,
        "degradation_attributed": len(single) == 1,
        "attributed_reason": single[0] if len(single) == 1 else None,
        "repro": collect(
            "W3 脉冲化：D7 退化归因",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "臂：D6-B（logit_scale 0.1→0.02）",
                "每 10 回合评测/探针；每回合熵与 |δ|",
                "判定表见预注册",
            ],
            device=torch.device("cpu"),
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "degraded": len(degraded),
                "counts": counts,
                "attributed": summary["attributed_reason"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part]


if __name__ == "__main__":
    raise SystemExit(main())
