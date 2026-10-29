"""D8：贪心读出还剩多少状态信息（判据见 sweep_results/rstdp_spiking_readout_prereg.md）。

只加记录：训练时每 100 回合在极角网格上采窗口，记两动作计数差 Δ=counts_left−counts_right
的均值与标准差；训练后用最终权重再扫 θ/θ̇/x/ẋ 四个轴。回答末期动作恒定的原因。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.common.provenance import collect
from research.components.cartpole_encoding import encode_state
from research.rstdp_spiking.cartpole import train

OUT = Path(__file__).resolve().parents[2] / "sweep_results" / "rstdp_spiking_readout"

ANGLES = [round(-0.2 + 0.02 * index, 2) for index in range(21)]
PROBE_ANGLES = tuple((angle, f"a{index}") for index, angle in enumerate(ANGLES))
AXES = (
    ("theta", 2, [-0.2, -0.1, 0.0, 0.1, 0.2]),
    ("thetadot", 3, [-1.0, -0.5, 0.0, 0.5, 1.0]),
    ("x", 0, [-1.0, -0.5, 0.0, 0.5, 1.0]),
    ("xdot", 1, [-1.0, -0.5, 0.0, 0.5, 1.0]),
)


def _scan(
    actor, centers, sigma: float, *, seed: int, index: int, value: float, repeats: int
) -> dict:
    state = torch.zeros(4)
    state[index] = value
    features = encode_state(state, centers, sigma)
    generator = torch.Generator(device="cpu").manual_seed(
        seed * 100_000 + index * 100 + int(value * 100)
    )
    deltas = []
    for _ in range(repeats):
        _, post = actor.window_spikes(features, generator=generator)
        counts = post.sum(dim=0).squeeze(0)
        deltas.append(float(counts[0] - counts[1]))
    mean = sum(deltas) / repeats
    sd = (sum((value - mean) ** 2 for value in deltas) / repeats) ** 0.5
    return {"value": value, "delta_mean": mean, "delta_sd": sd}


def _verdict(last_probe: dict, repeats: int) -> dict:
    deltas = [last_probe[f"delta_mean_a{index}"] for index in range(len(ANGLES))]
    sds = [last_probe[f"delta_sd_a{index}"] for index in range(len(ANGLES))]
    sign_constant = (deltas[0] > 0) == (deltas[-1] > 0)
    spread = max(deltas) - min(deltas)
    snr = statistics.median(
        abs(delta) / sd if sd > 0 else float("inf") for delta, sd in zip(deltas, sds, strict=True)
    )
    if not sign_constant:
        reason = "greedy_should_move"
    elif spread < 0.5:
        reason = "no_state_information"
    else:
        reason = "biased_but_informative"
    return {
        "delta_at_neg": deltas[0],
        "delta_at_pos": deltas[-1],
        "sign_constant": sign_constant,
        "spread": spread,
        "median_snr": snr,
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D8 贪心读出的状态信息")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    rows = []
    for seed in seeds:
        result = train(
            seed=seed,
            episodes=args.episodes,
            eval_every=25,
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
            logit_scale_start=0.1,
            logit_scale_end=0.02,
            probe_every=100,
            probe_angles=PROBE_ANGLES,
            keep_actor=True,
            actor_after="final",
        )
        last_probe = result.probes[-1]
        axes = {}
        for axis, index, values in AXES:
            axes[axis] = [
                _scan(
                    result.actor,
                    result.centers,
                    1.0,
                    seed=seed,
                    index=index,
                    value=value,
                    repeats=args.repeats,
                )
                for value in values
            ]
        verdict = _verdict(last_probe, args.repeats)
        rows.append(
            {
                "seed": seed,
                "angles": ANGLES,
                "last_probe": last_probe,
                "verdict": verdict,
                "axes": axes,
                "best_steps": result.best_steps,
                "last_steps": result.eval_scores[-1]["mean_steps"] if result.eval_scores else None,
            }
        )
        print(f"seed {seed}: {verdict['reason']} (spread {verdict['spread']:.2f})", flush=True)

    reasons = [row["verdict"]["reason"] for row in rows]
    counts = {reason: reasons.count(reason) for reason in set(reasons)}
    majority = [reason for reason, count in counts.items() if count >= 3]
    summary = {
        "kind": "diagnostic: how much state information the greedy readout retains",
        "angles": ANGLES,
        "seeds": rows,
        "reason_counts": counts,
        "majority_reason": majority[0] if len(majority) == 1 else None,
        "repro": collect(
            "W3 脉冲化：D8 读出状态信息",
            seeds=",".join(str(seed) for seed in seeds),
            notes=[
                "臂：D6-B（logit_scale 0.1→0.02）",
                "每 100 回合在极角网格采 20 窗口；训练后扫四个轴",
            ],
            device=torch.device("cpu"),
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"counts": counts, "majority": summary["majority_reason"]}, ensure_ascii=False)
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
