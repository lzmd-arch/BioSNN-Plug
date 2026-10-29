"""阶段五 M3i：STDP 侧向传播的自主触发（判据见 phase5_m3i_prereg.md）。

10,000 ms 合成显著性流；脉冲群体触发 vs 规则门。只比较触发决策。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import SpikingTrigger  # noqa: E402

OUT = REPO / "sweep_results" / "phase5_m3i"
EVENTS = [3000, 5000, 7000, 9000]
EVENT_LENGTH = 300
BASELINE_END = 3000


def build_stream() -> list[float]:
    stream = [0.0] * 10_000
    for start in EVENTS:
        for index in range(start, min(start + EVENT_LENGTH, len(stream))):
            stream[index] = 0.5
    return stream


def rule_triggers(stream: list[float], threshold: float = 0.25, cooldown: int = 200) -> list[int]:
    """规则门：salience > 0.25 且冷却 200 ms（预注册写死的对照）。"""
    triggers = []
    cooldown_left = 0
    for index, salience in enumerate(stream):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        if salience > threshold:
            triggers.append(index)
            cooldown_left = cooldown
    return triggers


def score_triggers(triggers: list[int]) -> dict:
    hits = sum(1 for start in EVENTS if any(start <= moment <= start + 200 for moment in triggers))
    false_triggers = sum(1 for moment in triggers if moment < BASELINE_END)
    return {"hits": hits, "false": false_triggers, "total": len(triggers)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3i 脉冲触发")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--hit-window", type=int, default=1, help="M3j：冲动累计窗口（1=连续）")
    parser.add_argument("--lateral-eta", type=float, default=0.01, help="M3k：侧向 STDP 步长")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    stream = build_stream()
    rule = score_triggers(rule_triggers(stream))
    rows = []
    for seed in seeds:
        trigger = SpikingTrigger(
            seed=seed, hit_window=args.hit_window, lateral_eta=args.lateral_eta
        )
        moments = [index for index, salience in enumerate(stream) if trigger.process(salience)]
        trigger_again = SpikingTrigger(
            seed=seed, hit_window=args.hit_window, lateral_eta=args.lateral_eta
        )
        moments_again = [
            index for index, salience in enumerate(stream) if trigger_again.process(salience)
        ]
        metrics = score_triggers(moments)
        row = {
            "seed": seed,
            "triggers": moments,
            "hits": metrics["hits"],
            "false": metrics["false"],
            "deterministic": moments == moments_again,
            "not_worse": metrics["hits"] >= rule["hits"] and metrics["false"] <= rule["false"] + 1,
        }
        rows.append(row)
        print(
            f"seed {seed}: hits {row['hits']}/4, false {row['false']}, "
            f"total {len(moments)}, det {row['deterministic']}",
            flush=True,
        )

    gate = {
        "hits": all(row["hits"] >= 3 for row in rows),
        "false": all(row["false"] <= 1 for row in rows),
        "deterministic": all(row["deterministic"] for row in rows),
        "not_worse_than_rule": all(row["not_worse"] for row in rows),
    }
    gate["ready"] = all(gate.values())
    summary = {
        "kind": "protocol: spiking trigger with lateral STDP vs rule gate",
        "hit_window": args.hit_window,
        "lateral_eta": args.lateral_eta,
        "events": EVENTS,
        "rule": rule,
        "seeds": rows,
        "gate": gate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"rule": rule, "gate": gate}, ensure_ascii=False, indent=2))
    return 0


def _parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part]


if __name__ == "__main__":
    raise SystemExit(main())
