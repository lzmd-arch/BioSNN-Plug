"""阶段五 M3l：真实对话流上的触发 ROC（判据见 sweep_results/phase5_m3l_prereg.md）。

数据是 `build_phase5_m3l_stream.py` 生成的固定 fixture（5 段 × 20 轮，25 个目标轮）。
两个检测器用**同一批向量**：

* 规则门：`R_t = 1 − max_{1≤t−j≤5} cos(e_t, e_j)`；
* 脉冲触发：把 `R_t` 当每步脉冲概率，每轮展开 50 ms，推进 M3k 参数的 `SpikingTrigger`；
  分数 `S_t` 是该轮窗口内发放率 z 分数的峰值。

ROC/AUC 在**轮**上算（25 目标 vs 75 普通）；工作点判据另跑一遍看触发时刻。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import SpikingTrigger  # noqa: E402
from biosnn_mcp.embeddings import OllamaEmbedding  # noqa: E402
from biosnn_mcp.roc import auc, novelty_scores, workpoint  # noqa: E402

OUT = REPO / "sweep_results" / "phase5_m3l"

#: 每轮展开成多少毫秒（预注册写死）。
MS_PER_TURN = 50
#: 命中窗口：目标轮起点起 200 ms（= 4 轮，M3k 的 hit window）。
HIT_WINDOW_MS = 200
#: M3k 的原参数，一个都不改。
TRIGGER_KWARGS = {
    "n_neurons": 64,
    "tau": 20.0,
    "threshold": 1.0,
    "bias": 0.02,
    "noise_sigma": 0.05,
    "input_weight": 1.0,
    "lateral_eta": 0.002,
    "ema_tau": 200.0,
    "k": 3.0,
    "n_hits": 5,
    "hit_window": 10,
    "cooldown": 200,
}

#: 判据阈值（改这里必须改预注册文件）。
AUC_MIN = 0.80
AUC_GAP_MAX = 0.05
HITS_MIN = 4
FALSE_MAX = 2


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def load_stream(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("conversations"):
        raise ValueError("fixture 里没有 conversations。")
    return payload


def embed_turns(stream: dict, *, model: str, host: str) -> list[list[list[float]]]:
    embedder = OllamaEmbedding(model=model, host=host, opener=_opener())
    result = []
    for conversation in stream["conversations"]:
        texts = [turn["text"] for turn in conversation["turns"]]
        result.append(embedder.embed(texts))
    return result


def run_trigger(saliences: list[float], *, seed: int) -> tuple[list[bool], list[float]]:
    """每轮展开 `MS_PER_TURN` 步；返回触发标记与世界步号对齐的 z 分数。"""
    trigger = SpikingTrigger(seed=seed, **TRIGGER_KWARGS)
    fired: list[bool] = []
    z_scores: list[float] = []
    for salience in saliences:
        for _ in range(MS_PER_TURN):
            fired.append(trigger.process(salience))
            z_scores.append(trigger.rate_z)
    return fired, z_scores


def pulse_scores(z_scores: list[float]) -> list[float]:
    """每轮窗口内 z 分数的峰值（与规则门同长度、同方向：越高越该触发）。"""
    return [
        max(z_scores[turn * MS_PER_TURN : (turn + 1) * MS_PER_TURN])
        for turn in range(len(z_scores) // MS_PER_TURN)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3l 真实对话流触发 ROC")
    parser.add_argument("--stream", type=Path, default=OUT / "stream.json")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    stream = load_stream(args.stream)
    target_turns = list(stream["target_turns"])
    vectors = embed_turns(stream, model=args.model, host=args.host)
    rule = [novelty_scores(conversation) for conversation in vectors]

    labels: list[bool] = []
    rule_all: list[float] = []
    for conversation, scores in zip(stream["conversations"], rule, strict=True):
        for turn, score in zip(conversation["turns"], scores, strict=True):
            labels.append(bool(turn["target"]))
            rule_all.append(score)
    auc_rule = auc(rule_all, labels)

    rows = []
    for seed in seeds:
        pulse_all: list[float] = []
        per_conversation = []
        determinism = True
        for scores in rule:
            fired, z_scores = run_trigger(scores, seed=seed)
            fired_again, _ = run_trigger(scores, seed=seed)
            determinism = determinism and fired == fired_again
            pulse_all.extend(pulse_scores(z_scores))
            per_conversation.append(
                workpoint(
                    fired,
                    target_turns,
                    ms_per_turn=MS_PER_TURN,
                    hit_window_ms=HIT_WINDOW_MS,
                )
            )
        auc_pulse = auc(pulse_all, labels)
        rows.append(
            {
                "seed": seed,
                "auc_pulse": auc_pulse,
                "auc_rule": auc_rule,
                "per_conversation": per_conversation,
                "deterministic": determinism,
            }
        )
        print(
            f"seed {seed}: AUC pulse {auc_pulse:.3f} rule {auc_rule:.3f}; "
            f"hits {[item['hits'] for item in per_conversation]} "
            f"false {[item['false'] for item in per_conversation]}",
            flush=True,
        )

    pulse_aucs = [row["auc_pulse"] for row in rows]
    criteria = {
        "auc_median": statistics.median(pulse_aucs),
        "auc_seeds_above_min": sum(1 for value in pulse_aucs if value >= AUC_MIN),
        "gap_holds": sum(1 for row in rows if row["auc_pulse"] >= row["auc_rule"] - AUC_GAP_MAX),
        "workpoint_holds": sum(
            1
            for row in rows
            if all(
                item["hits"] >= HITS_MIN and item["false"] <= FALSE_MAX
                for item in row["per_conversation"]
            )
        ),
        "deterministic": sum(1 for row in rows if row["deterministic"]),
    }
    ready = (
        criteria["auc_median"] >= AUC_MIN
        and criteria["auc_seeds_above_min"] >= 4
        and criteria["gap_holds"] >= 4
        and criteria["workpoint_holds"] >= 4
        and criteria["deterministic"] == len(seeds)
    )
    summary = {
        "kind": "phase5-m3l: trigger ROC on a real-model dialogue fixture",
        "stream": {
            "path": str(args.stream),
            "sha256": hashlib.sha256(args.stream.read_bytes()).hexdigest(),
            "model": stream.get("model"),
            "prompt_sha256": stream.get("prompt_sha256"),
            "conversations": len(stream["conversations"]),
            "turns": len(labels),
            "targets": sum(1 for label in labels if label),
        },
        "embedding_model": args.model,
        "ms_per_turn": MS_PER_TURN,
        "hit_window_ms": HIT_WINDOW_MS,
        "trigger_kwargs": TRIGGER_KWARGS,
        "criteria_thresholds": {
            "auc_min": AUC_MIN,
            "auc_gap_max": AUC_GAP_MAX,
            "hits_min": HITS_MIN,
            "false_max": FALSE_MAX,
        },
        "auc_rule": auc_rule,
        "rule_scores": rule_all,
        "seeds": rows,
        "criteria": criteria,
        "m3l_ready": ready,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"criteria": criteria, "ready": ready}, ensure_ascii=False))
    return 0


def _parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part]


if __name__ == "__main__":
    raise SystemExit(main())
