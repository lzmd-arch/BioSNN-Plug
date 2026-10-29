"""阶段五 M3n：比较五个 embedding 新颖度分数（判据见 sweep_results/phase5_m3n_prereg.md）。

只用 embedding、不用标签；触发器按 M3m 的众数参数固定，只作参考工作点，不进判据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import SpikingTrigger  # noqa: E402
from biosnn_mcp.embeddings import OllamaEmbedding  # noqa: E402
from biosnn_mcp.roc import auc, cosine, normalise, workpoint  # noqa: E402

OUT = REPO / "sweep_results" / "phase5_m3n"
DEFAULT_STREAM = REPO / "sweep_results" / "phase5_m3l" / "stream.json"
EXPECTED_STREAM_SHA256 = "4fc74296112f6d29414959c22b8df18072f3df0e44853fcf84517655827b6786"

#: M3m 选出的众数参数（触发器只作参考，不进判据）。
TRIGGER_KWARGS = {
    "n_neurons": 64,
    "tau": 20.0,
    "threshold": 1.0,
    "bias": 0.02,
    "noise_sigma": 0.05,
    "input_weight": 1.0,
    "lateral_eta": 0.002,
    "ema_tau": 200.0,
    "k": 1.5,
    "n_hits": 3,
    "hit_window": 10,
    "cooldown": 200,
}
MS_PER_TURN = 100
HIT_WINDOW_MS = 200

#: 判据（改这里必须改预注册文件）。
AUC_MIN = 0.80
PER_CONVERSATION_MIN = 0.70
CONVERSATIONS_MIN = 4


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def load_stream(path: Path, *, expected_sha256: str) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(
            f"fixture 的 sha256 与预期不一致：{digest}（预期 {expected_sha256}）；换数据要新预注册。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def embed_conversations(stream: dict, *, model: str, host: str) -> list[list[list[float]]]:
    embedder = OllamaEmbedding(model=model, host=host, timeout=240.0, opener=_opener())
    return [
        embedder.embed([turn["text"] for turn in conversation["turns"]])
        for conversation in stream["conversations"]
    ]


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def score_r5(vectors: list[list[float]], *, history: int = 5) -> list[float]:
    normalised = [normalise(vector) for vector in vectors]
    scores = [0.0]
    for index in range(1, len(normalised)):
        window = normalised[max(0, index - history) : index]
        scores.append(_clip(1.0 - max(cosine(normalised[index], other) for other in window)))
    return scores


def score_rall(vectors: list[list[float]], **_kwargs) -> list[float]:
    normalised = [normalise(vector) for vector in vectors]
    scores = [0.0]
    for index in range(1, len(normalised)):
        history = normalised[:index]
        scores.append(_clip(1.0 - max(cosine(normalised[index], other) for other in history)))
    return scores


def score_rmean(vectors: list[list[float]], *, history: int = 5) -> list[float]:
    normalised = [normalise(vector) for vector in vectors]
    scores = [0.0]
    for index in range(1, len(normalised)):
        window = normalised[max(0, index - history) : index]
        scores.append(
            _clip(1.0 - sum(cosine(normalised[index], other) for other in window) / len(window))
        )
    return scores


def score_rcentroid(vectors: list[list[float]], **_kwargs) -> list[float]:
    normalised = [normalise(vector) for vector in vectors]
    scores = [0.0]
    for index in range(1, len(normalised)):
        history = normalised[:index]
        centroid = [sum(column) / len(history) for column in zip(*history, strict=True)]
        scores.append(_clip(1.0 - cosine(normalised[index], normalise(centroid))))
    return scores


def score_rz(vectors: list[list[float]], **_kwargs) -> list[float]:
    """`R5` 在会话内的在线 z 分数，再过 logistic 映到 [0, 1]（AUC 不受映射影响）。"""
    base = score_r5(vectors)
    scores = [0.0]
    for index in range(1, len(base)):
        history = base[:index]
        mean = sum(history) / len(history)
        variance = sum((value - mean) ** 2 for value in history) / len(history)
        sd = math.sqrt(max(variance, 1e-12))
        z = (base[index] - mean) / sd
        scores.append(1.0 / (1.0 + math.exp(-z)))
    return scores


SCORES = {
    "R5": score_r5,
    "Rall": score_rall,
    "Rmean": score_rmean,
    "Rcentroid": score_rcentroid,
    "Rz": score_rz,
}


def run_trigger(saliences: list[float], *, seed: int) -> tuple[list[bool], list[float]]:
    trigger = SpikingTrigger(seed=seed, **TRIGGER_KWARGS)
    fired: list[bool] = []
    z_scores: list[float] = []
    for salience in saliences:
        for _ in range(MS_PER_TURN):
            fired.append(trigger.process(salience))
            z_scores.append(trigger.rate_z)
    return fired, z_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3n 分数比较")
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--expect-sha256",
        default=EXPECTED_STREAM_SHA256,
        help="fixture 的 sha256（默认 M3l 那份；M3o 复测时传新 fixture 的）",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    stream = load_stream(args.stream, expected_sha256=args.expect_sha256)
    target_turns = list(stream["target_turns"])
    labels = [turn in target_turns for turn in range(stream["turns_per_conversation"])]
    vectors = embed_conversations(stream, model=args.model, host=args.host)

    rows = []
    for name, scorer in SCORES.items():
        per_conversation = []
        pooled_scores: list[float] = []
        pooled_labels: list[bool] = []
        workpoint_rows = []
        for index, conversation in enumerate(vectors):
            scores = scorer(conversation)
            per_conversation.append(auc(scores, labels))
            pooled_scores.extend(scores)
            pooled_labels.extend(labels)
            for seed in range(5):
                fired, z_scores = run_trigger(scores, seed=seed)
                point = workpoint(
                    fired,
                    target_turns,
                    ms_per_turn=MS_PER_TURN,
                    hit_window_ms=HIT_WINDOW_MS,
                )
                peaks = [
                    max(z_scores[turn * MS_PER_TURN : (turn + 1) * MS_PER_TURN])
                    for turn in range(len(z_scores) // MS_PER_TURN)
                ]
                workpoint_rows.append(
                    {
                        "conversation": index,
                        "seed": seed,
                        "hits": point["hits"],
                        "false": point["false"],
                        "trigger_auc": auc(peaks, labels),
                    }
                )
        rows.append(
            {
                "score": name,
                "per_conversation_auc": per_conversation,
                "pooled_auc": auc(pooled_scores, pooled_labels),
                "conversations_at_or_above": sum(
                    1 for value in per_conversation if value >= PER_CONVERSATION_MIN
                ),
                "trigger_reference": workpoint_rows,
            }
        )
        print(
            f"{name:<10} pooled {rows[-1]['pooled_auc']:.4f} per-conv "
            + " ".join(f"{value:.2f}" for value in per_conversation),
            flush=True,
        )

    winners = [
        row
        for row in rows
        if row["pooled_auc"] >= AUC_MIN and row["conversations_at_or_above"] >= CONVERSATIONS_MIN
    ]
    summary = {
        "kind": "phase5-m3n: which embedding-novelty score separates memorable turns",
        "stream_sha256": args.expect_sha256,
        "criteria": {
            "auc_min": AUC_MIN,
            "per_conversation_min": PER_CONVERSATION_MIN,
            "conversations_min": CONVERSATIONS_MIN,
        },
        "trigger_kwargs": TRIGGER_KWARGS,
        "ms_per_turn": MS_PER_TURN,
        "scores": rows,
        "winners": [row["score"] for row in winners],
        "score_family_found": bool(winners),
        "note": "描述性比较（同一批 5 段）；过线者需在新 fixture 上复测才算成立",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "found": summary["score_family_found"],
                "winners": summary["winners"],
                "table": [
                    {
                        "score": row["score"],
                        "pooled": round(row["pooled_auc"], 4),
                        "above": row["conversations_at_or_above"],
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
