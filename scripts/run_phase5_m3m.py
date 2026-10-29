"""阶段五 M3m：在真实分数尺度上重标定触发工作点（判据见 sweep_results/phase5_m3m_prereg.md）。

留一会话交叉验证：每次留出 1 段对话评测，用其余 4 段挑（k, n_hits, 每轮毫秒）；5 折轮流。
分数与编码完全沿用 M3l，一格不动。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import SpikingTrigger  # noqa: E402
from biosnn_mcp.embeddings import OllamaEmbedding  # noqa: E402
from biosnn_mcp.roc import auc, novelty_scores, workpoint  # noqa: E402

OUT = REPO / "sweep_results" / "phase5_m3m"
DEFAULT_STREAM = REPO / "sweep_results" / "phase5_m3l" / "stream.json"

#: M3l 记录的 fixture sha256——换数据就是换实验，必须拦住。
EXPECTED_STREAM_SHA256 = "4fc74296112f6d29414959c22b8df18072f3df0e44853fcf84517655827b6786"

#: 参数网格（预注册写死）。
KS = (1.5, 2.0, 2.5, 3.0)
N_HITS = (2, 3, 5)
MS_PER_TURN = (50, 100)

#: M3k 原参数里**不动**的那些。
BASE_KWARGS = {
    "n_neurons": 64,
    "tau": 20.0,
    "threshold": 1.0,
    "bias": 0.02,
    "noise_sigma": 0.05,
    "input_weight": 1.0,
    "lateral_eta": 0.002,
    "ema_tau": 200.0,
    "hit_window": 10,
    "cooldown": 200,
}

HIT_WINDOW_MS = 200
SEEDS = (0, 1, 2, 3, 4)

#: 判据（改这里必须改预注册文件）。
HITS_MIN = 4
FALSE_MAX = 2
FOLDS_MIN = 4
AUC_GAP_MAX = 0.05


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def load_stream(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_STREAM_SHA256:
        raise SystemExit(f"fixture 的 sha256 与 M3l 记录不一致：{digest}；换数据要新预注册。")
    return json.loads(path.read_text(encoding="utf-8"))


def embed_conversations(stream: dict, *, model: str, host: str) -> list[list[list[float]]]:
    embedder = OllamaEmbedding(model=model, host=host, timeout=240.0, opener=_opener())
    return [
        embedder.embed([turn["text"] for turn in conversation["turns"]])
        for conversation in stream["conversations"]
    ]


def run_trigger(
    saliences: list[float],
    *,
    seed: int,
    k: float,
    n_hits: int,
    ms_per_turn: int,
) -> tuple[list[bool], list[float]]:
    trigger = SpikingTrigger(seed=seed, k=k, n_hits=n_hits, **BASE_KWARGS)
    fired: list[bool] = []
    z_scores: list[float] = []
    for salience in saliences:
        for _ in range(ms_per_turn):
            fired.append(trigger.process(salience))
            z_scores.append(trigger.rate_z)
    return fired, z_scores


def turn_scores(z_scores: list[float], *, ms_per_turn: int) -> list[float]:
    return [
        max(z_scores[turn * ms_per_turn : (turn + 1) * ms_per_turn])
        for turn in range(len(z_scores) // ms_per_turn)
    ]


def evaluate_conversation(
    saliences: list[float],
    targets: list[int],
    *,
    seed: int,
    k: float,
    n_hits: int,
    ms_per_turn: int,
) -> dict:
    fired, z_scores = run_trigger(saliences, seed=seed, k=k, n_hits=n_hits, ms_per_turn=ms_per_turn)
    point = workpoint(fired, targets, ms_per_turn=ms_per_turn, hit_window_ms=HIT_WINDOW_MS)
    scores = turn_scores(z_scores, ms_per_turn=ms_per_turn)
    labels = [turn in targets for turn in range(len(scores))]
    return {
        "hits": point["hits"],
        "false": point["false"],
        "scores": scores,
        "labels": labels,
        "triggers": point["triggers"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3m 真实尺度上的工作点重标定")
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    stream = load_stream(args.stream)
    target_turns = list(stream["target_turns"])
    label_turns = [turn in target_turns for turn in range(stream["turns_per_conversation"])]
    vectors = embed_conversations(stream, model=args.model, host=args.host)
    rule = [novelty_scores(conversation) for conversation in vectors]
    auc_rule_per_conversation = [auc(scores, label_turns) for scores in rule]

    grid = [(k, n, ms) for k in KS for n in N_HITS for ms in MS_PER_TURN]
    folds = []
    for hold_out in range(len(stream["conversations"])):
        calibration = [index for index in range(len(rule)) if index != hold_out]
        per_seed = []
        for seed in SEEDS:
            # 校准：只许看校准段的命中和误触发。
            best = None
            for k, n_hits, ms in grid:
                hits = 0
                false = 0
                positives = []
                labels_all = []
                for index in calibration:
                    result = evaluate_conversation(
                        rule[index], target_turns, seed=seed, k=k, n_hits=n_hits, ms_per_turn=ms
                    )
                    hits += result["hits"]
                    false += result["false"]
                    positives.extend(result["scores"])
                    labels_all.extend(result["labels"])
                candidate = {
                    "k": k,
                    "n_hits": n_hits,
                    "ms_per_turn": ms,
                    "calibration_hits": hits,
                    "calibration_false": false,
                    "calibration_auc": auc(positives, labels_all),
                }
                # 选择规则：命中优先 → 误触发少 → AUC 高 → 代价小。
                key = (
                    -hits,
                    false,
                    -candidate["calibration_auc"],
                    ms * 1.0,
                    n_hits,
                )
                if best is None or key < best[0]:
                    best = (key, candidate)
            chosen = best[1]
            held = evaluate_conversation(
                rule[hold_out],
                target_turns,
                seed=seed,
                k=chosen["k"],
                n_hits=chosen["n_hits"],
                ms_per_turn=chosen["ms_per_turn"],
            )
            again = evaluate_conversation(
                rule[hold_out],
                target_turns,
                seed=seed,
                k=chosen["k"],
                n_hits=chosen["n_hits"],
                ms_per_turn=chosen["ms_per_turn"],
            )
            per_seed.append(
                {
                    "seed": seed,
                    "chosen": {k: chosen[k] for k in ("k", "n_hits", "ms_per_turn")},
                    "calibration_auc": chosen["calibration_auc"],
                    "held_hits": held["hits"],
                    "held_false": held["false"],
                    "held_auc": auc(held["scores"], held["labels"]),
                    "held_scores": held["scores"],
                    "held_labels": held["labels"],
                    "deterministic": held["triggers"] == again["triggers"],
                }
            )
        folds.append(
            {
                "hold_out": hold_out,
                "seed_rows": per_seed,
                "held_auc_rule": auc_rule_per_conversation[hold_out],
            }
        )
        print(
            f"fold {hold_out}: "
            + "; ".join(
                f"seed{row['seed']} {row['chosen']} hits {row['held_hits']}/{len(target_turns)} "
                f"false {row['held_false']} auc {row['held_auc']:.2f}"
                for row in per_seed
            ),
            flush=True,
        )

    # 判据按「折」汇总：每折要求 5 个种子都满足工作点与 AUC 差（种子只影响触发随机流）。
    fold_pass = []
    for fold in folds:
        workpoint_ok = all(
            row["held_hits"] >= HITS_MIN and row["held_false"] <= FALSE_MAX
            for row in fold["seed_rows"]
        )
        gap_ok = all(
            row["held_auc"] >= fold["held_auc_rule"] - AUC_GAP_MAX for row in fold["seed_rows"]
        )
        fold_pass.append({"hold_out": fold["hold_out"], "workpoint": workpoint_ok, "gap": gap_ok})

    pooled_scores = [s for fold in folds for row in fold["seed_rows"] for s in row["held_scores"]]
    pooled_labels = [s for fold in folds for row in fold["seed_rows"] for s in row["held_labels"]]
    deterministic = all(row["deterministic"] for fold in folds for row in fold["seed_rows"])
    summary = {
        "kind": "phase5-m3m: recalibrate the trigger operating point on real-scale salience",
        "stream_sha256": EXPECTED_STREAM_SHA256,
        "grid": {"ks": list(KS), "n_hits": list(N_HITS), "ms_per_turn": list(MS_PER_TURN)},
        "criteria": {
            "hits_min": HITS_MIN,
            "false_max": FALSE_MAX,
            "folds_min": FOLDS_MIN,
            "auc_gap_max": AUC_GAP_MAX,
        },
        "folds": folds,
        "fold_pass": fold_pass,
        "workpoint_folds": sum(1 for item in fold_pass if item["workpoint"]),
        "gap_folds": sum(1 for item in fold_pass if item["gap"]),
        "deterministic": deterministic,
        "pooled_auc": auc(pooled_scores, pooled_labels),
        "m3m_ready": (
            sum(1 for item in fold_pass if item["workpoint"]) >= FOLDS_MIN
            and sum(1 for item in fold_pass if item["gap"]) >= FOLDS_MIN
            and deterministic
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "workpoint_folds": summary["workpoint_folds"],
                "gap_folds": summary["gap_folds"],
                "pooled_auc": summary["pooled_auc"],
                "ready": summary["m3m_ready"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
