"""重合检测：同步 vs 延迟，C = A、B 的共享位。

预注册：[`sweep_results/axo_spiking_coincidence_prereg.md`](../sweep_results/axo_spiking_coincidence_prereg.md)。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_spiking_prototype as base
import axo_spiking_sparse_prototype as sparse_mod

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path("D:/AxoData")


def score_for(
    codes: sparse_mod.SparseCodes,
    inputs: list[np.ndarray],
    shared: np.ndarray,
    *,
    alpha: float,
    theta: float,
    spiking: bool,
    beta: float,
    rho: float,
    readout: str,
) -> float:
    currents = [codes.transpose_matvec(codes.matvec(item)) for item in inputs]
    scale = max((float(current.max()) for current in currents), default=0.0)
    if scale > 0:
        currents = [current / scale for current in currents]
    width = codes.width
    v = np.zeros(width, dtype=np.float32)
    z = np.zeros(width, dtype=np.float32)
    adaptation = np.zeros(width, dtype=np.float32)
    refractory = np.zeros(width, dtype=np.int32)
    best = -np.inf
    for current in currents:
        recurrent = base.normalize(codes.transpose_matvec(codes.matvec(z))) if spiking else 0.0
        threshold = theta + beta * adaptation
        v = alpha * v + current + recurrent - (threshold * z if spiking else 0.0)
        if spiking:
            z_new = (v >= threshold).astype(np.float32)
            z_new[refractory > 0] = 0.0
            z = z_new
            refractory = np.where(z > 0, 2, np.maximum(refractory - 1, 0))
            adaptation = rho * adaptation + z
        state = z if (spiking and readout == "spikes") else v
        best = max(best, float(state[shared].sum()))
    return best


def balanced_accuracy(scores: list[float], labels: list[int], threshold: float) -> float:
    predictions = [int(score >= threshold) for score in scores]
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        return float("nan")
    tpr = sum(predictions[index] for index in positives) / len(positives)
    tnr = sum(1 - predictions[index] for index in negatives) / len(negatives)
    return (tpr + tnr) / 2


def best_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    values = sorted(set(scores))
    candidates = [values[0] - 1.0, *values, values[-1] + 1.0]
    best = max(
        ((balanced_accuracy(scores, labels, threshold), threshold) for threshold in candidates),
        key=lambda item: item[0],
    )
    return best[1], best[0]


def auc(sync_scores: list[float], delayed_scores: list[float]) -> float:
    wins = 0.0
    for sync, delayed in zip(sync_scores, delayed_scores, strict=True):
        wins += 1.0 if sync > delayed else 0.5 if sync == delayed else 0.0
    return wins / len(sync_scores)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sweep_results/axo_spiking_coincidence/scan.json"),
    )
    args = parser.parse_args(argv)

    probe = base.load_probe_module()
    axo = probe.Axo(args.axo_src)
    items, temporary = base.load_items(args.data_root, axo)
    try:
        lex = [item.lex_sdr for item in items]
        df = probe.document_frequency(lex, axo.width)
        codes = probe.encode_arm("idf-2.0", lex, df, len(items), args.k, axo)
        sparse = sparse_mod.sparse_from_codes(codes, axo.width)
        index_of = {item.memory_id: index for index, item in enumerate(items)}
        turns: dict[tuple[str, int], int] = {}
        for item in items:
            if item.source_ref.startswith("turn://") and item.meta.get("turn_seq") is not None:
                turns[(item.session_id, int(item.meta["turn_seq"]))] = index_of[item.memory_id]
        pairs = []
        for (session, turn), index in sorted(turns.items()):
            following = turns.get((session, turn + 1))
            if following is not None:
                shared = np.intersect1d(codes[index].idx, codes[following].idx)
                if shared.size:
                    pairs.append((index, following, shared))
        zero = np.zeros(axo.width, dtype=np.float32)

        def inputs_for(sync: bool, first: int, second: int) -> list[np.ndarray]:
            a = base.dense_probe(codes[first], axo.width)
            b = base.dense_probe(codes[second], axo.width)
            if sync:
                return [a, b, zero, zero, zero]
            return [a, zero, zero, zero, b]

        models = {
            "linear-a095": {"alpha": 0.95, "spiking": False, "readout": "membrane"},
            "linear-a098": {"alpha": 0.98, "spiking": False, "readout": "membrane"},
            "lif-a095": {"alpha": 0.95, "spiking": True, "beta": 0.0},
            "lif-a098": {"alpha": 0.98, "spiking": True, "beta": 0.0},
            "alif-b1": {"alpha": 0.98, "spiking": True, "beta": 1.0, "rho": 0.9},
        }
        thetas = (0.3, 0.5, 0.7)
        results = {}
        for name, model in models.items():
            readouts = ("membrane",) if not model["spiking"] else ("membrane", "spikes")
            for theta in thetas if model["spiking"] else (0.5,):
                for readout in readouts:
                    sync_scores = []
                    delayed_scores = []
                    for first, second, shared in pairs:
                        sync_scores.append(
                            score_for(
                                sparse,
                                inputs_for(True, first, second),
                                shared,
                                alpha=float(model["alpha"]),
                                theta=theta,
                                spiking=bool(model["spiking"]),
                                beta=float(model.get("beta", 0.0)),
                                rho=float(model.get("rho", 0.9)),
                                readout=readout,
                            )
                        )
                        delayed_scores.append(
                            score_for(
                                sparse,
                                inputs_for(False, first, second),
                                shared,
                                alpha=float(model["alpha"]),
                                theta=theta,
                                spiking=bool(model["spiking"]),
                                beta=float(model.get("beta", 0.0)),
                                rho=float(model.get("rho", 0.9)),
                                readout=readout,
                            )
                        )
                    results[(name, theta, readout)] = (sync_scores, delayed_scores)

        folds = []
        for fold in (0, 1):
            train = [index for index in range(len(pairs)) if index % 2 != fold]
            test = [index for index in range(len(pairs)) if index % 2 == fold]

            def best_for(prefix: str, train=train, test=test):
                candidates = []
                for (name, theta, readout), (sync, delayed) in results.items():
                    if not name.startswith(prefix):
                        continue
                    scores = [sync[index] for index in train] + [delayed[index] for index in train]
                    labels = [1] * len(train) + [0] * len(train)
                    threshold, train_acc = best_threshold(scores, labels)
                    test_scores = [sync[index] for index in test] + [
                        delayed[index] for index in test
                    ]
                    test_labels = [1] * len(test) + [0] * len(test)
                    test_acc = balanced_accuracy(test_scores, test_labels, threshold)
                    test_auc = auc(
                        [sync[index] for index in test], [delayed[index] for index in test]
                    )
                    candidates.append(
                        {
                            "model": name,
                            "theta": theta,
                            "readout": readout,
                            "train_acc": train_acc,
                            "test_acc": test_acc,
                            "test_auc": test_auc,
                        }
                    )
                return max(candidates, key=lambda item: item["train_acc"])

            linear = best_for("linear")
            spiking = max([best_for("lif"), best_for("alif")], key=lambda item: item["train_acc"])
            folds.append(
                {
                    "fold": fold,
                    "linear": linear,
                    "spiking": spiking,
                    "delta_acc": spiking["test_acc"] - linear["test_acc"],
                    "delta_auc": spiking["test_auc"] - linear["test_auc"],
                }
            )
            print(
                f"fold {fold}: linear {linear['model']} θ={linear['theta']} {linear['readout']} "
                f"test acc {linear['test_acc']:.4f} auc {linear['test_auc']:.4f}; "
                f"spiking {spiking['model']} θ={spiking['theta']} {spiking['readout']} "
                f"test acc {spiking['test_acc']:.4f} auc {spiking['test_auc']:.4f}; "
                f"Δacc {folds[-1]['delta_acc']:+.4f} Δauc {folds[-1]['delta_auc']:+.4f}"
            )
        summary = {
            "n_pairs": len(pairs),
            "folds": folds,
            "mean_delta_acc": float(np.mean([fold["delta_acc"] for fold in folds])),
            "mean_delta_auc": float(np.mean([fold["delta_auc"] for fold in folds])),
        }
        print(
            f"mean Δacc {summary['mean_delta_acc']:+.4f} mean Δauc {summary['mean_delta_auc']:+.4f}"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[落盘] {args.out}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
