"""时序读出的受控优化：同一读出/阈值预算 + 2 折交叉验证。

预注册：[`sweep_results/axo_spiking_temporal_optimize_prereg.md`](../sweep_results/axo_spiking_temporal_optimize_prereg.md)。
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
import axo_spiking_temporal_readout as temporal_mod

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path("D:/AxoData")

READOUTS = ("sum_inv", "min_inv", "top3_inv", "top5_inv", "exp1", "exp2", "window1", "window2")


def scores_for(codes: sparse_mod.SparseCodes, times: np.ndarray, name: str) -> np.ndarray:
    t = times[codes.idx]
    weights = codes.weights
    finite = np.isfinite(t) & (weights > 0.0)
    if name == "sum_inv":
        return np.where(finite, weights / (1.0 + t), 0.0).sum(axis=1)
    if name == "min_inv":
        earliest = np.where(finite, t, np.inf).min(axis=1)
        return np.where(np.isfinite(earliest), 1.0 / (1.0 + earliest), 0.0)
    if name in ("top3_inv", "top5_inv"):
        k = 3 if name == "top3_inv" else 5
        values = np.where(finite, weights / (1.0 + t), -np.inf)
        selected = np.sort(values, axis=1)[:, -k:]
        selected = np.where(np.isfinite(selected), selected, 0.0)
        return selected.mean(axis=1)
    if name in ("exp1", "exp2"):
        tau = 1.0 if name == "exp1" else 2.0
        return np.where(finite, weights * np.exp(-t / tau), 0.0).sum(axis=1)
    if name in ("window1", "window2"):
        limit = 1 if name == "window1" else 2
        return np.where(finite & (t <= limit), weights, 0.0).sum(axis=1)
    raise ValueError(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sweep_results/axo_spiking_temporal_optimize/scan.json"),
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
                pairs.append((index, following))
        zero = np.zeros(axo.width, dtype=np.float32)
        queries = []
        for pair_index, (first, second) in enumerate(pairs):
            for order, target in (([first, second], first), ([second, first], second)):
                queries.append(
                    {
                        "inputs": [
                            base.dense_probe(codes[order[0]], axo.width),
                            base.dense_probe(codes[order[1]], axo.width),
                            zero,
                            zero,
                            zero,
                        ],
                        "target": target,
                        "fold": pair_index % 2,
                    }
                )

        models = {
            "linear-a095": {"alpha": 0.95, "spiking": False},
            "linear-a098": {"alpha": 0.98, "spiking": False},
            "lif-a095": {"alpha": 0.95, "spiking": True, "beta": 0.0},
            "lif-a098": {"alpha": 0.98, "spiking": True, "beta": 0.0},
            "alif-b1": {"alpha": 0.98, "spiking": True, "beta": 1.0, "rho": 0.9},
        }
        thetas = (0.3, 0.5, 0.7)
        results: dict[tuple[str, float, str], list[int]] = {}
        for name, model in models.items():
            for theta in thetas:
                times_list = []
                for query in queries:
                    first_cross, first_spike = temporal_mod.first_times(
                        sparse,
                        query["inputs"],
                        alpha=float(model["alpha"]),
                        theta=theta,
                        spiking=bool(model["spiking"]),
                        beta=float(model.get("beta", 0.0)),
                        rho=float(model.get("rho", 0.9)),
                        hetero=0.0,
                        inhibition="none",
                    )
                    times_list.append(first_spike if model["spiking"] else first_cross)
                for readout in READOUTS:
                    correctness = [
                        int(np.argmax(scores_for(sparse, times, readout)) == query["target"])
                        for times, query in zip(times_list, queries, strict=True)
                    ]
                    results[(name, theta, readout)] = correctness

        folds = []
        for fold in (0, 1):
            train_idx = [index for index, query in enumerate(queries) if query["fold"] != fold]
            test_idx = [index for index, query in enumerate(queries) if query["fold"] == fold]

            def best_for(prefix: str, train_idx=train_idx, test_idx=test_idx):
                candidates = []
                for (name, theta, readout), values in results.items():
                    if not name.startswith(prefix):
                        continue
                    train = float(np.mean([values[index] for index in train_idx]))
                    test = float(np.mean([values[index] for index in test_idx]))
                    candidates.append(
                        {
                            "model": name,
                            "theta": theta,
                            "readout": readout,
                            "train": train,
                            "test": test,
                        }
                    )
                return max(candidates, key=lambda item: (item["train"], item["test"]))

            linear = max(
                [best_for("linear")],
                key=lambda item: item["train"],
            )
            spiking = max(
                [best_for("lif"), best_for("alif")],
                key=lambda item: item["train"],
            )
            folds.append(
                {
                    "fold": fold,
                    "linear": linear,
                    "spiking": spiking,
                    "delta": spiking["test"] - linear["test"],
                }
            )
            print(
                f"fold {fold}: linear {linear['model']} θ={linear['theta']} "
                f"{linear['readout']} train {linear['train']:.4f} test {linear['test']:.4f}; "
                f"spiking {spiking['model']} θ={spiking['theta']} {spiking['readout']} "
                f"train {spiking['train']:.4f} test {spiking['test']:.4f}; "
                f"Δ {folds[-1]['delta']:+.4f}"
            )

        average_delta = float(np.mean([fold["delta"] for fold in folds]))
        print(f"average held-out Δ = {average_delta:+.4f}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "n_queries": len(queries),
                    "folds": folds,
                    "average_delta": average_delta,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[落盘] {args.out}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
