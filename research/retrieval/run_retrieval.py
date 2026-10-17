"""跨模态检索的基准运行：程序生成配对集上的 R@1 / R@5。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

跑法（模块方式，见 `research/README.md`）::

    uv run python -m research.retrieval.run_retrieval --seeds 0,1,2,3,4

产出：每个种子一个 JSON（``sweep_results/retrieval_procedural/seed<N>.json``）与一份
汇总 ``summary.json``。数字的解释边界写在
``sweep_results/retrieval_procedural_prereg.md``——**先读那份，再读这里的结果**。

## 报什么

| 栏目 | 含义 |
| :--- | :--- |
| ``control_within`` | **读法 (a)**：各插件自身表示的**同模态**检索。编码有没有保住身份 |
| ``bus_direct`` | **读法 (b) 的零联想对照**：把两路总线输出直接比余弦。预注册预期它落在随机水平 |
| ``bus_associated`` | **验收线**：外积 Hebb 联想把文本侧翻到图像侧（及反向）之后的 R@K |
| ``chance`` | 1 / 测试集条数 |

⚠️ ``bus_direct`` 低**不是**失败，是接口的性质；``bus_associated`` 才是这条实验要回答的东西。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from biosnn_bus.plugins.text_zscore import OllamaEmbedding

from research.common.provenance import collect

from .association import (
    DECAY_GRID,
    ETA_GRID,
    DeltaAssociator,
    HebbianAssociator,
    RidgeAssociator,
    evaluate_direction,
    select_delta_hyperparameters,
    select_hyperparameters,
    select_ridge_lambda,
)
from .encode import CachedEmbedding, build_bus, encode_items
from .metrics import recall_at_k
from .procedural import generate, split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "sweep_results" / "retrieval_procedural"
DEFAULT_DELTA_OUT = REPO_ROOT / "sweep_results" / "retrieval_delta"
DEFAULT_RIDGE_OUT = REPO_ROOT / "sweep_results" / "retrieval_ridge"
DEFAULT_CACHE = REPO_ROOT / "data" / "embeddings" / "nomic-embed-text.json"

__all__ = ["main", "run_seed"]


def run_seed(
    *,
    seed: int,
    bus_dim: int,
    test_fraction: float,
    backend_factory,
    associator_kind: str = "hebb",
    representation: str = "bus",
    verbose: bool = True,
) -> dict:
    """跑一个种子，返回该种子的全部指标。"""
    items = generate()
    train, test = split(items, test_fraction=test_fraction, seed=seed)

    bus = build_bus(bus_dim=bus_dim, seed=seed, backend=backend_factory())
    train_repr = encode_items(bus, train)
    test_repr = encode_items(bus, test)

    if representation == "bus":
        train_text, train_image = train_repr.text_bus, train_repr.image_bus
        test_text, test_image = test_repr.text_bus, test_repr.image_bus
    elif representation == "plugin":
        train_text, train_image = train_repr.text_plugin, train_repr.image_plugin
        test_text, test_image = test_repr.text_plugin, test_repr.image_plugin
    else:  # pragma: no cover - argparse 已限制取值
        raise ValueError(f"未知的表示：{representation!r}")

    # 读法 (a)：同模态检索。编码保住身份的话这里应该很高。
    control_within = {
        "text": recall_at_k(
            test_repr.text_plugin, test_repr.text_plugin, k=1, direction="text2text"
        ).recall,
        "image": recall_at_k(
            test_repr.image_plugin, test_repr.image_plugin, k=1, direction="image2image"
        ).recall,
    }

    # 读法 (b) 的零联想对照：直接拿两路表示比。预期 ≈ 随机。
    if test_text.shape[1] == test_image.shape[1]:
        direct = {
            "text2image": recall_at_k(test_text, test_image, k=1, direction="text2image").recall,
            "image2text": recall_at_k(test_image, test_text, k=1, direction="image2text").recall,
        }
    else:
        # 插件原始表示维度不同（文本 256 / 图像 1152），"直接比余弦"没有定义。
        direct = {"text2image": None, "image2text": None, "note": "维度不同，直接比较无定义"}

    # 验收线：在**训练集**上选参并拟合，再到**测试集**上评。
    if associator_kind == "hebb":
        eta_t2i, decay_t2i, grid_t2i = select_hyperparameters(
            train_text, train_image, eta_grid=ETA_GRID, decay_grid=DECAY_GRID
        )
        eta_i2t, decay_i2t, grid_i2t = select_hyperparameters(
            train_image, train_text, eta_grid=ETA_GRID, decay_grid=DECAY_GRID
        )
        assoc_t2i = HebbianAssociator(eta=eta_t2i, decay=decay_t2i).fit(train_text, train_image)
        assoc_i2t = HebbianAssociator(eta=eta_i2t, decay=decay_i2t).fit(train_image, train_text)
        hyper = {
            "text2image": {"eta": eta_t2i, "decay": decay_t2i},
            "image2text": {"eta": eta_i2t, "decay": decay_i2t},
        }
        loo_best = {"text2image": max(grid_t2i.values()), "image2text": max(grid_i2t.values())}
    elif associator_kind == "delta":
        eta_t2i, epochs_t2i, grid_t2i = select_delta_hyperparameters(
            train_text, train_image, seed=seed
        )
        eta_i2t, epochs_i2t, grid_i2t = select_delta_hyperparameters(
            train_image, train_text, seed=seed
        )
        assoc_t2i = DeltaAssociator(eta=eta_t2i, epochs=epochs_t2i, seed=seed).fit(
            train_text, train_image
        )
        assoc_i2t = DeltaAssociator(eta=eta_i2t, epochs=epochs_i2t, seed=seed).fit(
            train_image, train_text
        )
        hyper = {
            "text2image": {
                "eta": eta_t2i,
                "epochs": epochs_t2i,
                "diverged": bool(assoc_t2i.diverged),
            },
            "image2text": {
                "eta": eta_i2t,
                "epochs": epochs_i2t,
                "diverged": bool(assoc_i2t.diverged),
            },
        }
        loo_best = {"text2image": max(grid_t2i.values()), "image2text": max(grid_i2t.values())}
    elif associator_kind == "ridge":
        lam_t2i, grid_t2i = select_ridge_lambda(train_text, train_image, seed=seed)
        lam_i2t, grid_i2t = select_ridge_lambda(train_image, train_text, seed=seed)
        assoc_t2i = RidgeAssociator(lam=lam_t2i).fit(train_text, train_image)
        assoc_i2t = RidgeAssociator(lam=lam_i2t).fit(train_image, train_text)
        hyper = {
            "text2image": {"lam": lam_t2i, "oracle": True, "local": False},
            "image2text": {"lam": lam_i2t, "oracle": True, "local": False},
        }
        loo_best = {"text2image": max(grid_t2i.values()), "image2text": max(grid_i2t.values())}
    else:  # pragma: no cover - argparse 已限制取值
        raise ValueError(f"未知的联想规则：{associator_kind!r}")

    forward = evaluate_direction(assoc_t2i, test_text, test_image, direction="text2image")
    backward = evaluate_direction(assoc_i2t, test_image, test_text, direction="image2text")

    result = {
        "seed": seed,
        "n_train": len(train),
        "n_test": len(test),
        "chance_r1": 1.0 / len(test),
        "control_within": control_within,
        "bus_direct": direct,
        "associator": associator_kind,
        "representation": representation,
        "hyperparameters": hyper,
        "selection_best_r1": loo_best,
        "bus_associated": {
            "text2image": forward,
            "image2text": backward,
            "mean_r1": (forward["r1"] + backward["r1"]) / 2.0,
            "mean_r5": (forward["r5"] + backward["r5"]) / 2.0,
        },
    }
    if verbose:
        print(
            f"seed {seed}: within(text)={control_within['text']:.3f} "
            f"within(image)={control_within['image']:.3f} | "
            f"direct(t2i/i2t)={direct['text2image']:.3f}/{direct['image2text']:.3f} | "
            f"assoc R@1 = {result['bus_associated']['mean_r1']:.3f} "
            f"(chance {result['chance_r1']:.3f})"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨模态检索 R@K（程序生成配对集）")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="逗号分隔的种子，默认 0..4")
    parser.add_argument("--bus-dim", type=int, default=128, help="总线维度，默认 128")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="测试集比例，默认 0.2")
    parser.add_argument(
        "--associator",
        choices=("hebb", "delta", "ridge"),
        default="hebb",
        help="联想规则，默认 hebb",
    )
    parser.add_argument("--out", type=Path, default=None, help="记录输出目录")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="token embedding 缓存")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama embedding 模型")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.out is None:
        args.out = {"hebb": DEFAULT_OUT, "delta": DEFAULT_DELTA_OUT, "ridge": DEFAULT_RIDGE_OUT}[
            args.associator
        ]
    started = time.perf_counter()
    cache = CachedEmbedding(OllamaEmbedding(model=args.model), args.cache, model=args.model)

    def backend_factory() -> CachedEmbedding:
        return cache

    results = []
    for seed in seeds:
        results.append(
            run_seed(
                seed=seed,
                bus_dim=args.bus_dim,
                test_fraction=args.test_fraction,
                backend_factory=backend_factory,
                associator_kind=args.associator,
                representation=args.representation,
            )
        )
        cache.save()

    summary = {
        "experiment": (
            f"cross-modal retrieval, procedural paired set "
            f"({args.associator}, {args.representation}@"
            f"{args.bus_dim if args.representation == 'bus' else 'raw'})"
        ),
        "associator": args.associator,
        "dataset": "procedural (4 shapes x 3 shades x 6 positions x 2 sizes = 144)",
        "bus_dim": args.bus_dim,
        "test_fraction": args.test_fraction,
        "embedding_model": args.model,
        "seeds": seeds,
        "mean_r1": float(np.mean([r["bus_associated"]["mean_r1"] for r in results])),
        "mean_r5": float(np.mean([r["bus_associated"]["mean_r5"] for r in results])),
        "mean_chance": float(np.mean([r["chance_r1"] for r in results])),
        "per_seed": results,
    }
    summary["repro"] = collect(
        "检索线 W4：跨模态 R@K（程序生成配对集）",
        seeds=",".join(str(s) for s in seeds),
        elapsed_s=time.perf_counter() - started,
        notes=[
            f"bus_dim={args.bus_dim}",
            f"embedding={args.model}（冻结前端）",
            "零联想对照与联想结果分开报；见预注册",
        ],
    ).render_block()

    args.out.mkdir(parents=True, exist_ok=True)
    for record in results:
        (args.out / f"seed{record['seed']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nmean R@1 = {summary['mean_r1']:.3f}  (chance {summary['mean_chance']:.3f})")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
