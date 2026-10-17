"""跑前端验收口径：图像侧与文本侧的 caption 内容词线性探针。

用法::

    uv run python -m research.retrieval.run_probe

产出 ``sweep_results/retrieval_frontend_probe/summary.json``。
预注册见 ``sweep_results/retrieval_frontend_probe_prereg.md``。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.common.provenance import collect

from .flickr8k import make_items
from .frontends import resnet_features, text_features
from .probe import probe_words

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "data" / "flickr8k"
OUT = REPO_ROOT / "sweep_results" / "retrieval_frontend_probe"


def _summary(results) -> dict:
    values = [r.auc for r in results if np.isfinite(r.auc)]
    return {
        "n_words": len(values),
        "mean_auc": float(np.mean(values)),
        "median_auc": float(np.median(values)),
        "min_auc": float(np.min(values)),
        "max_auc": float(np.max(values)),
        "words": [
            {
                "word": r.word,
                "auc": r.auc,
                "pos_train": r.pos_train,
                "pos_test": r.pos_test,
                "lam": r.lam,
            }
            for r in sorted(results, key=lambda r: -r.auc)
        ],
    }


def main() -> int:
    print("加载 Flickr8k…")
    train_items = make_items(ROOT, split="train", decode_frames=False)
    test_items = make_items(ROOT, split="test", decode_frames=False)
    print(f"  train={len(train_items)}  test={len(test_items)}")

    print("读取冻结前端特征（缓存命中则不解图）…")
    image_train = resnet_features([i.path for i in train_items], cache=ROOT / "_resnet_train.npz")
    image_test = resnet_features([i.path for i in test_items], cache=ROOT / "_resnet_test.npz")
    text_train = text_features([i.caption for i in train_items], cache=ROOT / "_nomic_train.npz")
    text_test = text_features([i.caption for i in test_items], cache=ROOT / "_nomic_test.npz")

    print("图像侧探针…")
    image_results, image_dropped = probe_words(
        image_train, image_test, [i.caption for i in train_items], [i.caption for i in test_items]
    )
    print("文本侧探针（自查，应接近 1.0）…")
    text_results, text_dropped = probe_words(
        text_train, text_test, [i.caption for i in train_items], [i.caption for i in test_items]
    )

    summary = {
        "experiment": "front-end acceptance: caption content-word linear probes",
        "dataset": "Flickr8k official split",
        "image_frontend": "resnet18-imagenet1k-v1/224/avgpool",
        "text_frontend": "nomic-embed-text (whole sentence)",
        "image_side": _summary(image_results),
        "text_side": _summary(text_results),
        "dropped_image": image_dropped,
        "dropped_text": text_dropped,
        "repro": collect(
            "检索线 W4：前端线性探针（第十一轮）",
            seeds="0",
            notes=[
                "linear probe, ridge closed form, no sklearn",
                "词表：train 中 df>=50 的前 50 个内容词",
            ],
        ).render_block(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"\n图像侧 AUC 中位数 {summary['image_side']['median_auc']:.3f}"
        f"（均值 {summary['image_side']['mean_auc']:.3f}，{summary['image_side']['n_words']} 词）"
    )
    print(f"文本侧 AUC 中位数 {summary['text_side']['median_auc']:.3f}（自查用，应接近 1.0）")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
