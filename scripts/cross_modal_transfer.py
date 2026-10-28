"""跨模态迁移报告：程序化配对集上的 CKA、线性探测与双向迁移。

不是验收实验（阶段四只要求“以可测量指标报告”）；使用确定性 token 嵌入，不依赖 Ollama。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from biosnn_bus.plugins.text_zscore import EmbeddingBackend  # noqa: E402

from research.retrieval.encode import build_bus, encode_items  # noqa: E402
from research.retrieval.procedural import generate, split  # noqa: E402

OUT = REPO / "sweep_results" / "cross_modal_transfer" / "summary.json"


class HashEmbedding(EmbeddingBackend):
    """确定性 token 嵌入：blake2b 派生 64 维，不联网。"""

    dim = 64

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float64) / 255.0 - 0.5


def _cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(x.T @ y) ** 2)
    denominator = float(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))
    return numerator / denominator if denominator else 0.0


def _one_hot(values: list[str]) -> np.ndarray:
    classes = sorted(set(values))
    index = {value: i for i, value in enumerate(classes)}
    out = np.zeros((len(values), len(classes)))
    out[np.arange(len(values)), [index[value] for value in values]] = 1.0
    return out


def _ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)


def _probe(x_train, y_train, x_test, y_test) -> float:
    coef = _ridge(x_train, y_train)
    predicted = np.argmax(x_test @ coef, axis=1)
    return float(np.mean(predicted == np.argmax(y_test, axis=1)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨模态迁移报告")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    items = generate()
    rows = []
    for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
        train_items, test_items = split(items, seed=seed)
        bus = build_bus(bus_dim=128, seed=seed, backend=HashEmbedding())
        train = encode_items(bus, train_items)
        test = encode_items(bus, test_items)
        attrs_train = [
            _one_hot([getattr(item, name) for item in train_items])
            for name in ("shape", "shade", "position", "size")
        ]
        attrs_test = [
            _one_hot([getattr(item, name) for item in test_items])
            for name in ("shape", "shade", "position", "size")
        ]
        row = {
            "seed": seed,
            "cka_text_image": _cka(train.text_bus, train.image_bus),
            "text_probe": float(
                np.mean(
                    [
                        _probe(train.text_bus, y, test.text_bus, yt)
                        for y, yt in zip(attrs_train, attrs_test, strict=True)
                    ]
                )
            ),
            "image_probe": float(
                np.mean(
                    [
                        _probe(train.image_bus, y, test.image_bus, yt)
                        for y, yt in zip(attrs_train, attrs_test, strict=True)
                    ]
                )
            ),
        }
        # 双向迁移：一侧线性映射到另一侧，再用目标侧分类器
        text_to_image = _ridge(train.text_bus, train.image_bus)
        image_to_text = _ridge(train.image_bus, train.text_bus)
        mapped_text = test.text_bus @ text_to_image
        mapped_image = test.image_bus @ image_to_text
        row["text_to_image_transfer"] = float(
            np.mean(
                [
                    _probe(train.image_bus, y, mapped_text, yt)
                    for y, yt in zip(attrs_train, attrs_test, strict=True)
                ]
            )
        )
        row["image_to_text_transfer"] = float(
            np.mean(
                [
                    _probe(train.text_bus, y, mapped_image, yt)
                    for y, yt in zip(attrs_train, attrs_test, strict=True)
                ]
            )
        )
        rows.append(row)
    summary = {
        "kind": "report only; deterministic HashEmbedding; procedural paired set",
        "rows": rows,
        "means": {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "seed"
        },
        "boundary": "synthetic attributes, no Flickr8k transfer claim",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary["means"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
