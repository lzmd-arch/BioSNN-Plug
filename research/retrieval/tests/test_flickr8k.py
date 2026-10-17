"""Flickr8k 加载器的测试：用临时目录造一个最小可解析的结构，不下载。"""

from __future__ import annotations

import numpy as np
import pytest

from research.retrieval.flickr8k import (
    DATASET_DIR,
    load_captions,
    load_split,
    make_items,
)
from research.retrieval.procedural import CANVAS, FRAMES


def _make_root(tmp_path, n_images=2, n_captions=5):
    root = tmp_path / "flickr8k"
    (root / DATASET_DIR).mkdir(parents=True)
    tokens, names = [], []
    for i in range(n_images):
        image_id = f"img{i}.jpg"
        names.append(image_id)
        for c in range(n_captions):
            tokens.append(f"{image_id}#{c}\ta caption number {c} for image {i}")
        _write_jpeg(root / DATASET_DIR / image_id, value=40 + 40 * i)
    (root / "Flickr8k.token.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    (root / "Flickr_8k.testImages.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (root / "Flickr_8k.trainImages.txt").write_text(names[0] + "\n", encoding="utf-8")
    (root / "Flickr_8k.devImages.txt").write_text(names[0] + "\n", encoding="utf-8")
    return root


def _write_jpeg(path, value: int) -> None:
    from PIL import Image

    array = np.full((32, 32), value, dtype=np.uint8)
    Image.fromarray(array).save(path, format="JPEG")


class TestParsing:
    def test_captions_grouped_by_image(self, tmp_path):
        root = _make_root(tmp_path, n_images=2, n_captions=5)
        captions = load_captions(root)
        assert set(captions) == {"img0.jpg", "img1.jpg"}
        assert len(captions["img0.jpg"]) == 5
        assert captions["img0.jpg"][0] == "a caption number 0 for image 0"

    def test_missing_token_file_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"Flickr8k\.token\.txt"):
            load_captions(tmp_path)

    def test_split_names(self, tmp_path):
        root = _make_root(tmp_path)
        assert load_split(root, "test") == ["img0.jpg", "img1.jpg"]
        with pytest.raises(ValueError, match="split"):
            load_split(root, "validation")


class TestMakeItems:
    def test_frames_have_the_dvs_shape(self, tmp_path):
        items = make_items(_make_root(tmp_path))
        assert len(items) == 2
        assert items[0].image.shape == (FRAMES, CANVAS, CANVAS)
        assert items[0].image[0].max() == 0.0
        assert items[0].image[1].max() > 0.0

    def test_caption_index_is_honoured(self, tmp_path):
        items = make_items(_make_root(tmp_path), caption_index=2)
        assert items[0].caption == "a caption number 2 for image 0"

    def test_out_of_range_caption_index_is_loud(self, tmp_path):
        with pytest.raises(IndexError, match="caption"):
            make_items(_make_root(tmp_path, n_captions=3), caption_index=4)

    def test_missing_images_are_skipped_not_silently(self, tmp_path, capsys):
        root = _make_root(tmp_path)
        (root / DATASET_DIR / "img1.jpg").unlink()
        items = make_items(root)
        assert len(items) == 1
        assert "缺图或缺 caption" in capsys.readouterr().out
