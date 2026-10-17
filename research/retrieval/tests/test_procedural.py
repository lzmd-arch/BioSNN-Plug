"""程序生成配对集的测试：不下载、不联网、不用 Ollama。"""

from __future__ import annotations

import numpy as np

from research.retrieval.procedural import CANVAS, FRAMES, generate, split


class TestGenerate:
    def test_full_grid(self):
        items = generate()
        assert len(items) == 4 * 3 * 6 * 2 == 144

    def test_captions_are_unique(self):
        captions = [item.caption for item in generate()]
        assert len(set(captions)) == len(captions)

    def test_image_shape_and_range(self):
        item = generate()[0]
        assert item.image.shape == (FRAMES, CANVAS, CANVAS)
        assert item.image.min() >= 0.0 and item.image.max() <= 1.0

    def test_first_frame_is_blank_so_the_encoder_sees_an_onset(self):
        """DVS 编码的是帧差；首帧不空就没有 ON 事件，整条样本会退化成全零。"""
        for item in generate():
            assert item.image[0].max() == 0.0
            assert item.image[1].max() > 0.0

    def test_shapes_differ(self):
        """四种形状的像素掩码不能是同一个——否则"形状"这个属性根本没进图像。"""
        by_shape = {}
        for item in generate():
            by_shape.setdefault(item.shape, item.image[1])
        masks = [tuple((frame > 0).ravel()) for frame in by_shape.values()]
        assert len(set(masks)) == 4


class TestSplit:
    def test_disjoint_and_complete(self):
        items = generate()
        train, test = split(items, test_fraction=0.2, seed=0)
        assert len(train) + len(test) == len(items)
        assert set(map(id, train)).isdisjoint(set(map(id, test)))
        assert len(test) == round(len(items) * 0.2)

    def test_deterministic(self):
        items = generate()
        a = [item.caption for item in split(items, seed=3)[1]]
        b = [item.caption for item in split(items, seed=3)[1]]
        assert a == b

    def test_different_seeds_differ(self):
        items = generate()
        a = {item.caption for item in split(items, seed=0)[1]}
        b = {item.caption for item in split(items, seed=1)[1]}
        assert a != b

    def test_rejects_bad_fraction(self):
        import pytest

        with pytest.raises(ValueError, match="test_fraction"):
            split(generate(), test_fraction=0.0)


class TestImageIsNotConstant:
    def test_position_and_size_move_pixels(self):
        items = {(i.shape, i.shade, i.position, i.size): i for i in generate()}
        top_left = items[("square", "bright", "top left", "small")].image[1]
        bottom_right = items[("square", "bright", "bottom right", "small")].image[1]
        assert not np.allclose(top_left, bottom_right)
        small = items[("square", "bright", "top left", "small")].image[1]
        large = items[("square", "bright", "top left", "large")].image[1]
        assert (large > 0).sum() > (small > 0).sum()
