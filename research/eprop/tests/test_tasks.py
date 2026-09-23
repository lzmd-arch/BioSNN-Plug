"""``research.eprop.tasks`` 的测试，重点是 SHD（旁证数据集）那条路径。

SHD 的加载此前**从未被运行过**，于是两个缺陷一直藏着：可变长的 ``times``/``units`` 在
``with h5py.File(...)`` 之外才被索引（文件已关，直接 ``RuntimeError``），以及分箱按毫秒
截断、只取每段音频的**开头 100 ms**（发布集的脉冲最远到 1.37 s）。这个文件把两条都钉住。
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from research.eprop.tasks import SHD_WINDOW_S, load_shd

#: 造一个小号的"SHD 形状"文件：可变长 times/units + uint16 labels，20 类。
N_UNITS = 700
N_CLASSES = 20


def write_fake_shd(path, samples):
    """``samples`` 是 ``[(times, units, label), ...]``。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    vlen_float = h5py.vlen_dtype(np.dtype("float32"))
    vlen_uint = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as handle:
        spikes = handle.create_group("spikes")
        times = spikes.create_dataset("times", (len(samples),), dtype=vlen_float)
        units = spikes.create_dataset("units", (len(samples),), dtype=vlen_uint)
        for index, (t, u, _) in enumerate(samples):
            times[index] = np.asarray(t, dtype="float32")
            units[index] = np.asarray(u, dtype="uint8")
        handle.create_dataset(
            "labels", data=np.asarray([s[2] for s in samples], dtype="uint16")
        )


class TestLoadShd:
    def test_reads_variable_length_spikes_after_the_file_is_closed(self, tmp_path):
        """回归：vlen 数据必须在 ``with`` 块内取出来。

        旧写法把 Dataset 对象带出 ``with`` 再索引，文件已经关了，抛
        ``RuntimeError: Unable to synchronously get dataspace``。这个测试在旧代码上失败。
        """
        samples = [
            ([0.0, 0.5], [3, 4], 0),
            ([1.0], [7], 1),
        ]
        write_fake_shd(tmp_path / "shd" / "train.h5", samples)

        task = load_shd(tmp_path, split="train", seed=0)
        assert task.inputs.shape == (100, 2, N_UNITS)
        assert task.labels.tolist() == [0, 1]
        assert task.inputs.dtype == torch.float32

    def test_spikes_land_in_the_bin_their_time_implies(self, tmp_path):
        """箱号 = ``t / SHD_WINDOW_S × max_time``，二值化后每格至多 1。"""
        write_fake_shd(tmp_path / "shd" / "train.h5", [([0.0, 0.7, 0.7], [1, 2, 2], 3)])
        task = load_shd(tmp_path, split="train", seed=0)

        assert task.inputs[0, 0, 1].item() == 1.0  # t=0 → 箱 0
        assert task.inputs[50, 0, 2].item() == 1.0  # t=0.7 → 0.7/1.4×100 = 箱 50
        assert task.inputs[:, 0, :].sum().item() == 2.0  # 同一格的第二颗脉冲被二值化吃掉

    def test_covers_the_whole_clip_not_just_the_first_100ms(self, tmp_path):
        """回归：脉冲在 1.3 s 处也要收进来。

        旧写法 ``times × 1000`` 把它当成毫秒、按 100 个箱截断，于是每段音频只剩开头
        100 ms——发布集里 1.37 s 以内的脉冲全被挤到最后那一格。
        """
        late = 1.3
        write_fake_shd(tmp_path / "shd" / "train.h5", [([late], [11], 5)])
        task = load_shd(tmp_path, split="train", seed=0)

        expected_bin = int(late / SHD_WINDOW_S * 100)
        assert expected_bin < 99  # 不是被截断挤到最后一格
        assert task.inputs[expected_bin, 0, 11].item() == 1.0
        assert task.inputs[:, 0, :].sum().item() == 1.0

    def test_labels_are_int64_indices(self, tmp_path):
        write_fake_shd(tmp_path / "shd" / "train.h5", [([0.1], [0], N_CLASSES - 1)])
        task = load_shd(tmp_path, split="train", seed=0)
        assert task.labels.dtype == torch.int64
        assert task.labels.tolist() == [N_CLASSES - 1]

    def test_missing_file_points_at_the_download_script(self, tmp_path):
        with pytest.raises(SystemExit, match="download_data.py shd"):
            load_shd(tmp_path, split="train", seed=0)

    def test_window_covers_the_longest_released_clip(self):
        """``SHD_WINDOW_S`` 要盖得住发布集里最长的一段（实测 1.369 s）。"""
        assert SHD_WINDOW_S >= 1.37
