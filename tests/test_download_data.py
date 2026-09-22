"""``scripts/download_data.py`` 的测试（不联网）。

分两部分：

* **IDX 解析**——可以直接喂合成字节，覆盖合法与各类畸形输入；
* **下载与校验流程**——用预先摆好、哈希已对的本地文件走通"已就绪"这条路，
  以及用哈希对不上的文件验证它**会在联网之前**失败。

刻意不打桩 ``urlopen``：那会把测试变成"我的桩与我的实现一致"的同义反复。校验
失败的路径本来就在发起下载之前返回，所以不需要网络也能测到。
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DATA = REPO_ROOT / "scripts" / "download_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_data", DOWNLOAD_DATA)
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules 再 exec：模块里有 dataclass，而 dataclasses 处理
    # `from __future__ import annotations` 的字符串注解时会回查
    # sys.modules[cls.__module__]，查不到就抛 AttributeError。
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dd():
    return _load_module()


def _idx(dtype_code: int, shape: tuple[int, ...], payload: bytes) -> bytes:
    """按 IDX 格式拼一段字节：magic 4 字节 + 每维 4 字节 + 数据。"""
    magic = bytes([0, 0, dtype_code, len(shape)])
    dims = b"".join(int(d).to_bytes(4, "big") for d in shape)
    return magic + dims + payload


class TestReadIdx:
    def test_parses_images(self, dd):
        payload = bytes(range(12))
        raw = _idx(0x08, (2, 2, 3), payload)
        array = dd.read_idx(raw)
        assert array.shape == (2, 2, 3)
        assert array.dtype == np.uint8
        np.testing.assert_array_equal(array.reshape(-1), np.arange(12, dtype=np.uint8))

    def test_parses_labels(self, dd):
        raw = _idx(0x08, (5,), bytes([3, 1, 4, 1, 5]))
        array = dd.read_idx(raw)
        assert array.shape == (5,)
        assert array.tolist() == [3, 1, 4, 1, 5]

    def test_parses_big_endian_int32(self, dd):
        """维度数与元素宽度都要按大端读——写反了在小数据上往往看不出来。"""
        raw = _idx(0x0C, (2,), (258).to_bytes(4, "big") + (1).to_bytes(4, "big"))
        assert dd.read_idx(raw).tolist() == [258, 1]

    def test_rejects_short_data(self, dd):
        with pytest.raises(ValueError, match="太短"):
            dd.read_idx(b"\x00\x00")

    def test_rejects_bad_magic(self, dd):
        with pytest.raises(ValueError, match="magic"):
            dd.read_idx(bytes([1, 2, 0x08, 1]) + (1).to_bytes(4, "big") + b"\x00")

    def test_rejects_unsupported_dtype_code(self, dd):
        with pytest.raises(ValueError, match="类型码"):
            dd.read_idx(_idx(0x0A, (1,), b"\x00"))

    def test_rejects_zero_dimensions(self, dd):
        with pytest.raises(ValueError, match="维度数"):
            dd.read_idx(_idx(0x08, (), b""))

    def test_rejects_truncated_payload(self, dd):
        """形状说 4 个元素却只给了 3 字节——截断的下载必须当场暴露。"""
        with pytest.raises(ValueError, match="长度与形状不符"):
            dd.read_idx(_idx(0x08, (4,), b"\x00\x00\x00"))

    def test_rejects_trailing_garbage(self, dd):
        with pytest.raises(ValueError, match="长度与形状不符"):
            dd.read_idx(_idx(0x08, (2,), b"\x00\x00\x00"))


class TestFetch:
    """校验流程。用本地文件走，不联网。"""

    @staticmethod
    def _spec(dd, md5: str, name: str = "toy"):
        return dd.DatasetSpec(
            name=name,
            mirrors=("https://example.invalid/",),
            files=(dd.DataFile("toy-images-idx3-ubyte.gz", md5),),
            license_note="测试用",
            parse={},
        )

    def test_accepts_an_already_verified_file(self, dd, tmp_path):
        payload = b"payload"
        md5 = hashlib.md5(payload).hexdigest()
        (tmp_path / "toy").mkdir()
        (tmp_path / "toy" / "toy-images-idx3-ubyte.gz").write_bytes(payload)

        spec = self._spec(dd, md5)
        result = dd.fetch(spec, spec.files[0], tmp_path, verify_only=True)
        assert result.read_bytes() == payload

    def test_corrupted_file_fails_before_any_download(self, dd, tmp_path):
        """哈希对不上时必须当场失败，而不是拿去用、也不是悄悄重下。"""
        (tmp_path / "toy").mkdir()
        (tmp_path / "toy" / "toy-images-idx3-ubyte.gz").write_bytes(b"corrupted")

        spec = self._spec(dd, "0" * 32)
        with pytest.raises(SystemExit, match="verify-only"):
            dd.fetch(spec, spec.files[0], tmp_path, verify_only=True)

    def test_missing_file_fails_under_verify_only(self, dd, tmp_path):
        spec = self._spec(dd, "0" * 32)
        with pytest.raises(SystemExit, match="verify-only"):
            dd.fetch(spec, spec.files[0], tmp_path, verify_only=True)


class TestCli:
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(DOWNLOAD_DATA), *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
            check=False,
        )

    def test_list_names_the_registered_datasets(self):
        result = self._run("--list")
        assert result.returncode == 0
        assert "mnist" in result.stdout

    def test_unknown_dataset_is_rejected(self):
        result = self._run("definitely-not-a-dataset")
        assert result.returncode == 2
        assert "未登记的数据集" in result.stderr


class TestRegistry:
    def test_mnist_hashes_are_wellformed(self, dd):
        """四个文件、四个 32 位十六进制 md5。写错一位就等于校验形同虚设。"""
        assert len(dd.DATASETS["mnist"].files) == 4
        for data_file in dd.DATASETS["mnist"].files:
            assert len(data_file.md5) == 32
            int(data_file.md5, 16)  # 非十六进制会抛 ValueError

    def test_every_registered_file_is_parsed_to_an_npy(self, dd):
        """登记了文件却没有对应的预处理产物，下游就拿不到 .npy。"""
        for spec in dd.DATASETS.values():
            for data_file in spec.files:
                assert data_file.filename in spec.parse, data_file.filename

    def test_preprocess_roundtrips_through_gzip(self, dd, tmp_path):
        payload = _idx(0x08, (2, 2), bytes([1, 2, 3, 4]))
        raw = tmp_path / "x-images-idx3-ubyte.gz"
        with gzip.open(raw, "wb") as handle:
            handle.write(payload)

        spec = dd.DatasetSpec(name="toy", mirrors=(), files=(), license_note="", parse={})
        out = tmp_path / "x.npy"
        dd.preprocess(spec, raw, out)
        np.testing.assert_array_equal(np.load(out), np.array([[1, 2], [3, 4]], dtype=np.uint8))
