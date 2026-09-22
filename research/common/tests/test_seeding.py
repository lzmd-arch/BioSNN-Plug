"""``research.common.seeding`` 的测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.common.seeding import SeedBook, apply_seed, derive_seed

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestDeriveSeed:
    def test_is_deterministic(self):
        assert derive_seed(0, "权重初始化") == derive_seed(0, "权重初始化")

    def test_different_purposes_give_different_seeds(self):
        seeds = {derive_seed(0, p) for p in ("初始化", "打乱", "dropout", "环境重置")}
        assert len(seeds) == 4

    def test_different_bases_give_different_seeds(self):
        assert derive_seed(0, "初始化") != derive_seed(1, "初始化")

    def test_result_is_a_non_negative_int32(self):
        seed = derive_seed(2**31 - 1, "任意用途")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32

    @pytest.mark.parametrize("hash_seed", ["1", "2", "random"])
    def test_survives_a_process_boundary(self, hash_seed):
        """回归测试：种子派生不能用内置 ``hash()``。

        ``hash()`` 对字符串按进程加盐（``PYTHONHASHSEED``）。若用它派生，同一个
        用途名在不同进程里会得到不同种子——"固定种子即可复现"就只在单进程内成立，
        而实验恰恰总是跨进程做的（这一次跑和下一次跑）。骨架库
        （``SparseRandomProjection``）出于同样的理由用 ``zlib.crc32``，并有同名的
        回归测试盯着它。
        """
        code = (
            "from research.common.seeding import derive_seed;"
            "print(derive_seed(0, '权重初始化'), derive_seed(7, '数据打乱'))"
        )
        # 只覆盖 PYTHONHASHSEED，其余环境照抄——清空 PATH 在 Windows 上会让子进程
        # 找不到运行库，那是测试环境的问题，不是被测行为的问题。
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"{derive_seed(0, '权重初始化')} {derive_seed(7, '数据打乱')}"


class TestApplySeed:
    def test_makes_numpy_reproducible(self):
        apply_seed(123)
        first = np.random.random(8)
        apply_seed(123)
        np.testing.assert_array_equal(first, np.random.random(8))

    def test_different_seeds_differ(self):
        apply_seed(1)
        first = np.random.random(8)
        apply_seed(2)
        assert not np.array_equal(first, np.random.random(8))


class TestSeedBook:
    def test_derive_registers_and_returns(self):
        book = SeedBook(base=0)
        seed = book.derive("权重初始化")
        assert seed == derive_seed(0, "权重初始化")
        assert book.as_dict() == {"权重初始化": seed}

    def test_same_purpose_same_value_is_fine(self):
        """同用途重复登记同一个值是幂等的——多次调用不该报错。"""
        book = SeedBook(base=0)
        assert book.register("x", 5) == book.register("x", 5) == 5

    def test_same_purpose_different_value_is_an_error(self):
        """同名不同值说明有两处代码在抢同一件事的随机源，必须当场暴露。"""
        book = SeedBook(base=0)
        book.register("权重初始化", 1)
        with pytest.raises(ValueError, match="同名不同值|已被登记"):
            book.register("权重初始化", 2)

    def test_apply_sets_the_registered_seed(self):
        book = SeedBook(base=42)
        seed = book.apply("初始化")
        assert book.as_dict()["初始化"] == seed
        expected = np.random.RandomState(seed % (2**32)).random(4)
        np.testing.assert_array_equal(np.random.random(4), expected)

    def test_render_lists_every_purpose_and_the_base(self):
        book = SeedBook(base=3)
        book.derive("初始化")
        book.derive("打乱")
        text = book.render()
        assert "base=3" in text
        assert "初始化=" in text
        assert "打乱=" in text

    def test_render_without_entries_is_just_the_base(self):
        assert SeedBook(base=9).render() == "base=9"

    def test_as_dict_returns_a_copy(self):
        book = SeedBook(base=0)
        book.derive("x")
        book.as_dict()["注入"] = 1
        assert "注入" not in book.as_dict()
