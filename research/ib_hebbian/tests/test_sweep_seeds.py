"""``research.ib_hebbian.sweep_seeds`` 的测试。

只测这个 runner **自己的**那点逻辑（种子解析、拒绝冒烟、汇总算术）——训练本身由
`train_mnist.py` 的既有测试盯着，这里不重复。
"""

from __future__ import annotations

import pytest

from research.ib_hebbian.sweep_seeds import main, parse_seeds


class TestParseSeeds:
    def test_range(self):
        assert parse_seeds("0-4") == [0, 1, 2, 3, 4]

    def test_single(self):
        assert parse_seeds("7") == [7]

    def test_comma_list(self):
        assert parse_seeds("0,3,5") == [0, 3, 5]

    def test_mixed(self):
        assert parse_seeds("0-1,7") == [0, 1, 7]

    def test_whitespace_is_tolerated(self):
        assert parse_seeds(" 0-1 , 4 ") == [0, 1, 4]

    def test_reversed_range_is_an_error(self):
        with pytest.raises(ValueError, match="区间反了"):
            parse_seeds("4-1")

    def test_duplicates_are_an_error(self):
        """重复的种子会让"均值"偷偷给某个种子加权，必须拦下来。"""
        with pytest.raises(ValueError, match="重复"):
            parse_seeds("0-2,2")

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            parse_seeds("")


class TestSmokeIsRejected:
    def test_smoke_is_refused(self):
        """冒烟规模的精度没有意义，不能拿它做多种子统计。"""
        with pytest.raises(SystemExit, match="冒烟"):
            main(["--seeds", "0", "--smoke"])


class TestSeedSwitching:
    def test_each_seed_builds_a_different_seed_book(self):
        """种子的**用途**不变、**基值**变——否则「换种子」只是换了个标签。

        这里只查口径：同一份配置下 `SeedBook` 派生出的全局种子随 `--seed` 改变，
        与 `train()` 内部 `SeedBook(base=args.seed)` 的用法一致。
        """
        from research.common.seeding import SeedBook

        first = SeedBook(base=0).derive("全局种子")
        second = SeedBook(base=1).derive("全局种子")
        assert first != second
