"""``research.eprop.sweep_seeds`` 的测试。

只测这个 runner **自己的**那点逻辑——种子解析、拒绝冒烟、汇总算术、以及「每个种子的两条判据都被记下来」。
训练本身由 `train_sequential.py` 的既有测试盯着，这里不重复。
"""

from __future__ import annotations

import json

import pytest

from research.eprop import sweep_seeds
from research.eprop.sweep_seeds import _summary, main, parse_seeds


class TestParseSeeds:
    def test_range(self):
        assert parse_seeds("0-4") == [0, 1, 2, 3, 4]

    def test_single(self):
        assert parse_seeds("7") == [7]

    def test_comma_list(self):
        assert parse_seeds("0,3,5") == [0, 3, 5]

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


class TestSummary:
    def test_reports_mean_min_max_and_spread_in_points(self):
        """极差按**百分点**报——与 W1/W3 的口径一致，免得读者把 0.0014 当成百分点。"""
        text = _summary([0.90, 0.92, 0.91], "测试准确率", "判据")
        assert "0.9100" in text  # 均值
        assert "0.9000" in text and "0.9200" in text
        assert "2.00 个百分点" in text

    def test_single_value_has_zero_spread(self):
        assert "0.00 个百分点" in _summary([0.8], "测试准确率", "判据")


class TestPerSeedRecording:
    def test_each_seed_lands_a_json_with_both_criteria(self, tmp_path, monkeypatch):
        """**两条判据都要落盘**：准确率与活跃神经元比例。

        W2 的验收判据有两条（§七：准确率显著高于随机 + 活跃比例 > 60%），只记一条会让
        另一条的种子方差看不见。
        """
        calls: list[int] = []

        def fake_train(args, device, history=None):
            calls.append(args.seed)
            if history is not None:
                history.extend(
                    {"epoch": float(e), "val_accuracy": 0.5 + 0.01 * e} for e in (1, 2, 3)
                )
            return 0.80 + 0.01 * args.seed, 0.65, 20

        monkeypatch.setattr(sweep_seeds, "train", fake_train)
        out_dir = tmp_path / "w2_seeds"
        assert main(["--seeds", "0-2", "--out-dir", str(out_dir)]) == 0

        assert calls == [0, 1, 2]
        for seed in range(3):
            record = json.loads((out_dir / f"seed{seed}.json").read_text(encoding="utf-8"))
            assert record["seed"] == seed
            assert record["test_accuracy"] == pytest.approx(0.80 + 0.01 * seed)
            assert record["active_fraction"] == pytest.approx(0.65)
            assert record["n_classes"] == 20  # 汇总里要拿它算随机基线
            # 逐 epoch 曲线是「epoch–准确率」那张图的数据源，必须落盘。
            assert [point["epoch"] for point in record["curve"]] == [1.0, 2.0, 3.0]

    def test_runner_asks_for_an_evaluation_every_epoch(self, tmp_path, monkeypatch):
        """曲线要够密——runner 默认给 `train` 传 `--eval-every 1`。

        这条不是洁癖：`epochs//5` 的旧节奏在 30 epoch 下只产出 **6 个点**，那样的曲线画出来
        看不出形状。评测只做前向、不消耗训练随机数，所以加密不会改变学习结果
        （那一点由 `test_eval_frequency_does_not_change_learning` 盯着）。
        """
        seen: list[int] = []

        def fake_train(args, device, history=None):
            seen.append(args.eval_every)
            return 0.5, 0.7, 10

        monkeypatch.setattr(sweep_seeds, "train", fake_train)
        assert main(["--seeds", "0", "--out-dir", str(tmp_path / "w2")]) == 0
        assert seen == [1]

    def test_an_explicit_eval_every_is_respected(self, tmp_path, monkeypatch):
        seen: list[int] = []

        def fake_train(args, device, history=None):
            seen.append(args.eval_every)
            return 0.5, 0.7, 10

        monkeypatch.setattr(sweep_seeds, "train", fake_train)
        assert main(["--seeds", "0", "--eval-every", "7", "--out-dir", str(tmp_path / "w2")]) == 0
        assert seen == [7]

    def test_running_one_seed_does_not_require_the_others(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sweep_seeds, "train", lambda args, device, history=None: (0.5, 0.7, 20))
        out_dir = tmp_path / "w2_seeds"
        assert main(["--seeds", "3", "--out-dir", str(out_dir)]) == 0
        assert [p.name for p in sorted(out_dir.glob("*.json"))] == ["seed3.json"]
