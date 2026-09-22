"""``research.common.provenance`` 的测试。

复现记录的价值全在**字段是不是真的从环境读出来的**。手抄一份模板当然也能过
"字段齐全"的测试，所以这里重点测两件事：脏工作区必须被标出来，以及模板要求的
每个字段都在输出里。
"""

from __future__ import annotations

import re

import pytest

from research.common.provenance import DegradationLog, ReproRecord, collect, uv_lock_hash


class TestUvLockHash:
    def test_is_a_sha256_hex_digest(self):
        digest = uv_lock_hash()
        assert re.fullmatch(r"[0-9a-f]{64}", digest), digest

    def test_is_stable(self):
        assert uv_lock_hash() == uv_lock_hash()


class TestDegradationLog:
    def test_defaults_to_nothing_fired(self):
        log = DegradationLog()
        assert not log.any_fired
        assert log.render() == "未触发"

    def test_names_the_paths_that_fired(self):
        log = DegradationLog(scale_reduction=True, int8_traces=True)
        assert log.any_fired
        assert "规模降级" in log.render()
        assert "INT8 痕迹量化" in log.render()
        assert "分块训练" not in log.render()

    def test_carries_notes(self):
        log = DegradationLog(scale_reduction=True, notes=["5 万 → 1 万神经元"])
        assert "5 万 → 1 万神经元" in log.render()


class TestReproRecord:
    @staticmethod
    def _record(**overrides) -> ReproRecord:
        fields = {
            "experiment": "ib_hebbian/mnist",
            "command": "python -m research.ib_hebbian.train_mnist",
            "seeds": "base=0；初始化=1",
            "elapsed_s": 12.5,
            "gpu": "NVIDIA GeForce RTX 5060，8,151 MB，sm_120",
            "commit": "0" * 40,
            "collected_at": "2026-09-22 18:00:00",
        }
        fields.update(overrides)
        return ReproRecord(**fields)

    def test_render_covers_every_template_field(self):
        text = self._record().render()
        for field in (
            "实验名称：",
            "日期：",
            "Git commit：",
            "Git 状态：",
            "Python：",
            "操作系统 / 架构：",
            "硬件：",
            "随机种子：",
            "依赖快照：",
            "运行命令：",
            "耗时：",
        ):
            assert field in text, field

    def test_clean_tree_says_clean(self):
        record = self._record()
        assert not record.is_dirty
        assert "Git 状态：干净" in record.render()

    def test_dirty_tree_is_flagged_as_untrustworthy(self):
        """工作区不干净 = 复现结果不可信。这条必须显眼，不能只写成一行状态。"""
        record = self._record(dirty_lines=3)
        assert record.is_dirty
        text = record.render()
        assert "3 处未提交改动" in text
        assert "不可信" in text

    def test_gpu_fields_appear_only_when_measured(self):
        """CPU 上不该凭空多出"显存峰值"一栏——那会让人以为测过。"""
        assert "显存峰值" not in self._record().render()
        with_peak = self._record(peak_mb=1234.5)
        assert "1,234.5 MiB" in with_peak.render()

    def test_degradation_is_recorded_when_registered(self):
        record = self._record(degradation=DegradationLog(chunked_training=True))
        assert "§6.2 降级路径：分块训练" in record.render()

    def test_render_block_is_a_text_fence(self):
        block = self._record().render_block()
        assert block.startswith("```text\n")
        assert block.endswith("\n```")


class TestCollect:
    def test_reads_a_real_commit_hash(self):
        record = collect("测试实验", command="pytest", seeds="base=0", gpu="(测试)")
        assert re.fullmatch(r"[0-9a-f]{40}", record.commit), record.commit

    def test_fills_in_the_timestamp(self):
        record = collect("测试实验", command="pytest", gpu="(测试)")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} .+", record.collected_at)

    def test_dirty_count_matches_git_porcelain(self):
        """对照一次真实的 ``git status --porcelain``，而不是断言一个常数。

        这个仓库在开发中通常是不干净的，所以不能假设一定是 0；能做的是确认记录里
        的数字与命令输出**一致**——不一致意味着这个字段在骗人。
        """
        import subprocess

        from research.common.provenance import REPO_ROOT

        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

        expected = len(porcelain.splitlines()) if porcelain else 0
        record = collect("测试实验", command="pytest", gpu="(测试)")
        assert record.dirty_lines == expected

    def test_explicit_gpu_string_is_not_overwritten(self):
        record = collect("测试实验", command="pytest", gpu="手工指定的硬件描述")
        assert record.gpu == "手工指定的硬件描述"

    def test_render_mentions_the_experiment_name(self):
        record = collect("e-prop/sMNIST", command="pytest", gpu="(测试)")
        assert "实验名称：e-prop/sMNIST" in record.render()
