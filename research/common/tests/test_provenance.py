"""``research.common.provenance`` 的测试。

复现记录的价值全在**字段是不是真的从环境读出来的**。手抄一份模板当然也能过
"字段齐全"的测试，所以这里重点测两件事：脏工作区必须被标出来，以及模板要求的
每个字段都在输出里。
"""

from __future__ import annotations

import re

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


class TestDirtyCountingIsNotFooled:
    """`count_dirty_entries` 的三分类：假脏不算、真改动算、未跟踪算。

    **这组测试是一次真实事故的护栏**（2026-10-13）：手工把工作区文件的行尾从 CRLF
    归一成 LF 之后，`git status --porcelain` 会报 ` M`，而 `git diff` 为空、索引与
    工作区的 `git hash-object` 完全相同——racy timestamp 卡在 stat 缓存上。
    当时我一度以为文件真被改了。

    ⚠️ **但修复不能顺手改成 `--untracked-files=no`**：`docs/reproducibility.md` 把
    "Git 状态"定义成「`git status --porcelain` 的输出；非空说明有未提交改动」，
    **未跟踪文件也在内**——实验纪律（产物先落 data/ 再搬）靠的就是这一条。
    所以下面**三个方向**都要钉住；只钉"假脏不算"的测试会放过那次修复引入的回归。
    """

    @staticmethod
    def _fake(monkeypatch, *, porcelain, diff, cached):
        from research.common import provenance

        table = {
            ("status", "--porcelain"): porcelain,
            ("diff", "--name-only"): diff,
            ("diff", "--cached", "--name-only"): cached,
        }

        def fake_git_lines(*args):
            return table.get(args)

        monkeypatch.setattr(provenance, "_git_lines", fake_git_lines)
        return provenance.count_dirty_entries()

    def test_stat_only_change_is_not_dirty_2026_10_13(self, monkeypatch):
        """CRLF 归一造成的 stat 假脏不该判脏——那次事故的直接回归。

        形状就是当时观测到的：status 报 ` M`，但 diff 与 cached 都是空的。
        """
        n = self._fake(
            monkeypatch,
            porcelain=[" M README.en.md"],
            diff=[],
            cached=[],
        )
        assert n == 0, "stat 假脏被算成脏了——2026-10-13 那个坑回来了"

    def test_untracked_file_still_counts_as_dirty(self, monkeypatch):
        """⚠️ **未跟踪必须仍算脏**——明文语义，也是实验纪律依赖的那一条。

        改成 `--untracked-files=no` 会修好上面那条，却把这条一起砍掉：
        产物中途落进 `sweep_results/` 不再被判脏，"结论不可信"的警示就失效了。
        """
        n = self._fake(
            monkeypatch,
            porcelain=["?? sweep_results/foo.json"],
            diff=[],
            cached=[],
        )
        assert n == 1, "未跟踪文件不再算脏了——实验纪律赖以成立的那条语义没了"

    def test_tracked_content_change_counts(self, monkeypatch):
        """已跟踪文件的内容改动算脏（走 `git diff`，它会真比内容而不是比 stat）。"""
        n = self._fake(
            monkeypatch,
            porcelain=[" M a.py", " M b.py"],
            diff=["a.py"],
            cached=["b.py"],
        )
        assert n == 2

    def test_git_unavailable_falls_back_to_porcelain(self, monkeypatch):
        """diff 拿不到时保守回退到 porcelain 计数——宁可多报，不可漏报。"""
        n = self._fake(
            monkeypatch,
            porcelain=[" M a.py"],
            diff=None,
            cached=None,
        )
        assert n == 1

    def test_git_entirely_unavailable_is_zero(self, monkeypatch):
        """连 porcelain 都拿不到（例如从 tarball 解压的源码树）→ 0，与旧行为一致。"""
        n = self._fake(monkeypatch, porcelain=None, diff=None, cached=None)
        assert n == 0


class TestCollect:
    def test_reads_a_real_commit_hash(self):
        record = collect("测试实验", command="pytest", seeds="base=0", gpu="(测试)")
        assert re.fullmatch(r"[0-9a-f]{40}", record.commit), record.commit

    def test_fills_in_the_timestamp(self):
        record = collect("测试实验", command="pytest", gpu="(测试)")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} .+", record.collected_at)

    def test_dirty_count_matches_the_real_worktree(self):
        """对照真实工作区算一遍，而不是断言一个常数。

        这个仓库在开发中通常是不干净的，所以不能假设一定是 0；能做的是把记录里的
        数字**按定义**重算一遍——不一致意味着这个字段在骗人。

        ⚠️ **注意这里不是 `len(git status --porcelain)`**：从 2026-10-13 起，
        ``dirty_lines`` 的定义是"**真的**没提交的条目数"，会排除 stat 假脏
        （见 ``count_dirty_entries`` 的 docstring）。旧版本曾经断言等于 porcelain
        行数，那正好会把假脏算进去。
        """
        from research.common.provenance import _git_lines

        untracked = sum(
            1 for line in (_git_lines("status", "--porcelain") or []) if line.startswith("??")
        )
        content = _git_lines("diff", "--name-only") or []
        staged = _git_lines("diff", "--cached", "--name-only") or []
        expected = untracked + len(content) + len(staged)

        record = collect("测试实验", command="pytest", gpu="(测试)")
        assert record.dirty_lines == expected

    def test_explicit_gpu_string_is_not_overwritten(self):
        record = collect("测试实验", command="pytest", gpu="手工指定的硬件描述")
        assert record.gpu == "手工指定的硬件描述"

    def test_render_mentions_the_experiment_name(self):
        record = collect("e-prop/sMNIST", command="pytest", gpu="(测试)")
        assert "实验名称：e-prop/sMNIST" in record.render()
