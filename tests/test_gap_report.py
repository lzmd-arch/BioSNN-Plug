"""``scripts/gap_report.py`` 的回归测试。

**为什么要有负向测试**：这个脚本的产出是一份 artifact，而 artifact 没人会去核对——
它"永远生成成功"和"真的抓到了漂移"从外面看一模一样。所以这里的主体是**故意造出漂移，
断言脚本失败**：文档改了数字、三语各指一次不同的运行、日志被截断、汇总行被手改。

正向用例只用来确认"一份自洽的记录 + 三语一致的文档"能生成报告。
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GAP_REPORT = REPO_ROOT / "scripts" / "gap_report.py"

DOCS = ("README.md", "README.en.md", "README.ja.md")

#: 一次自洽的运行记录。`{...}` 留给各个用例去改。
RUN_LOG = """\
[e-prop] epoch   1  train_ce 2.1786  (16s)
[e-prop] 测试准确率 {eprop}
[BPTT] epoch   1  train_ce 2.1014  (13s)
[BPTT] 测试准确率 {bptt}

量化差距报告：e-prop {eprop} vs BPTT {bptt}（差距 {gap}）
配置：n_rec=256, epochs=10, n_train=20000, batch=64, seed=0

```text
实验名称：eprop/bptt-gap
日期：2026-09-23 20:39:11 中国标准时间
Git commit：632f01b88b5755256c0075cf0bc8e8c73d867da9
耗时：306.0 s
硬件：NVIDIA GeForce RTX 5060，8,123 MB，sm_120
```
"""

#: 三语各自的那张差距表；`{...}` 是三个数字。
DOC_BODY = """\
# 文档

| 臂 | 学习规则 | 测试准确率 |
| :--- | :--- | ---: |
| e-prop | 局部资格痕迹，无交叉层梯度 | **{eprop}** |
| BPTT | 代理梯度穿脉冲 | **{bptt}** |
| | **{label}** | **{gap}** |
"""


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GAP_REPORT), "--root", str(root), "--out-dir", str(root), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )


def write_run(root: Path, name: str, *, eprop: str, bptt: str, gap: str) -> Path:
    path = root / "sweep_results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RUN_LOG.format(eprop=eprop, bptt=bptt, gap=gap), encoding="utf-8", newline="\n")
    return path


def write_docs(root: Path, *, eprop: str, bptt: str, gap: str, only: str | None = None) -> None:
    """写三语文档。``only`` 指定时只写那一份（用来测"文档不见了"）。"""
    labels = {"README.md": "差距", "README.en.md": "Gap", "README.ja.md": "ギャップ"}
    for name in DOCS:
        if only is not None and name != only:
            continue
        path = root / "research" / "eprop" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            textwrap.dedent(DOC_BODY).format(eprop=eprop, bptt=bptt, gap=gap, label=labels[name]),
            encoding="utf-8",
            newline="\n",
        )


def coherent_repo(root: Path, *, eprop="0.5274", bptt="0.7602", gap="+0.2328") -> Path:
    """一份自洽的仓库：一次记录 + 三语一致的文档。"""
    write_run(root, "w2_bptt.log", eprop=eprop, bptt=bptt, gap=gap)
    write_docs(root, eprop=eprop, bptt=bptt, gap=gap)
    return root


def combined(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


class TestHappyPath:
    def test_generates_both_artifacts(self, tmp_path):
        coherent_repo(tmp_path)
        result = run(tmp_path)
        assert result.returncode == 0, combined(result)

        report = json.loads((tmp_path / "gap-report.json").read_text(encoding="utf-8"))
        assert report["headline"]["e_prop"] == 0.5274
        assert report["headline"]["gap"] == 0.2328
        assert report["headline"]["source"] == "sweep_results/w2_bptt.log"
        assert len(report["docs"]) == 3

        markdown = (tmp_path / "gap-report.md").read_text(encoding="utf-8")
        assert "+0.2328" in markdown
        assert "2026-09-23 20:39:11 中国标准时间" in markdown

    def test_run_without_provenance_fields_still_counts(self, tmp_path):
        """``w2_bptt_first.log`` 那块是手工写的，没有日期与耗时——缺字段不是错误。"""
        coherent_repo(tmp_path)
        (tmp_path / "sweep_results" / "w2_bptt_first.log").write_text(
            RUN_LOG.format(eprop="0.5274", bptt="0.7602", gap="+0.2328").split("```text")[0],
            encoding="utf-8",
            newline="\n",
        )
        result = run(tmp_path)
        assert result.returncode == 0, combined(result)

        report = json.loads((tmp_path / "gap-report.json").read_text(encoding="utf-8"))
        assert len(report["runs"]) == 2
        # 有日期的排在前面：没有日期的那次不能冒充最新。
        assert report["runs"][0]["date"] == "2026-09-23 20:39:11 中国标准时间"
        assert report["runs"][1]["date"] == ""
        assert report["runs"][1]["elapsed"] == ""


class TestDriftIsCaught:
    """这一节才是这个脚本存在的理由。"""

    def test_doc_quoting_an_unrecorded_number_fails(self, tmp_path):
        """把文档里的差距改成记录里没有的数——最典型的一种漂移。"""
        coherent_repo(tmp_path)
        # 只改英文那一份，中文与日文保持 coherent_repo 写的原样。
        write_docs(tmp_path, eprop="0.5274", bptt="0.7602", gap="+0.2500", only="README.en.md")

        result = run(tmp_path)
        assert result.returncode == 1
        assert "README.en.md" in combined(result)
        assert "找不到对应的一次运行" in combined(result)
        assert not (tmp_path / "gap-report.md").exists(), "漂移时宁可不产出报告"

    def test_docs_pointing_at_different_runs_fail(self, tmp_path):
        """三语各指一次不同的运行——同一次比较在不同语言里成了两个数。"""
        write_run(tmp_path, "a.log", eprop="0.5274", bptt="0.7602", gap="+0.2328")
        write_run(tmp_path, "b.log", eprop="0.6000", bptt="0.7602", gap="+0.1602")
        write_docs(tmp_path, eprop="0.5274", bptt="0.7602", gap="+0.2328")
        # 日文改指第二次运行。
        (tmp_path / "research" / "eprop" / "README.ja.md").write_text(
            textwrap.dedent(DOC_BODY).format(
                eprop="0.6000", bptt="0.7602", gap="+0.1602", label="ギャップ"
            ),
            encoding="utf-8",
            newline="\n",
        )

        result = run(tmp_path)
        assert result.returncode == 1
        assert "不是同一次运行" in combined(result)

    def test_one_doc_disagreeing_with_itself_fails(self, tmp_path):
        """同一份文档里同一个标签两个值——多半只改了一半。"""
        coherent_repo(tmp_path)
        path = tmp_path / "research" / "eprop" / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n| | **差距** | **+0.9999** |\n",
            encoding="utf-8",
            newline="\n",
        )
        result = run(tmp_path)
        assert result.returncode == 1
        assert "多个不同的值" in combined(result)

    def test_hand_edited_summary_line_fails(self, tmp_path):
        """汇总行被手改成与逐臂打印不符——只看汇总行就会把这个数当真。"""
        coherent_repo(tmp_path)
        path = tmp_path / "sweep_results" / "w2_bptt.log"
        path.write_text(
            path.read_text(encoding="utf-8").replace("差距 +0.2328", "差距 +0.1000"),
            encoding="utf-8",
            newline="\n",
        )
        result = run(tmp_path)
        assert result.returncode == 1
        assert "对不上" in combined(result)

    def test_gap_not_equal_to_the_difference_fails(self, tmp_path):
        """三个数都自洽、但差距那一项算错了。"""
        coherent_repo(tmp_path, gap="+0.2000")
        result = run(tmp_path)
        assert result.returncode == 1
        assert "BPTT − e-prop" in combined(result)

    def test_truncated_run_is_not_silently_ignored(self, tmp_path):
        """半份输出：有逐臂结果、没有汇总行。不能当作"没有过这次运行"。"""
        coherent_repo(tmp_path)
        (tmp_path / "sweep_results" / "truncated.log").write_text(
            "[e-prop] 测试准确率 0.5274\n", encoding="utf-8", newline="\n"
        )
        result = run(tmp_path)
        assert result.returncode == 1
        assert "被截断" in combined(result)

    def test_record_dir_without_any_gap_run_fails(self, tmp_path):
        coherent_repo(tmp_path)
        for path in (tmp_path / "sweep_results").glob("*.log"):
            path.unlink()
        result = run(tmp_path)
        assert result.returncode == 1
        assert "报告没有来源" in combined(result)

    def test_missing_translation_fails(self, tmp_path):
        coherent_repo(tmp_path)
        (tmp_path / "research" / "eprop" / "README.ja.md").unlink()
        result = run(tmp_path)
        assert result.returncode == 1
        assert "README.ja.md" in combined(result)


class TestRealRepository:
    """跑一次真仓库：它现在的状态必须是绿的，否则这个 job 一接上就红。"""

    def test_current_repository_passes(self, tmp_path):
        result = run(REPO_ROOT, "--out-dir", str(tmp_path))
        assert result.returncode == 0, combined(result)

        report = json.loads((tmp_path / "gap-report.json").read_text(encoding="utf-8"))
        assert report["headline"]["e_prop"] == 0.5274
        assert report["headline"]["bptt"] == 0.7602
        assert report["headline"]["gap"] == 0.2328
        assert len(report["docs"]) == 3
        assert len(report["runs"]) >= 2
