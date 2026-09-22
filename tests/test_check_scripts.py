"""检查脚本自身的回归测试。

**为什么值得给"检查脚本"写测试**：一个永远通过的检查比没有检查更糟——它给人一种
已经守住了的错觉。计划书 §12.3 把这些脚本定位成"提交即拦截"，那它们就必须被证明
真的抓得到东西。

所以这里全部是**负向测试**：故意造出问题，断言脚本失败。正向用例只用来确认还原后
是绿的。
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

DOC_BLOCKS = SCRIPTS / "check_doc_code_blocks.py"
REFERENCES = SCRIPTS / "check_references.py"
VERSIONS = SCRIPTS / "check_version_consistency.py"
LICENSES = SCRIPTS / "check_licenses.py"


def run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
        check=False,
    )


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


class TestDocCodeBlocks:
    """``scripts/check_doc_code_blocks.py``。"""

    def test_catches_the_undefined_attribute_it_was_written_for(self, tmp_path):
        """计划书 §12.3 点名的 ``self.threshold`` 那类 bug。

        语法完全合法（``ast.parse`` 与 ``compile`` 都放行），错在于运行时这个属性
        根本不存在——这就是本脚本必须真的执行代码块、而不能只做语法检查的原因。
        """
        doc = write(
            tmp_path / "bad.md",
            """
            # 坏文档

            ```python
            class Controller:
                def __init__(self):
                    self.other = 1

                def should_grow(self):
                    return self.threshold > 0.5

            Controller().should_grow()
            ```
            """,
        )

        syntax_only = run(DOC_BLOCKS, "--syntax-only", str(doc))
        assert syntax_only.returncode == 0, "语法检查本就不该抓到这个——这正是要证明的"

        executed = run(DOC_BLOCKS, str(doc))
        assert executed.returncode == 1
        assert "threshold" in (executed.stdout + executed.stderr)

    def test_reports_the_document_line_not_the_library_line(self, tmp_path):
        """报错要指向文档里该改的那一行，而不是库内部抛异常的那一行。"""
        doc = write(
            tmp_path / "bad.md",
            """
            ```python
            from biosnn_bus import get_plugin

            get_plugin("definitely-not-registered")
            ```
            """,
        )
        result = run(DOC_BLOCKS, str(doc))
        assert result.returncode == 1

        combined = result.stdout + result.stderr
        lines = doc.read_text(encoding="utf-8").splitlines()
        offending = next(
            index
            for index, line in enumerate(lines, start=1)
            if "get_plugin(" in line and "import" not in line
        )
        assert f"{doc}:{offending}" in combined, f"应指向文档第 {offending} 行：\n{combined}"
        assert "registry.py" not in combined, "不该把库内部的抛异常位置报给写文档的人"

    def test_no_run_escape_hatch_works(self, tmp_path):
        doc = write(
            tmp_path / "ok.md",
            """
            ```python no-run
            this_name_does_not_exist_anywhere()
            ```
            """,
        )
        assert run(DOC_BLOCKS, str(doc)).returncode == 0

    def test_escape_hatch_is_opt_in_per_block(self, tmp_path):
        """``no-run`` 只豁免它自己那一个块。"""
        doc = write(
            tmp_path / "mixed.md",
            """
            ```python no-run
            undefined_in_the_first_block()
            ```

            ```python
            undefined_in_the_second_block()
            ```
            """,
        )
        assert run(DOC_BLOCKS, str(doc)).returncode == 1

    def test_catches_syntax_errors(self, tmp_path):
        doc = write(
            tmp_path / "syntax.md",
            """
            ```python no-run
            def broken(:
            ```
            """,
        )
        assert run(DOC_BLOCKS, str(doc)).returncode == 1

    def test_blocks_share_a_namespace(self, tmp_path):
        """教程是一路写下去的：后面的块要用得到前面定义的变量。"""
        doc = write(
            tmp_path / "shared.md",
            """
            ```python
            shared_value = 41
            ```

            ```python
            assert shared_value + 1 == 42
            ```
            """,
        )
        assert run(DOC_BLOCKS, str(doc)).returncode == 0

    def test_non_python_fences_are_ignored(self, tmp_path):
        doc = write(
            tmp_path / "bash.md",
            """
            ```bash
            pip install definitely-not-a-real-package-xyz
            ```
            """,
        )
        assert run(DOC_BLOCKS, str(doc)).returncode == 0

    def test_indented_blocks_are_dedented(self, tmp_path):
        """代码块常常嵌在列表项下，整体带缩进——不去缩进会全部变成语法错误。"""
        doc = write(
            tmp_path / "indented.md",
            """
            1. 第一步：

               ```python
               nested = 1 + 1
               assert nested == 2
               ```
            """,
        )
        assert run(DOC_BLOCKS, str(doc)).returncode == 0


class TestReferences:
    """``scripts/check_references.py``。"""

    HEADER = """
        | 编号 | 文献 | URL | 核验状态 | 备注 |
        | :--- | :--- | :--- | :--- | :--- |
    """

    def _doc(self, tmp_path: Path, rows: str) -> Path:
        # 表头与数据行分别 dedent 再拼接。两者在测试源码里的缩进深度不同，
        # 一起 dedent 只会去掉公共的那部分，数据行会残留缩进——而表格行必须以
        # "|" 开头才会被解析，于是整张表被静默跳过。
        content = f"{textwrap.dedent(self.HEADER).strip()}\n{textwrap.dedent(rows).strip()}\n"
        path = tmp_path / "references.md"
        path.write_text(content, encoding="utf-8")
        return path

    def test_accepts_a_valid_list(self, tmp_path):
        doc = self._doc(
            tmp_path,
            """
            | 1 | A (2020). | https://example.org/a | verified | |
            | 2 | B (2021). | https://example.org/b | unverified | |
            """,
        )
        assert run(REFERENCES, str(doc)).returncode == 0

    def test_catches_numbering_gaps(self, tmp_path):
        doc = self._doc(
            tmp_path,
            """
            | 1 | A (2020). | https://example.org/a | verified | |
            | 3 | C (2022). | https://example.org/c | verified | |
            """,
        )
        result = run(REFERENCES, str(doc))
        assert result.returncode == 1
        assert "连续" in result.stderr

    def test_catches_malformed_urls(self, tmp_path):
        doc = self._doc(
            tmp_path,
            """
            | 1 | A (2020). | arxiv.org/abs/1234.5678 | verified | |
            """,
        )
        result = run(REFERENCES, str(doc))
        assert result.returncode == 1
        assert "URL 不合法" in result.stderr

    def test_catches_duplicate_urls(self, tmp_path):
        doc = self._doc(
            tmp_path,
            """
            | 1 | A (2020). | https://example.org/same | verified | |
            | 2 | B (2021). | https://example.org/same | verified | |
            """,
        )
        result = run(REFERENCES, str(doc))
        assert result.returncode == 1
        assert "重复" in result.stderr

    def test_catches_unknown_verification_status(self, tmp_path):
        doc = self._doc(
            tmp_path,
            """
            | 1 | A (2020). | https://example.org/a | probably-fine | |
            """,
        )
        result = run(REFERENCES, str(doc))
        assert result.returncode == 1
        assert "核验状态" in result.stderr

    def test_only_parses_the_references_table(self, tmp_path):
        """文档里还有别的表格；全文扫描会把它们也当成引用条目。"""
        doc = write(
            tmp_path / "references.md",
            """
            # 引用清单

            ## 核验状态

            | 状态 | 含义 |
            | :--- | :--- |
            | verified | 逐条核验过 |
            | unverified | 尚未核验 |

            ## 清单

            | 编号 | 文献 | URL | 核验状态 | 备注 |
            | :--- | :--- | :--- | :--- | :--- |
            | 1 | A (2020). | https://example.org/a | verified | |
            """,
        )
        assert run(REFERENCES, str(doc)).returncode == 0

    def test_fails_loudly_when_the_file_is_missing(self, tmp_path):
        result = run(REFERENCES, str(tmp_path / "nope.md"))
        assert result.returncode == 1


class TestVersionConsistency:
    """``scripts/check_version_consistency.py``。"""

    @staticmethod
    def _make_root(tmp_path: Path, *, pyproject: str, init: str, plan: str | None = None) -> Path:
        write(
            tmp_path / "packages" / "biosnn-bus" / "pyproject.toml",
            f'[project]\nname = "biosnn-bus"\nversion = "{pyproject}"\n',
        )
        write(
            tmp_path / "packages" / "biosnn-bus" / "src" / "biosnn_bus" / "__init__.py",
            f'__version__ = "{init}"\n',
        )
        if plan is not None:
            write(tmp_path / f"项目计划书_v{plan}.md", f"**版本 {plan} | 测试**\n")
        return tmp_path

    def test_accepts_consistent_versions(self, tmp_path):
        root = self._make_root(tmp_path, pyproject="0.1.0", init="0.1.0", plan="6.2")
        assert run(VERSIONS, "--root", str(root)).returncode == 0

    def test_catches_pyproject_init_mismatch(self, tmp_path):
        """装到的版本与运行时自报的版本对不上，是排查 bug 时最浪费时间的一类问题。"""
        root = self._make_root(tmp_path, pyproject="0.1.0", init="9.9.9", plan="6.2")
        result = run(VERSIONS, "--root", str(root))
        assert result.returncode == 1
        assert "不一致" in result.stderr

    def test_catches_plan_filename_content_mismatch(self, tmp_path):
        """计划书 §12.3 点名要拦的"文件名与内容版本号错位"。"""
        root = self._make_root(tmp_path, pyproject="0.1.0", init="0.1.0", plan="6.2")
        write(root / "项目计划书_v5.md", "**版本 6.2 | 测试**\n")
        result = run(VERSIONS, "--root", str(root))
        assert result.returncode == 1
        assert "错位" in result.stderr or "不一致" in result.stderr

    def test_coarse_filename_version_is_a_warning_not_an_error(self, tmp_path):
        """文件名只写到主版本号（``_v6`` vs 正文 ``6.2``）是提醒，不是拦截。

        这样 CI 不会因为一个历史遗留的文件名常年飘红，而真正的错位
        （``_v6.2`` vs 正文 ``6.3``）仍然会硬失败。
        """
        root = self._make_root(tmp_path, pyproject="0.1.0", init="0.1.0")
        write(root / "项目计划书_v6.md", "**版本 6.2 | 测试**\n")

        assert run(VERSIONS, "--root", str(root)).returncode == 0
        assert run(VERSIONS, "--root", str(root), "--strict").returncode == 1

    def test_reads_version_from_a_toml_with_other_sections(self, tmp_path):
        """回归测试：``version = "..."`` 在别的 section 里也出现过，别读错那一行。"""
        write(
            tmp_path / "packages" / "biosnn-bus" / "pyproject.toml",
            """
            [build-system]
            requires = ["hatchling"]

            [project]
            name = "biosnn-bus"
            version = "0.3.1"

            [tool.something]
            version = "999.0.0"
            """,
        )
        write(
            tmp_path / "packages" / "biosnn-bus" / "src" / "biosnn_bus" / "__init__.py",
            '__version__ = "0.3.1"\n',
        )
        assert run(VERSIONS, "--root", str(tmp_path)).returncode == 0


class TestLicenses:
    """``scripts/check_licenses.py``。

    这一组测试的存在理由很具体：第一版许可检查是 CI 里的一条内联 grep，
    它在**首次 CI 运行时就误报**了——`pip-licenses --with-license-file` 会把许可证
    正文灌进表格，而 PSF 许可证正文里有一句 "GPL-compatible licenses make it
    possible to combine Python with ..."，整行 grep 直接命中，于是 CI 红了一个根本
    没有强 copyleft 依赖的仓库。下面第一条就是这个 bug 的回归测试。
    """

    @staticmethod
    def _json(tmp_path: Path, packages: list[dict]) -> Path:
        path = tmp_path / "licenses.json"
        path.write_text(json.dumps(packages, ensure_ascii=False), encoding="utf-8")
        return path

    def test_psf_license_mentioning_gpl_in_its_text_is_not_a_violation(self, tmp_path):
        """回归测试：许可证**正文**里提到 GPL，不等于这个依赖是 GPL。"""
        path = self._json(
            tmp_path,
            [
                {
                    "Name": "typing_extensions",
                    "Version": "4.16.0",
                    "License": "PSF-2.0",
                    "LicenseText": (
                        "GPL-compatible licenses make it possible to combine Python with "
                        "other software that is released under the GNU General Public License."
                    ),
                }
            ],
        )
        assert run(LICENSES, str(path)).returncode == 0

    @pytest.mark.parametrize("license_name", ["AGPL-3.0", "GPL-3.0-only", "SSPL-1.0", "BUSL-1.1"])
    def test_rejects_strong_copyleft(self, tmp_path, license_name):
        path = self._json(tmp_path, [{"Name": "bad", "Version": "1.0", "License": license_name}])
        result = run(LICENSES, str(path))
        assert result.returncode == 1
        assert "强 copyleft" in result.stderr

    @pytest.mark.parametrize("license_name", ["LGPL-3.0", "MPL-2.0"])
    def test_weak_copyleft_is_reported_but_not_blocking(self, tmp_path, license_name):
        path = self._json(tmp_path, [{"Name": "weak", "Version": "1.0", "License": license_name}])
        assert run(LICENSES, str(path)).returncode == 0
        assert run(LICENSES, str(path), "--strict").returncode == 1

    def test_unknown_license_is_flagged_but_not_blocking(self, tmp_path):
        path = self._json(tmp_path, [{"Name": "mystery", "Version": "1.0", "License": "UNKNOWN"}])
        result = run(LICENSES, str(path))
        assert result.returncode == 0
        assert "未登记的未知许可证" in result.stdout
        assert run(LICENSES, str(path), "--strict").returncode == 1

    def test_strict_unknown_rejects_an_unregistered_unknown(self, tmp_path):
        """CI 用的就是这个开关：许可证不明的依赖必须被显式登记，否则失败。"""
        path = self._json(tmp_path, [{"Name": "mystery", "Version": "1.0", "License": "UNKNOWN"}])
        result = run(LICENSES, str(path), "--strict-unknown")
        assert result.returncode == 1
        assert "ACCEPTED_NON_STANDARD" in result.stderr

    def test_strict_unknown_tolerates_caution(self, tmp_path):
        """``--strict-unknown`` 只管 unknown。弱 copyleft 在 Python 生态里太常见，
        连带判死会把审计变成噪声，所以它仍然只是提醒。"""
        path = self._json(tmp_path, [{"Name": "weak", "Version": "1.0", "License": "MPL-2.0"}])
        assert run(LICENSES, str(path), "--strict-unknown").returncode == 0

    def test_registered_unknown_license_passes(self, tmp_path):
        """回归测试：SpikingJelly 的 License 字段是空的，前缀比对认不出它。

        它的许可证是启智开源许可证 1.0（OIOSL），且计划书 §12.1 一度以为它同为
        Apache-2.0（见 docs/adr/ADR-0008）。登记进白名单后，``--strict-unknown``
        必须放行，并且**把理由打印出来**——接受一件事就要看得见接受的代价。
        """
        path = self._json(
            tmp_path, [{"Name": "spikingjelly", "Version": "2.0.0rc1", "License": ""}]
        )
        result = run(LICENSES, str(path), "--strict-unknown")
        assert result.returncode == 0
        assert "已登记的非标准许可证" in result.stdout
        assert "ADR-0008" in result.stdout

    def test_allowlist_does_not_rescue_strong_copyleft(self, tmp_path):
        """白名单只救 unknown。上了白名单的包哪天换成强 copyleft，仍必须被拦住。"""
        path = self._json(
            tmp_path, [{"Name": "spikingjelly", "Version": "9.9", "License": "GPL-3.0"}]
        )
        assert run(LICENSES, str(path)).returncode == 1
        assert run(LICENSES, str(path), "--strict-unknown").returncode == 1

    def test_stale_allowlist_entry_is_reported(self, tmp_path):
        """登记了却不在依赖里的条目要报出来，免得白名单腐烂成一句无人在意的旧话。"""
        path = self._json(
            tmp_path, [{"Name": "numpy", "Version": "2.5.3", "License": "BSD-3-Clause"}]
        )
        result = run(LICENSES, str(path))
        assert result.returncode == 0
        assert "白名单已失效" in result.stdout

    def test_empty_dependency_list_fails(self, tmp_path):
        """空清单意味着这次审计什么都没查——那本身就该失败，不能算"通过"。"""
        path = self._json(tmp_path, [])
        result = run(LICENSES, str(path))
        assert result.returncode == 1
        assert "没有实际检查任何东西" in result.stderr

    def test_accepts_a_normal_apache_mit_tree(self, tmp_path):
        path = self._json(
            tmp_path,
            [
                {"Name": "numpy", "Version": "2.5.3", "License": "BSD-3-Clause"},
                {"Name": "ruff", "Version": "0.16.8", "License": "MIT"},
                {"Name": "biosnn-bus", "Version": "0.1.0", "License": "Apache-2.0"},
            ],
        )
        assert run(LICENSES, str(path)).returncode == 0


@pytest.mark.parametrize("script", [DOC_BLOCKS, REFERENCES, VERSIONS, LICENSES])
def test_scripts_are_runnable_as_cli(script):
    """三个脚本都必须能被 CI 与 pre-commit 直接调用（``--help`` 不报错）。"""
    assert script.exists()
    assert run(script, "--help").returncode == 0
