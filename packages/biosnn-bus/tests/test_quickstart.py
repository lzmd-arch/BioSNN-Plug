"""冒烟测试：仓库里的示例必须真的能跑。

计划书 §12.3 把"最小复现脚本"列为 CI 检查项，要拦截的正是"论文说能跑但仓库跑不通"
这一类问题。示例脚本是这个项目对外的第一印象，它坏了没人会往下看——所以它进 CI。

同时校验 Colab notebook 与脚本同步：notebook 由 ``scripts/build_notebook.py`` 从
脚本生成，脚本是唯一真相源，两者不可能各自漂移。

**三语版本**：``examples/quickstart_register_plugin.{en,ja}.py`` 与中文版代码完全相同，
只有 markdown 说明块不同。三份都要跑——不是为了重复验证同一段代码，而是为了拦住
"翻译时手滑改了代码"。因此本文件的断言**必须是语言无关的**：断言示例的输出文字
（例如中文的 ``"总线输出:"``）会让译文脚本必然失败。
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples"
BUILDER = REPO_ROOT / "scripts" / "build_notebook.py"

#: 三语示例脚本。``*.en.py`` / ``*.ja.py`` 与中文版代码相同、说明文字不同。
QUICKSTART_SCRIPTS = sorted(EXAMPLES.glob("quickstart_register_plugin*.py"))

#: 示例末尾会画脉冲栅格图。在带图形界面的机器上 matplotlib 默认选 tkagg，
#: ``plt.show()`` 会开一个窗口并**阻塞到用户关掉它为止**——对交互式使用是正常的，
#: 对冒烟测试就是永久挂起。强制无头后端，让 show() 变成空操作。
HEADLESS_ENV = {**os.environ, "MPLBACKEND": "Agg"}

#: 子进程超时。宁可失败也不要挂住 CI。
TIMEOUT_SECONDS = 300


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=HEADLESS_ENV,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:  # pragma: no cover - 只在真的卡住时触发
        pytest.fail(f"{args[0]} 运行超过 {TIMEOUT_SECONDS}s 未结束，疑似阻塞在交互式绘图上。")


@pytest.fixture(scope="module", params=QUICKSTART_SCRIPTS, ids=lambda p: p.name)
def quickstart(request) -> subprocess.CompletedProcess[str]:
    """跑一份示例脚本（三语各跑一次）。"""
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("示例需要 matplotlib；请安装 biosnn-bus[examples]")
    result = run_script(str(request.param))
    assert result.returncode == 0, (
        f"{request.param.name} 退出码 {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result


class TestQuickstartScripts:
    """三份脚本的代码相同，所以断言同一组**语言无关**的输出特征。

    刻意不断言输出的自然语言部分（``"总线输出:"`` 之类）——那会让译文脚本必然失败，
    而译文脚本本来就该用译文的文字输出。
    """

    def test_registers_both_modalities(self, quickstart):
        assert "sine_wave" in quickstart.stdout
        assert "image_diff" in quickstart.stdout

    def test_routes_the_two_channels(self, quickstart):
        """示例的核心卖点：时间通道与语义通道各走各的。"""
        assert "'temporal': ['sine_wave']" in quickstart.stdout
        assert "'semantic': ['image_diff']" in quickstart.stdout

    def test_produces_bus_output(self, quickstart):
        # SpikeTrain 的 repr 是语言无关的：形状与 dtype 不随译文变化
        assert "SpikeTrain(shape=(24, 128)" in quickstart.stdout

    def test_decodes_back_to_modality_space(self, quickstart):
        """解码误差是数字，不随语言变化。"""
        assert re.search(r"0\.0161", quickstart.stdout), (
            f"未在输出里找到解码误差 0.0161：\n{quickstart.stdout}"
        )

    def test_output_is_not_silently_empty(self, quickstart):
        """防止上面的断言因为输出为空而"通过"。"""
        assert len(quickstart.stdout.strip()) > 100


class TestNotebookSync:
    @pytest.mark.parametrize("script", QUICKSTART_SCRIPTS, ids=lambda p: p.name)
    def test_notebook_exists(self, script):
        notebook = script.with_suffix(".ipynb")
        assert notebook.exists(), f"缺少 {notebook.name}。请运行：python scripts/build_notebook.py"

    def test_all_notebooks_match_their_scripts(self):
        """回归测试：notebook 与脚本漂移时，CI 必须失败。

        CI 跑的是脚本，用户点开的是 notebook。两者一旦不同步，就会出现"CI 全绿但
        用户跑不通"的局面——这正是计划书 §12.3 要拦的那类问题。
        """
        result = run_script(str(BUILDER), "--check")
        assert result.returncode == 0, (
            f"notebook 与脚本不同步，请运行 python scripts/build_notebook.py\n"
            f"{result.stdout}{result.stderr}"
        )

    @pytest.mark.parametrize("script", QUICKSTART_SCRIPTS, ids=lambda p: p.name)
    def test_notebook_is_valid_nbformat(self, script):
        nbformat = pytest.importorskip("nbformat")
        notebook = nbformat.read(str(script.with_suffix(".ipynb")), as_version=4)
        nbformat.validate(notebook)
        assert any(cell.cell_type == "code" for cell in notebook.cells)
        assert any(cell.cell_type == "markdown" for cell in notebook.cells)


class TestNotebookLanguageParity:
    """三份 notebook 的**代码必须语义相同**，只有说明文字不同。

    比对复用 ``scripts/check_translations.py`` 的 ``normalise_code``：它剥掉注释与
    字符串字面量后再比 AST。直接逐字比对代码单元是错的——译文里的注释与
    ``print`` 输出标签本来就该翻译，逐字比会把这些正常翻译判成错误。
    代码的**语义**（标识符、数值、调用结构）仍被严格比对。
    """

    @staticmethod
    def _normalise(source: str) -> str:
        spec = importlib.util.spec_from_file_location(
            "check_translations_for_notebook_tests",
            REPO_ROOT / "scripts" / "check_translations.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.normalise_code("python", source)

    def test_code_cells_are_semantically_identical_across_languages(self):
        nbformat = pytest.importorskip("nbformat")
        if len(QUICKSTART_SCRIPTS) < 2:
            pytest.skip("只有一个语言版本，无需比对")

        reference: list[str] | None = None
        reference_name = ""
        for script in QUICKSTART_SCRIPTS:
            notebook = nbformat.read(str(script.with_suffix(".ipynb")), as_version=4)
            code = [
                self._normalise(cell.source) for cell in notebook.cells if cell.cell_type == "code"
            ]
            if reference is None:
                reference, reference_name = code, script.name
                continue
            assert code == reference, (
                f"{script.name} 的代码与 {reference_name} 语义不一致——"
                f"翻译只应改说明文字与注释，不应改代码。"
            )

    def test_markdown_cells_are_translated(self):
        """反向检查：说明文字**应该**不同，否则说明译文根本没翻译 markdown 单元。"""
        nbformat = pytest.importorskip("nbformat")
        if len(QUICKSTART_SCRIPTS) < 2:
            pytest.skip("只有一个语言版本，无需比对")

        marks = []
        for script in QUICKSTART_SCRIPTS:
            notebook = nbformat.read(str(script.with_suffix(".ipynb")), as_version=4)
            marks.append(
                tuple(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
            )
        assert len(set(marks)) == len(marks), (
            "多个语言版本的 markdown 单元完全相同——说明译文没有翻译说明文字。"
        )
