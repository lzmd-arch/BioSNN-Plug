"""冒烟测试：仓库里的示例必须真的能跑。

计划书 §12.3 把"最小复现脚本"列为 CI 检查项，要拦截的正是"论文说能跑但仓库跑不通"
这一类问题。示例脚本是这个项目对外的第一印象，它坏了没人会往下看——所以它进 CI。

同时校验 Colab notebook 与脚本同步：notebook 由 ``scripts/build_notebook.py`` 从
脚本生成，脚本是唯一真相源，两者不可能各自漂移。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
QUICKSTART = REPO_ROOT / "examples" / "quickstart_register_plugin.py"
NOTEBOOK = QUICKSTART.with_suffix(".ipynb")
BUILDER = REPO_ROOT / "scripts" / "build_notebook.py"

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


@pytest.fixture(scope="module")
def quickstart_output() -> str:
    """跑一次示例脚本，返回它的标准输出（全模块共用一个子进程）。"""
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("示例需要 matplotlib；请安装 biosnn-bus[examples]")
    result = run_script(str(QUICKSTART))
    assert result.returncode == 0, (
        f"示例脚本退出码 {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result.stdout


class TestQuickstartScript:
    def test_runs_to_completion(self, quickstart_output):
        assert "SpikeBus(" in quickstart_output

    def test_registers_both_modalities(self, quickstart_output):
        assert "sine_wave" in quickstart_output
        assert "image_diff" in quickstart_output

    def test_routes_the_two_channels(self, quickstart_output):
        """示例的核心卖点：时间通道与语义通道各走各的。"""
        assert "'temporal': ['sine_wave']" in quickstart_output
        assert "'semantic': ['image_diff']" in quickstart_output

    def test_produces_bus_output(self, quickstart_output):
        assert "总线输出: SpikeTrain(shape=(24, 128)" in quickstart_output

    def test_decodes_back_to_modality_space(self, quickstart_output):
        assert "最大误差 0.0161" in quickstart_output


class TestNotebookSync:
    def test_notebook_exists(self):
        assert NOTEBOOK.exists(), f"缺少 {NOTEBOOK.name}。请运行：python scripts/build_notebook.py"

    def test_notebook_matches_the_script(self):
        """回归测试：notebook 与脚本漂移时，CI 必须失败。

        CI 跑的是脚本，用户点开的是 notebook。两者一旦不同步，就会出现"CI 全绿但
        用户跑不通"的局面——这正是计划书 §12.3 要拦的那类问题。
        """
        result = run_script(str(BUILDER), "--check")
        assert result.returncode == 0, (
            f"notebook 与脚本不同步，请运行 python scripts/build_notebook.py\n"
            f"{result.stdout}{result.stderr}"
        )

    def test_notebook_is_valid_nbformat(self):
        nbformat = pytest.importorskip("nbformat")
        notebook = nbformat.read(str(NOTEBOOK), as_version=4)
        nbformat.validate(notebook)
        assert any(cell.cell_type == "code" for cell in notebook.cells)
        assert any(cell.cell_type == "markdown" for cell in notebook.cells)
