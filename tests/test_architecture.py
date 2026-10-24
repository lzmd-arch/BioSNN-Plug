"""分层守卫：`research/` 的依赖方向，以及骨架库的独立性。

规则写在 [`research/README.md`](../research/README.md) 的「目录约定」里，这里用 AST
把它变成可执行断言——**规则不靠自觉**，否则下一个主题会在没人注意时复制一份共享零件。

三层，依赖只朝一个方向：

```text
主题（eprop / ib_hebbian / rstdp / ...）
  └─→ components/   可参数化、不出结论的领域零件
        └─→ common/ 无领域知识的基建
```

跨主题只有两种合法的边：`arbiter/`、`continual/` 这类目的就是组合/基准的主题
**读对方包根导出的名字**（``from research.eprop import EPropLearner``），或者不读。
子模块路径（``research.eprop.eprop``）一律不行——它把对方的内部结构变成接口。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH = REPO_ROOT / "research"
SKELETON = REPO_ROOT / "packages" / "biosnn-bus" / "src"
SHARED = ("common", "components")


def _absolute_imports(path: Path) -> list[str]:
    """返回文件里所有**绝对** import 的模块名；相对 import 不出包，跳过。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def _owner(module: str) -> str | None:
    """``research.<x>....`` → ``<x>``；不是 research 内部的模块则返回 None。"""
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "research":
        return None
    return parts[1]


def _topic_of(path: Path) -> str | None:
    relative = path.relative_to(RESEARCH)
    if len(relative.parts) < 2:
        return None  # research/__init__.py 这类根文件
    return relative.parts[0]


def dependency_violations() -> list[str]:
    """按三层规则检查 `research/` 里的每一条 import。"""
    problems: list[str] = []
    for path in sorted(RESEARCH.rglob("*.py")):
        owner = _topic_of(path)
        if owner is None:
            continue
        for module in _absolute_imports(path):
            target = _owner(module)
            if target is None or target == owner:
                continue
            label = f"{path.relative_to(REPO_ROOT)}: {module}"
            if owner == "common":
                problems.append(f"{label} —— common/ 不能依赖 {target}/（它是最底层）")
            elif owner == "components":
                if target != "common":
                    problems.append(f"{label} —— components/ 只能依赖 common/，不能依赖 {target}/")
            elif target in SHARED:
                continue
            elif len(module.split(".")) != 2:
                problems.append(
                    f"{label} —— 跨主题只能 import 对方包根"
                    f"（from research.{target} import ...），不能碰子模块"
                )
    return problems


def skeleton_violations() -> list[str]:
    """ADR-0001：骨架库不依赖认知核心、研究代码或仓库脚本。"""
    problems: list[str] = []
    for path in sorted(SKELETON.rglob("*.py")):
        for module in _absolute_imports(path):
            if module.split(".")[0] in {"research", "scripts"}:
                problems.append(f"{path.relative_to(REPO_ROOT)}: {module}")
    return problems


def test_research_dependency_direction():
    assert dependency_violations() == []


def test_skeleton_stays_independent():
    assert skeleton_violations() == []


def test_the_guard_actually_sees_imports():
    """守卫自身的非空性检查：扫不到 import 的守卫会永远为绿。"""
    seen = _absolute_imports(RESEARCH / "arbiter" / "run_synergy.py")
    assert "research.eprop" in seen
