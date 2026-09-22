#!/usr/bin/env python
"""检查 Markdown 文档里的 Python 代码块。

计划书 §12.3 要求用"文档代码块语法检查"拦截 ``self.threshold`` 这类未定义引用。
这里必须说清楚一件事：**纯 ``ast.parse`` 抓不到那类 bug**。它只看得懂语法，
``self.threshold`` 语法完全合法，问题在于运行时这个属性根本不存在。

所以本脚本做两件事：

1. **语法检查**（所有块）：``compile()`` 一遍，报出语法错误。
2. **执行检查**（默认开启）：同一个文档里的代码块**共享一个命名空间、按出现顺序
   真正跑一遍**。这才是能抓到 ``AttributeError: 'X' object has no attribute
   'threshold'`` 的那一层。

要写只作片段展示、无法独立运行的代码块，在围栏上标注即可::

    ```python no-run
    plugin = MyPlugin()   # MyPlugin 来自上文，这里只是示意
    ```

``python no-run`` 的块只做语法检查。这个约定写进了 CONTRIBUTING.md 的提交前
核查清单——**默认执行，例外显式标注**，比反过来更难糊弄过去。

用法::

    python scripts/check_doc_code_blocks.py            # 检查默认范围
    python scripts/check_doc_code_blocks.py README.md  # 只检查指定文件
    python scripts/check_doc_code_blocks.py --syntax-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 默认扫描范围：所有面向用户的文档。计划书本体是只读文件，不进扫描。
#:
#: 用通配符而非逐个列出语言版本：文档有三语（`X.md` / `X.en.md` / `X.ja.md`），
#: 每加一种语言就要改一次清单，迟早漏掉。`docs/**/*.md` 本来就是 glob，所以
#: `docs/` 下的译文一直是被覆盖的；这里把根目录与 packages/research 也对齐。
#:
#: **译文的代码块必须被真实执行**——译文里的示例烂掉，光靠结构检查发现不了。
DEFAULT_TARGETS = [
    "README*.md",
    "CONTRIBUTING*.md",
    "docs/**/*.md",
    "packages/*/README*.md",
    "research/README*.md",
    "research/**/README*.md",
]

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*(?P<info>.*?)\s*$")

#: 围栏信息串里带这些词就只做语法检查。
#: 刻意只留这两个写法——这是绕过检查的口子，写法越少越好审。
#: （早先还有 "skip" / "skip-exec"，它们过于宽泛，与将来可能引入的其它围栏
#:   属性容易撞名，已去掉。）
SKIP_EXEC_FLAGS = {"no-run", "norun"}

#: 会被当作 Python 处理的语言标记（含空标记）。
PYTHON_LANGS = {"python", "python3", "py", "py3"}

EXEC_TIMEOUT_SECONDS = 120

#: 在子进程里按文档原始行号执行代码块。把源码用换行补齐到原始行号再 compile，
#: 回溯里的行号就直接指向 Markdown 文件本身——不需要任何行号映射表。
EXEC_DRIVER = """
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    blocks = json.load(handle)

namespace = {"__name__": "__main__", "__file__": sys.argv[2]}
for block in blocks:
    padded = "\\n" * (block["line"] - 1) + block["source"]
    exec(compile(padded, block["path"], "exec"), namespace)
"""


@dataclass(frozen=True)
class CodeBlock:
    path: Path
    start_line: int
    source: str
    runnable: bool


@dataclass(frozen=True)
class Problem:
    path: Path
    line: int
    kind: str
    message: str

    def render(self) -> str:
        try:
            shown = self.path.relative_to(REPO_ROOT)
        except ValueError:
            shown = self.path
        return f"{shown}:{self.line}: [{self.kind}] {self.message}"


def is_python_fence(info: str) -> bool:
    tokens = info.split()
    if not tokens:
        return True  # ``` 不带语言标记的块，按 Python 处理
    return tokens[0].lower() in PYTHON_LANGS


def is_runnable_fence(info: str) -> bool:
    return not ({token.lower() for token in info.split()[1:]} & SKIP_EXEC_FLAGS)


def collect_blocks(path: Path) -> list[CodeBlock]:
    """抽出文档里的 Python 代码块。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[CodeBlock] = []

    fence: str | None = None
    fence_indent = 0
    fence_info = ""
    fence_start = 0
    buffer: list[str] = []

    for number, line in enumerate(lines, start=1):
        if fence is None:
            match = FENCE_RE.match(line)
            if match:
                fence = match.group(1)
                fence_indent = len(line) - len(line.lstrip())
                fence_info = match.group("info")
                fence_start = number
                buffer = []
            continue

        # 围栏内：只有同类型且不短于开启围栏的标记才算闭合
        stripped = line.lstrip()
        if stripped.startswith(fence) and set(stripped.rstrip()) == {fence[0]}:
            if is_python_fence(fence_info):
                # 去掉围栏自身的缩进（文档里代码块常常嵌在列表项下）
                dedented = [
                    (text[fence_indent:] if text.startswith(" " * fence_indent) else text.lstrip())
                    for text in buffer
                ]
                blocks.append(
                    CodeBlock(
                        path=path,
                        start_line=fence_start + 1,  # 第一个内容行的行号
                        source="\n".join(dedented),
                        runnable=is_runnable_fence(fence_info),
                    )
                )
            fence = None
            continue

        buffer.append(line)

    return blocks


def syntax_problems(block: CodeBlock) -> list[Problem]:
    padded = "\n" * (block.start_line - 1) + block.source
    try:
        compile(padded, str(block.path), "exec")
    except SyntaxError as exc:
        return [
            Problem(
                path=block.path,
                line=exc.lineno or block.start_line,
                kind="语法",
                message=f"{exc.msg}",
            )
        ]
    return []


def _with_repo_root_on_path() -> dict[str, str]:
    """当前环境变量，外加把仓库根放进 ``PYTHONPATH``。"""
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT), *(p for p in existing.split(os.pathsep) if p)]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(parts)}


def execution_problems(blocks: list[CodeBlock], *, python: str) -> list[Problem]:
    """在同一命名空间里按顺序执行可运行块，返回执行期错误。"""
    runnable = [b for b in blocks if b.runnable]
    if not runnable:
        return []

    payload = [{"path": str(b.path), "line": b.start_line, "source": b.source} for b in runnable]

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        driver = tmpdir / "_doc_blocks_driver.py"
        driver.write_text(EXEC_DRIVER, encoding="utf-8")
        payload_file = tmpdir / "blocks.json"
        payload_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        try:
            result = subprocess.run(
                [python, str(driver), str(payload_file), str(blocks[0].path)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                # 把仓库根放进 PYTHONPATH。驱动脚本在临时目录里，它的 sys.path[0]
                # 是那个临时目录，于是文档里 `import research` 这类**仓库内**的导入
                # 会失败——而读者在仓库根用 Jupyter 或 python 跑同一段代码时，
                # 当前目录本来就在 sys.path 上。检查器该复现读者的环境，不该比它更严。
                # （`biosnn_bus` 不在此列：它是装进虚拟环境的分发包，与路径无关。）
                env=_with_repo_root_on_path(),
                timeout=EXEC_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [
                Problem(
                    path=blocks[0].path,
                    line=blocks[0].start_line,
                    kind="超时",
                    message=f"文档代码块执行超过 {EXEC_TIMEOUT_SECONDS}s 未结束。",
                )
            ]

    if result.returncode == 0:
        return []

    return _parse_traceback(result.stderr, blocks[0].path, blocks[0].start_line)


_TRACEBACK_FRAME_RE = re.compile(r'^\s*File "(?P<path>.+?)", line (?P<line>\d+)')


def _parse_traceback(stderr: str, default_path: Path, default_line: int) -> list[Problem]:
    """从回溯里定位到文档代码块的具体行。

    回溯的最后一帧通常落在库内部（比如 ``raise PluginNotFoundError`` 那一行），
    对写文档的人毫无帮助。所以要优先挑**属于文档自己**的那一帧——那才是文档里需要
    改的地方。
    """
    frames = [
        (Path(match.group("path")), int(match.group("line")))
        for match in (_TRACEBACK_FRAME_RE.match(line) for line in stderr.splitlines())
        if match and match.group("path") != "<string>"
    ]

    document_frames = [frame for frame in frames if frame[0].suffix == ".md"]
    if document_frames:
        path, line = document_frames[-1]
    elif frames:
        path, line = frames[-1]
    else:
        path, line = default_path, default_line

    last_line = stderr.strip().splitlines()[-1] if stderr.strip() else "未知错误"
    return [Problem(path=path, line=line, kind="执行", message=last_line)]


def resolve_targets(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            found.extend(sorted(REPO_ROOT.glob(pattern)))
        else:
            candidate = REPO_ROOT / pattern
            if candidate.exists():
                found.append(candidate)
    # 去重并保持稳定顺序
    return list(dict.fromkeys(found))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="*", help="要检查的文件（支持 glob，默认见 DEFAULT_TARGETS）"
    )
    parser.add_argument("--syntax-only", action="store_true", help="只做语法检查，不执行")
    parser.add_argument(
        "--python", default=sys.executable, help="执行代码块用的解释器（默认当前解释器）"
    )
    args = parser.parse_args(argv)

    targets = resolve_targets(args.paths or DEFAULT_TARGETS)
    if not targets:
        print("没有找到任何待检查的文档。", file=sys.stderr)
        return 1

    problems: list[Problem] = []
    checked_blocks = 0
    executed_blocks = 0

    for path in targets:
        blocks = collect_blocks(path)
        checked_blocks += len(blocks)
        if not args.syntax_only:
            executed_blocks += sum(1 for b in blocks if b.runnable)

        for block in blocks:
            problems.extend(syntax_problems(block))
        if not args.syntax_only and not any(p.path == path for p in problems):
            problems.extend(execution_problems(blocks, python=args.python))

    for problem in problems:
        print(problem.render(), file=sys.stderr)

    summary = f"检查 {len(targets)} 个文档、{checked_blocks} 个代码块"
    if not args.syntax_only:
        summary += f"（其中 {executed_blocks} 个实际执行）"
    verdict = f"发现 {len(problems)} 个问题" if problems else "通过"
    print(f"{summary}：{verdict}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
