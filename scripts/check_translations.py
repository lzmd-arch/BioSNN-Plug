#!/usr/bin/env python3
"""文档三语一致性检查。

计划书 §12.4 要求文档可复现、可维护。文档三语化之后最大的风险不是"翻译得不好"，
而是**改了一处忘了另两处**——漂移的译文比没有译文更误导人，读者会以为读到的是当前
状态。本脚本把"漏改"从静默失败变成 CI 红灯。

三项检查，对应三种真实漂移：

1. **完整性**——声明范围内每个文档都必须有 `.en.md` 与 `.ja.md` 姊妹文件。
   新增文档时忘记翻译会立刻暴露，而不是攒到某天才发现。

2. **结构对齐**——三个语言版本的标题层级序列必须完全一致。翻译时漏掉一节、多起一节、
   或把 `###` 写成 `##` 都会被抓到。同时校验顶部的语言切换行存在。

3. **代码块一致**——文档里的代码块在三语中必须**语义相同**。
   `check_doc_code_blocks.py` 只能发现代码块"跑不通"，发现不了"还跑得通但内容变了"
   （例如 `spike_dim=32` 被改成 `16`）。

   关键设计：**注释必须允许翻译**，所以不能逐字比较——把中文注释原样留在日文文档里
   才是错的。因此 Python 块比较 `ast.dump()`（AST 天然不含注释，语义变化必被抓到），
   其它代码块剔除整行注释后比较。

4. **链接语言一致**——`.ja.md` 里的内部链接必须指向 `.ja.md`，不能指回中文原文。
   这是三语文档最容易犯、也最难靠肉眼发现的错。

用法::

    python scripts/check_translations.py
    python scripts/check_translations.py --fix-hint   # 只列出需要补的文件
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 参与三语化的文档范围。与 check_doc_code_blocks.py 的 DEFAULT_TARGETS 保持一致。
SCOPE_PATTERNS = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/**/*.md",
    "packages/*/README.md",
    "research/README.md",
    "research/**/README.md",
]

#: 语言后缀。基准语言（中文）不带后缀。
LANGUAGES = ("en", "ja")

#: 豁免的文档及其理由。**每条都必须写明原因**——豁免是漏洞，必须可审计。
EXEMPT = {
    "docs/GLOSSARY.md": "术语表本身即三语对照（中文 / English / 日本語 三列），无需分语言版本",
    "CODE_OF_CONDUCT.md": "Contributor Covenant 标准文本，其官方译本由 contributor-covenant.org 维护；本仓库不复制官方译本（见下方说明）",
}

#: 不参与"链接语言一致"检查的目标——它们没有语言版本，任何语言指向它们都合理。
NON_TRANSLATED_TARGETS = {
    "LICENSE",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "docs/GLOSSARY.md",
    "BioSNN-Plug_项目计划书_v6.2.md",
}

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*(?P<info>.*?)\s*$")
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+\S")
LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s]+)\)")
DOC_SUFFIX_RE = re.compile(r"\.(?P<lang>en|ja)\.md$")

#: 只有这些围栏语言参与"代码块一致"比较。
#:
#: 刻意排除 `text` 与 `mermaid`：它们装的是**散文**——`text` 块常用来画目录树，
#: `mermaid` 块的节点标签就是图上的文字，两者本来就要翻译。拿它们逐字比较只会
#: 产生一堆假报错。（第一版就踩了这个坑。）
COMPARABLE_LANGS = {
    "python",
    "python3",
    "py",
    "bash",
    "sh",
    "shell",
    "console",
    "toml",
    "yaml",
    "yml",
    "json",
    "ini",
    "cfg",
    "dockerfile",
}


class Problem:
    def __init__(self, path: Path, kind: str, message: str) -> None:
        self.path = path
        self.kind = kind
        self.message = message

    def render(self) -> str:
        try:
            shown = self.path.relative_to(REPO_ROOT)
        except ValueError:
            shown = self.path
        return f"{shown}: [{self.kind}] {self.message}"


# ────────────────────────────── 文件发现 ──────────────────────────────


def resolve_scope() -> list[Path]:
    """返回参与三语化的**基准**（中文）文档。"""
    found: list[Path] = []
    for pattern in SCOPE_PATTERNS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if not path.is_file():
                continue
            if DOC_SUFFIX_RE.search(path.name):
                continue  # 已是语言变体
            found.append(path)
    return list(dict.fromkeys(found))


def relative_posix(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def variant_of(path: Path, lang: str) -> Path:
    """``README.md`` + ``ja`` -> ``README.ja.md``；``lang=""`` 返回基准文件本身。"""
    if not lang:
        return path
    return path.with_name(f"{path.stem}.{lang}{path.suffix}")


# ────────────────────────────── 解析 ──────────────────────────────


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def heading_levels(text: str) -> list[int]:
    """标题层级序列，忽略围栏内的内容。"""
    levels: list[int] = []
    fence: str | None = None
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            if fence is None:
                fence = match.group(1)
            elif line.lstrip().startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            levels.append(len(heading.group("hashes")))
    return levels


def code_blocks(text: str) -> list[tuple[str, str]]:
    """返回 ``(info, content)`` 列表。info 是围栏后的**完整**信息串
    （例如 ``python no-run``），因为判断一个块是否会被执行需要看到标志位。"""
    blocks: list[tuple[str, str]] = []
    fence: str | None = None
    info = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            if fence is None:
                fence = match.group(1)
                info = match.group("info").strip()
                buffer = []
            elif line.lstrip().startswith(fence):
                blocks.append((info, "\n".join(buffer)))
                fence = None
            continue
        if fence is not None:
            buffer.append(line)
    return blocks


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """去掉模块 / 类 / 函数体的首个字符串表达式（docstring）。

    docstring 是字符串字面量，会进入 AST。而示例代码里的 docstring 是**散文**，
    翻译它是应该的——不剥掉就会把"翻译了 docstring"误判成"改坏了代码"。
    剥掉之后，其余字符串（如 `get_plugin("audio")` 里的注册名）仍参与比较，
    改了照样会被抓到。
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


#: 与 check_doc_code_blocks.py 一致：带这些标志的围栏**不会被执行**。
NO_RUN_FLAGS = {"no-run", "norun"}


def is_executed(info: str) -> bool:
    """该代码块是否会被 ``check_doc_code_blocks.py`` 真实执行。"""
    flags = {token.lower() for token in info.split()[1:]}
    return not (flags & NO_RUN_FLAGS)


def _blank_string_constants(tree: ast.AST) -> None:
    """把所有字符串字面量替换成占位符。

    **为什么可以这么做**：能否翻译字符串要分两种情况。

    * **会被执行的块**（绝大多数）——字符串的功能性内容由 ``check_doc_code_blocks.py``
      的执行检查兜底：把 ``get_plugin("audio")`` 里的注册名译掉，代码会直接抛
      ``PluginNotFoundError``，CI 立刻红。所以这里可以放心抹平字符串，让译者能翻译
      ``print("解码回来:", ...)`` 这类**面向读者的输出标签**——它们不改程序语义，
      但留在译文里就是明显的半成品。
    * **``no-run`` 块**——不会被执行，没有执行检查兜底，所以**保留**字符串参与
      比较，AST 是唯一防线。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "<str>"


def normalise_code(info: str, content: str) -> str:
    """把代码块归一化成可比较的形式；返回空串表示"不比较"。

    - Python：比较剥掉 docstring 后的 AST。注释本就不在 AST 里、docstring 已剥离，
      所以**译文可以自由翻译注释与 docstring**，而表达式、标识符、数字、注册名等
      任何改动都会被抓到。
    - 其它代码语言：剔除整行注释后比较。
    - `text` / `mermaid` / 未知标记：**不比较**——它们装的是散文（目录树、
      图的节点标签），本来就要翻译。

    注释（整行或行尾）与 docstring 都允许翻译；代码、命令、标识符、字符串字面量的
    任何改动都会被抓到。
    """
    lang = info.split()[0].lower() if info.strip() else ""
    if lang not in COMPARABLE_LANGS:
        return ""
    if lang in ("python", "python3", "py"):
        try:
            tree = _strip_docstrings(ast.parse(content))
            if is_executed(info):
                _blank_string_constants(tree)
            return "AST:" + ast.dump(tree)
        except SyntaxError:
            # 语法错误由 check_doc_code_blocks.py 负责报告，这里退回文本比较
            pass
    kept = [stripped for line in content.splitlines() if (stripped := _strip_comment(line))]
    return "TEXT:" + "\n".join(kept)


def _strip_comment(line: str) -> str:
    """去掉一行里的注释（整行注释或行尾注释），返回剩余代码（已 rstrip）。

    **为什么必须处理行尾注释**：第一版只剔除"整行以 # 开头"的注释，于是
    ``uv sync  # 一条命令装好全部开发依赖`` 这行的中文注释无法翻译——译了就
    CI 红。结果是英文文档里被迫留着中文注释：**工具逼出了坏输出**。
    注释是散文，本来就该翻译，所以这里连同行尾注释一起剥掉。

    引号感知：``echo "a#b"`` 里的 # 是字符串内容而非注释，用简单的状态机处理
    单双引号，够用且不必为每种语言引入解析器。
    """
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            line = line[:index]
            break
    return line.rstrip()


# ────────────────────────────── 检查项 ──────────────────────────────


def check_completeness(bases: list[Path], problems: list[Problem]) -> None:
    for base in bases:
        rel = relative_posix(base)
        if rel in EXEMPT:
            continue
        for lang in LANGUAGES:
            if not variant_of(base, lang).exists():
                problems.append(
                    Problem(
                        base,
                        "完整性",
                        f"缺少 {lang} 版本：{variant_of(base, lang).name}。"
                        f"新增或改名文档时三份必须一起提交。",
                    )
                )


def check_structure(base: Path, texts: dict[str, str], problems: list[Problem]) -> None:
    base_levels = heading_levels(texts[""])
    for lang in LANGUAGES:
        if not texts.get(lang):
            continue
        levels = heading_levels(texts[lang])
        if levels != base_levels:
            detail = _first_divergence(base_levels, levels)
            problems.append(
                Problem(
                    variant_of(base, lang),
                    "结构",
                    f"标题层级与中文原文不一致：{detail}。"
                    f"原文 {len(base_levels)} 节，译文 {len(levels)} 节。",
                )
            )


def _first_divergence(base: list[int], other: list[int]) -> str:
    for index, (a, b) in enumerate(zip(base, other, strict=False)):
        if a != b:
            return f"第 {index + 1} 个标题层级为 H{b}，原文是 H{a}"
    if len(base) != len(other):
        longer = "译文" if len(other) > len(base) else "原文"
        return f"{longer}多出 {abs(len(base) - len(other))} 个标题"
    return "层级序列不同"


def check_switcher(base: Path, texts: dict[str, str], problems: list[Problem]) -> None:
    """每份译文顶部必须有指向另外两个语言版本的语言切换链接。"""
    for lang in LANGUAGES:
        text = texts.get(lang)
        if not text:
            continue
        head = "\n".join(text.splitlines()[:20])
        missing = []
        for other in ("", *LANGUAGES):
            if other == lang:
                continue
            if variant_of(base, other).name not in head:
                missing.append(variant_of(base, other).name)
        if missing:
            problems.append(
                Problem(
                    variant_of(base, lang),
                    "切换行",
                    f"文档顶部缺少指向 {'、'.join(missing)} 的语言切换链接。"
                    f"读者因此无法从这一版跳到另外两版。",
                )
            )


def check_code_blocks(base: Path, texts: dict[str, str], problems: list[Problem]) -> None:
    base_blocks = code_blocks(texts[""])
    for lang in LANGUAGES:
        text = texts.get(lang)
        if not text:
            continue
        blocks = code_blocks(text)
        if len(blocks) != len(base_blocks):
            problems.append(
                Problem(
                    variant_of(base, lang),
                    "代码块",
                    f"代码块数量与原文不一致：原文 {len(base_blocks)} 个，译文 {len(blocks)} 个。",
                )
            )
            continue
        # 上面已确认三语代码块数量相等，strict=True 顺带当一道断言
        for index, ((base_info, base_body), (info, body)) in enumerate(
            zip(base_blocks, blocks, strict=True), start=1
        ):
            expected = normalise_code(base_info, base_body)
            if not expected:
                continue  # 散文块（text / mermaid），本来就要翻译
            if expected != normalise_code(info, body):
                problems.append(
                    Problem(
                        variant_of(base, lang),
                        "代码块",
                        f"第 {index} 个代码块（```{base_info}）与原文语义不一致。"
                        f"代码、命令、标识符不应被翻译——注释与 docstring 可以。",
                    )
                )


def strip_lang_suffix(rel: str) -> str:
    """``docs/guide.ja.md`` -> ``docs/guide.md``；无后缀则原样返回。"""
    return DOC_SUFFIX_RE.sub(".md", rel)


def check_link_language(
    base: Path, texts: dict[str, str], scope_rel: set[str], problems: list[Problem]
) -> None:
    """译文的**内容链接**必须指向同一语言的版本。

    例外：**语言切换行**里的链接——那些链接的全部意义就是跳到别的语言版本，
    判它们违规是自相矛盾的。判据是"目标解析后与本文件同属一份基准文档"：
    `README.ja.md` 里的 `README.md` / `README.en.md` 是切换行，放行；
    而 `README.ja.md` 里的 `docs/guide.md` 是内容链接，必须指向 `guide.ja.md`。
    """
    base_rel = relative_posix(base)
    for lang in LANGUAGES:
        text = texts.get(lang)
        if not text:
            continue
        doc = variant_of(base, lang)
        for match in LINK_RE.finditer(text):
            target = match.group("target")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part.endswith(".md"):
                continue
            resolved = (doc.parent / path_part).resolve()
            try:
                rel = resolved.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            if rel in NON_TRANSLATED_TARGETS or rel not in scope_rel:
                continue
            if strip_lang_suffix(rel) == base_rel:
                continue  # 语言切换行，本就该跨语言
            expected = variant_of(Path(rel), lang).as_posix()
            if rel != expected:
                problems.append(
                    Problem(
                        doc,
                        "链接",
                        f"内部链接指向 {rel}（{'中文原文' if not DOC_SUFFIX_RE.search(rel) else '另一语言'}），"
                        f"应指向 {expected}——否则读者点一下会跳回看不懂的语言。",
                    )
                )


def github_slug(text: str) -> str:
    """近似 GitHub 由标题文本生成锚点的算法。

    GitHub 会小写化、剥掉 HTML 标签、去掉标点（保留字母/数字/下划线/连字符/**汉字**）、
    把空格换成连字符。这里是近似实现——够用来发现"锚点指向一个不存在的标题"，
    不必做到与 GitHub 逐字节一致。
    """
    text = re.sub(r"<[^>]+>", "", text.strip().lower())
    text = re.sub(r"[^\w\s一-鿿-]", "", text)
    return text.replace(" ", "-")


def heading_slugs(path: Path) -> set[str]:
    """文件里全部标题的锚点集合（忽略围栏内的代码）。"""
    slugs: set[str] = set()
    fence: str | None = None
    for line in read(path).splitlines():
        match = FENCE_RE.match(line)
        if match:
            if fence is None:
                fence = match.group(1)
            elif line.lstrip().startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = re.match(r"^#{1,6}\s+(?P<text>.+?)\s*$", line)
        if heading:
            slugs.add(github_slug(heading.group("text")))
    return slugs


def check_anchors(base: Path, texts: dict[str, str], problems: list[Problem]) -> None:
    """链接里的锚点必须真的对应目标文件里的某个标题。

    只校验链接**路径**是不够的：`docs/plugin_guide.en.md#third-party-plugin-integration`
    的路径可以完全正确，而锚点指向一个不存在的标题——读者点进去落在页首，
    以为内容被删了。译文标题是翻译的，锚点随之改变，这类错**几乎必然发生**，
    且肉眼极难发现。
    """
    for lang, text in texts.items():
        if not text:
            continue
        doc = variant_of(base, lang)
        for match in LINK_RE.finditer(text):
            target = match.group("target")
            if target.startswith(("http://", "https://", "mailto:")) or "#" not in target:
                continue
            path_part, anchor = target.split("#", 1)
            if not path_part.endswith(".md"):
                continue
            resolved = (doc.parent / path_part).resolve()
            if not resolved.exists():
                continue  # 路径问题由 check_link_language / lychee 负责
            if anchor not in heading_slugs(resolved):
                problems.append(
                    Problem(
                        doc,
                        "锚点",
                        f"链接 {target} 的锚点 #{anchor} 在 {path_part} 里找不到对应标题。"
                        f"读者会落在页首。译文标题是翻译的，锚点必须跟着译文的标题走。",
                    )
                )


# ────────────────────────────── 入口 ──────────────────────────────


def collect_texts(base: Path) -> dict[str, str]:
    texts = {"": read(base)}
    for lang in LANGUAGES:
        path = variant_of(base, lang)
        texts[lang] = read(path) if path.exists() else ""
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fix-hint",
        action="store_true",
        help="只列出缺失的译文文件（用于批量补译时生成清单）",
    )
    args = parser.parse_args(argv)

    bases = resolve_scope()
    problems: list[Problem] = []

    if args.fix_hint:
        missing = [
            variant_of(b, lang).relative_to(REPO_ROOT).as_posix()
            for b in bases
            if relative_posix(b) not in EXEMPT
            for lang in LANGUAGES
            if not variant_of(b, lang).exists()
        ]
        for name in missing:
            print(name)
        print(f"\n共 {len(missing)} 份待补译文（基准文档 {len(bases)} 份）")
        return 0

    check_completeness(bases, problems)

    scope_rel = {relative_posix(b) for b in bases}
    for base in bases:
        if relative_posix(base) in EXEMPT:
            continue
        texts = collect_texts(base)
        check_structure(base, texts, problems)
        check_switcher(base, texts, problems)
        check_code_blocks(base, texts, problems)
        check_link_language(base, texts, scope_rel, problems)
        check_anchors(base, texts, problems)

    for problem in problems:
        print(problem.render(), file=sys.stderr)

    if problems:
        print(f"\n三语一致性检查未通过：{len(problems)} 个问题。", file=sys.stderr)
        return 1

    print(
        f"三语一致性检查通过：{len(bases)} 份基准文档，"
        f"{len(bases) * (1 + len(LANGUAGES))} 份文件（含豁免 {len(EXEMPT)} 项）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
