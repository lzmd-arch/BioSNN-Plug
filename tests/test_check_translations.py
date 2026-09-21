"""``scripts/check_translations.py`` 的负向测试。

**为什么这个脚本必须被测试**：三语化之后最大的风险不是"翻译得不好"，而是
**改了一处忘了另两处**——漂移的译文比没有译文更误导人，读者会以为读到的是当前状态。
检查脚本是唯一的防线，所以它必须被证明真的抓得到东西。本模块里绝大多数用例是
"故意造出漂移，断言脚本失败"。

测试直接导入脚本里的检查函数，并把 ``REPO_ROOT`` 指向临时目录——这样测的是构造的
用例，而不是仓库自己的现状。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_translations.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_translations_under_test", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker(tmp_path, monkeypatch):
    """把检查器指向一个空的临时"仓库根"。"""
    module = _load_checker()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    return module


def run_checks(module) -> list[str]:
    """跑完整检查，返回问题的可读列表。"""
    problems: list = []
    bases = module.resolve_scope()
    module.check_completeness(bases, problems)
    scope_rel = {module.relative_posix(b) for b in bases}
    for base in bases:
        if module.relative_posix(base) in module.EXEMPT:
            continue
        texts = module.collect_texts(base)
        module.check_structure(base, texts, problems)
        module.check_switcher(base, texts, problems)
        module.check_code_blocks(base, texts, problems)
        module.check_link_language(base, texts, scope_rel, problems)
        module.check_anchors(base, texts, problems)
    return [p.render() for p in problems]


def kinds(problems: list[str]) -> set[str]:
    return {p.split("[", 1)[1].split("]", 1)[0] for p in problems}


# ────────────────────────── 构造用例 ──────────────────────────


SWITCHERS = {
    "": "[English](README.en.md) · [日本語](README.ja.md)",
    "en": "[中文](README.md) · [日本語](README.ja.md)",
    "ja": "[中文](README.md) · [English](README.en.md)",
}


def doc(root: Path, name: str, body: str, *, lang: str = "") -> Path:
    """写一份文档。``lang`` 非空时自动在顶部插入语言切换行。"""
    text = f"{SWITCHERS[lang]}\n\n{body}" if lang else body
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


BODY = "# 标题\n\n## 第一节\n\n```python\nx = 1\n```\n"


def trio(root: Path, body_zh: str = BODY, body_en: str | None = None, body_ja: str | None = None):
    """写一份文档的三语版本。"""
    doc(root, "README.md", body_zh)
    doc(root, "README.en.md", body_en or BODY, lang="en")
    doc(root, "README.ja.md", body_ja or BODY, lang="ja")


# ────────────────────────── 正向 ──────────────────────────


class TestAcceptsGoodInput:
    def test_consistent_trio_passes(self, checker, tmp_path):
        trio(tmp_path, BODY, BODY.replace("标题", "Title"), BODY.replace("标题", "タイトル"))
        assert run_checks(checker) == []

    def test_translated_comments_are_allowed(self, checker, tmp_path):
        """代码块里的注释本来就该翻译——不剥掉注释就会把正常的翻译判成错误。"""
        trio(
            tmp_path,
            "# T\n\n## S\n\n```python\nx = 1  # 中文注释\n```\n",
            "# T\n\n## S\n\n```python\nx = 1  # english comment\n```\n",
            "# T\n\n## S\n\n```python\nx = 1  # 日本語コメント\n```\n",
        )
        assert run_checks(checker) == []

    def test_translated_docstrings_are_allowed(self, checker, tmp_path):
        """docstring 是字符串字面量、会进 AST，但它同样是散文，翻译它是应该的。"""
        triple = '# T\n\n## S\n\n```python\ndef f():\n    """{text}"""\n    return 1\n```\n'
        trio(
            tmp_path,
            triple.format(text="中文说明"),
            triple.format(text="english"),
            triple.format(text="日本語"),
        )
        assert run_checks(checker) == []

    def test_translated_trailing_comment_is_allowed(self, checker, tmp_path):
        """回归测试：**行尾**注释也要能翻译。

        第一版只剔除"整行以 # 开头"的注释，于是
        `uv sync  # 一条命令装好全部开发依赖` 这行的中文注释无法翻译——译了就
        CI 红。结果是英文文档里被迫留着中文注释，**工具逼出了坏输出**。
        """
        trio(
            tmp_path,
            "# T\n\n## S\n\n```bash\nuv sync  # 一条命令装好全部依赖\n```\n",
            "# T\n\n## S\n\n```bash\nuv sync  # installs every dev dependency\n```\n",
            "# T\n\n## S\n\n```bash\nuv sync  # 開発依存をまとめて導入\n```\n",
        )
        assert run_checks(checker) == []

    def test_hash_inside_quotes_is_not_a_comment(self, checker, tmp_path):
        """引号里的 # 是字符串内容，不能被当注释剥掉——否则改坏了也发现不了。"""
        trio(
            tmp_path,
            '# T\n\n## S\n\n```bash\necho "a#b"\n```\n',
            '# T\n\n## S\n\n```bash\necho "a#b"\n```\n',
            '# T\n\n## S\n\n```bash\necho "x#y"\n```\n',
        )
        assert "代码块" in kinds(run_checks(checker))

    def test_translated_mermaid_labels_are_allowed(self, checker, tmp_path):
        """mermaid 的节点标签与 text 块的目录树都是散文，翻译它们是正确的。"""
        trio(
            tmp_path,
            "# T\n\n## S\n\n```mermaid\nA[中文] --> B\n```\n",
            "# T\n\n## S\n\n```mermaid\nA[English] --> B\n```\n",
            "# T\n\n## S\n\n```mermaid\nA[日本語] --> B\n```\n",
        )
        assert run_checks(checker) == []

    def test_registry_name_in_python_is_not_a_docstring(self, checker, tmp_path):
        """注册名这类字符串不是 docstring，改了必须被抓到。"""
        trio(
            tmp_path,
            '# T\n\n## S\n\n```python\nbus.register(get_plugin("audio")())\n```\n',
            '# T\n\n## S\n\n```python\nbus.register(get_plugin("audio")())\n```\n',
            '# T\n\n## S\n\n```python\nbus.register(get_plugin("audio")())\n```\n',
        )
        assert run_checks(checker) == []


# ────────────────────────── 负向：四种真实漂移 ──────────────────────────


class TestCatchesDrift:
    def test_missing_translation(self, checker, tmp_path):
        doc(tmp_path, "README.md", BODY)
        problems = run_checks(checker)
        assert "完整性" in kinds(problems)
        assert sum("完整性" in p for p in problems) == 2  # en 与 ja 各缺一份

    def test_missing_heading(self, checker, tmp_path):
        """译文漏掉一节——翻译时最常见、肉眼最难发现的错。"""
        trio(
            tmp_path,
            BODY,
            "# Title\n\n```python\nx = 1\n```\n",  # 少了 `## 第一节`
            BODY.replace("标题", "タイトル"),
        )
        assert "结构" in kinds(run_checks(checker))

    def test_extra_heading(self, checker, tmp_path):
        trio(
            tmp_path,
            BODY,
            "# Title\n\n## Section\n\n## Extra\n\n```python\nx = 1\n```\n",
            BODY.replace("标题", "タイトル"),
        )
        assert "结构" in kinds(run_checks(checker))

    def test_changed_code_value(self, checker, tmp_path):
        """改了一个参数值——代码还跑得通，但内容是错的。

        这正是 check_doc_code_blocks.py 抓不到的那一类：它能发现"跑不通"，
        发现不了"跑得通但不一样"。
        """
        trio(
            tmp_path,
            BODY,
            "# Title\n\n## Section\n\n```python\nx = 2\n```\n",
            BODY.replace("标题", "タイトル"),
        )
        assert "代码块" in kinds(run_checks(checker))

    def test_changed_shell_command(self, checker, tmp_path):
        trio(
            tmp_path,
            "# T\n\n## S\n\n```bash\npip install biosnn-bus\n```\n",
            "# T\n\n## S\n\n```bash\npip install something-else\n```\n",
            "# T\n\n## S\n\n```bash\npip install biosnn-bus\n```\n",
        )
        assert "代码块" in kinds(run_checks(checker))

    def test_registry_name_in_no_run_block_is_caught(self, checker, tmp_path):
        """``no-run`` 块里的注册名被翻译——必须抓。

        ``no-run`` 块**不会被执行**，所以没有执行检查兜底，AST 比对是唯一防线，
        因此这类块的字符串字面量保留参与比较。
        """
        trio(
            tmp_path,
            '# T\n\n## S\n\n```python no-run\nget_plugin("audio")\n```\n',
            '# T\n\n## S\n\n```python no-run\nget_plugin("audio")\n```\n',
            '# T\n\n## S\n\n```python no-run\nget_plugin("オーディオ")\n```\n',
        )
        assert "代码块" in kinds(run_checks(checker))

    def test_executable_block_strings_are_not_compared(self, checker, tmp_path):
        """可执行块里的字符串**不**参与比对——这是刻意的设计，不是漏检。

        可执行块的功能性字符串由 ``check_doc_code_blocks.py`` 的**执行**兜底：
        把 ``get_plugin("audio")`` 的注册名译掉，代码会直接抛
        ``PluginNotFoundError``。所以这里放开字符串，让译者能翻译
        ``print("解码回来:", ...)`` 这类面向读者的输出标签——它们不改程序语义，
        留在译文里却是明显的半成品。

        （早期版本比较字符串，结果是英日文档里被迫留着中文输出标签——
        工具逼出了坏输出。）
        """
        trio(
            tmp_path,
            '# T\n\n## S\n\n```python\nprint("解码回来:", x)\n```\n',
            '# T\n\n## S\n\n```python\nprint("decoded:", x)\n```\n',
            '# T\n\n## S\n\n```python\nprint("復号結果:", x)\n```\n',
        )
        assert run_checks(checker) == []

    def test_numbers_and_identifiers_are_still_compared(self, checker, tmp_path):
        """放开字符串不等于放开一切——数字与标识符改了仍然必须抓到。"""
        trio(
            tmp_path,
            "# T\n\n## S\n\n```python\nbus = SpikeBus(bus_dim=128, seed=0)\n```\n",
            "# T\n\n## S\n\n```python\nbus = SpikeBus(bus_dim=256, seed=0)\n```\n",
            "# T\n\n## S\n\n```python\nbus = SpikeBus(bus_dim=128, seed=0)\n```\n",
        )
        assert "代码块" in kinds(run_checks(checker))

    def test_broken_anchor_is_caught(self, checker, tmp_path):
        """锚点指向一个不存在的标题——路径对、锚点错，读者落在页首。

        译文标题是翻译的，锚点随之改变，这类错**几乎必然发生**，且肉眼极难发现。
        """
        doc(tmp_path, "docs/guide.md", "# 指南\n\n## 让第三方包提供插件\n")
        doc(
            tmp_path,
            "docs/guide.en.md",
            "# Guide\n\n## Third-party plugin integration\n",
            lang="en",
        )
        doc(tmp_path, "docs/guide.ja.md", "# ガイド\n\n## サードパーティ\n", lang="ja")
        doc(tmp_path, "README.md", "# T\n\n## S\n\n[x](docs/guide.md#让第三方包提供插件)\n")
        doc(
            tmp_path,
            "README.en.md",
            "# T\n\n## S\n\n[x](docs/guide.en.md#third-party-plugin-integration)\n",
            lang="en",
        )
        doc(
            tmp_path,
            "README.ja.md",
            "# T\n\n## S\n\n[x](docs/guide.ja.md#存在しない見出し)\n",
            lang="ja",
        )

        problems = run_checks(checker)
        assert "锚点" in kinds(problems)
        assert any("存在しない見出し" in p for p in problems)

    def test_link_pointing_at_chinese_original(self, checker, tmp_path):
        """译文的内部链接指回中文原文——读者点一下会跳回看不懂的语言。"""
        doc(tmp_path, "docs/guide.md", "# 指南\n\n## 节\n")
        doc(tmp_path, "docs/guide.en.md", "# Guide\n\n## S\n", lang="en")
        doc(tmp_path, "docs/guide.ja.md", "# ガイド\n\n## 節\n", lang="ja")
        doc(tmp_path, "README.md", "# T\n\n## S\n\n[指南](docs/guide.md)\n")
        doc(tmp_path, "README.en.md", "# T\n\n## S\n\n[guide](docs/guide.md)\n", lang="en")
        doc(tmp_path, "README.ja.md", "# T\n\n## S\n\n[ガイド](docs/guide.md)\n", lang="ja")
        problems = run_checks(checker)
        assert "链接" in kinds(problems)
        assert any("guide.ja.md" in p for p in problems)

    def test_missing_switcher_line(self, checker, tmp_path):
        doc(tmp_path, "README.md", BODY)
        doc(tmp_path, "README.en.md", BODY, lang="en")
        doc(tmp_path, "README.ja.md", BODY)  # 没有切换行
        assert "切换行" in kinds(run_checks(checker))

    def test_switcher_missing_one_language(self, checker, tmp_path):
        """切换行只链了中文，没链英文——读者从日文版跳不到英文版。"""
        doc(tmp_path, "README.md", BODY)
        doc(tmp_path, "README.en.md", BODY, lang="en")
        # 直接写文件而不走 doc() 助手，因为助手会自动补全三语切换行
        (tmp_path / "README.ja.md").write_text("[中文](README.md)\n\n" + BODY, encoding="utf-8")
        assert "切换行" in kinds(run_checks(checker))

    def test_switcher_links_are_not_flagged_as_wrong_language(self, checker, tmp_path):
        """回归测试：切换行的链接**本来就该**跨语言，不能判它违规。

        第一版检查把 `README.ja.md` 里指向 `README.md` 的切换链接报成"链接语言
        不一致"——那是自相矛盾的，切换行的全部意义就是跳到别的语言版本。
        """
        trio(tmp_path, BODY, BODY.replace("标题", "Title"), BODY.replace("标题", "タイトル"))
        assert run_checks(checker) == []
        assert "链接" not in kinds(run_checks(checker))

    def test_extra_code_block(self, checker, tmp_path):
        trio(
            tmp_path,
            BODY,
            "# Title\n\n## Section\n\n```python\nx = 1\n```\n\n```python\ny = 2\n```\n",
            BODY.replace("标题", "タイトル"),
        )
        assert "代码块" in kinds(run_checks(checker))


# ────────────────────────── 豁免与范围 ──────────────────────────


class TestScopeAndExemptions:
    def test_every_exemption_states_a_reason(self, checker):
        """豁免是检查的漏洞，必须逐条写明理由——否则它会变成藏问题的地方。"""
        for path, reason in checker.EXEMPT.items():
            assert reason.strip(), f"{path} 的豁免没有写明理由"

    def test_glossary_is_exempt(self, checker, tmp_path):
        """术语表本身即三语对照，不应要求它有 .en / .ja 变体。"""
        doc(tmp_path, "docs/GLOSSARY.md", "# 术语表\n\n## 一\n")
        assert run_checks(checker) == []

    def test_documents_outside_scope_are_ignored(self, checker, tmp_path):
        """计划书等不在范围内的文档不应触发完整性检查。"""
        doc(tmp_path, "BioSNN-Plug_项目计划书_v6.2.md", "# 计划书\n\n## 一\n")
        assert run_checks(checker) == []

    def test_scope_finds_nested_docs(self, checker, tmp_path):
        doc(tmp_path, "docs/adr/ADR-0001-x.md", "# ADR\n\n## Context\n")
        problems = run_checks(checker)
        assert any("ADR-0001-x" in p for p in problems)
