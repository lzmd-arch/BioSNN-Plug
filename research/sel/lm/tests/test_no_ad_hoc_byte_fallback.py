"""守卫：**"多字节字符被切成两半"这类错误不许再出现。**

背景（2026-11-12）：把千问 token 映射到 32k 子集时，子集外的 token 要走字节回退。
早先的写法是

    piece = tokenizer.decode([qwen_id]).encode("utf-8", "ignore")

它对**只覆盖某个汉字一部分字节**的 token（例如 `b"\\xa0\\x80"`）是错的：`decode` 先
得到一个替换字符，再 `encode` 就把原始字节永久换成了 `EF BF BD`。语料会被悄悄写坏，
**不报错**。同一个错误在三个脚本里各写过一遍。

这里做两件事：

1. **静态扫描**：仓库里不许再出现 `decode(...).encode(...)` 这种"先解码再编码"的回退；
2. **行为对照**：拿真实语料行做编码→解码往返，必须逐字相同。

第 1 条挡的是"又抄一遍"，第 2 条挡的是"抄了但看起来对"。`SubsetTokenizer` 是唯一
允许实现这套映射的地方。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# 本文件在 research/sel/lm/tests/ 下，往上第 5 层才是仓库根。
# 写成 parents[3] 会指到 research/，于是 SCAN_ROOTS 全部落空、两条断言双双假绿。
REPO = Path(__file__).resolve().parents[4]
ANTI_PATTERN = re.compile(r"\.decode\([^)]*\)\s*\.encode\(")
SKIP_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    "ToolsRuby",
    "sweep_results",
    # 守卫自己会提到这些名字，扫它没有意义
    "tests",
}
SCAN_ROOTS = ("scripts", "research", "packages")


def _python_files() -> list[Path]:
    found: list[Path] = []
    for root in SCAN_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            found.append(path)
    return found


def test_the_scan_actually_sees_files() -> None:
    """**先证明扫描器扫到了东西**：空集合上跑断言，永远是绿的。"""
    files = _python_files()
    assert len(files) > 50, f"只扫到 {len(files)} 个文件，路径根可能不对：{REPO}"
    assert any(path.name == "subset_tokenizer.py" for path in files)


def test_no_decode_then_encode_byte_fallback() -> None:
    """**这条是本次事故的直系守卫。**

    允许的写法只有一种：取 token 的原始字节（`_bytes_of_qwen`）再落到字节 token。
    """
    offenders: list[str] = []
    for path in _python_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ANTI_PATTERN.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{number}: {stripped[:100]}")
    assert not offenders, (
        "发现 `decode(...).encode(...)` 式的字节回退——它会把只覆盖半个多字节字符的 "
        "token 换成 U+FFFD，且不报错。请改用 SubsetTokenizer.to_local / _bytes_of_qwen。\n"
        + "\n".join(offenders)
    )


def test_the_mapping_is_implemented_in_exactly_one_module() -> None:
    """同一套映射只许有一份实现——两份迟早不一致，而不一致是静默的。"""
    owners = [
        path.relative_to(REPO).as_posix()
        for path in _python_files()
        if "def to_local(" in path.read_text(encoding="utf-8")
        or "_bytes_of_qwen" in path.read_text(encoding="utf-8")
    ]
    assert owners == ["research/sel/lm/subset_tokenizer.py"], owners


QWEN_DIR = Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2")
CORPUS = Path("D:/AxoData/sel/lm/corpus/fineweb2_cmn/data/000_00000.parquet")


@pytest.mark.skipif(
    not (QWEN_DIR / "subset_32k.json").exists() or not CORPUS.exists(),
    reason="语料与子集在 D:/AxoData，不入库",
)
def test_real_corpus_lines_round_trip() -> None:
    """真实语料上编码→解码必须逐字相同——含生僻字与混排。"""
    pq = pytest.importorskip("pyarrow.parquet")

    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    bridge = SubsetTokenizer(QWEN_DIR)
    checked = 0
    for batch in pq.ParquetFile(CORPUS).iter_batches(batch_size=64, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if not text or len(text) > 400:
                continue
            assert bridge.decode(bridge.encode(text).ids) == text, text[:80]
            checked += 1
            if checked >= 200:
                return
    assert checked >= 50, f"样本太少（{checked}），这条守卫等于没跑"
