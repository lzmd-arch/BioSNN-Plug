"""BPE 分词：**Rust 后端**（HuggingFace `tokenizers`）。

为什么换：自己写的增量版记账有 bug（第 8 条合并即分叉，未采用）；而 `str.replace` 版虽精确，
仍是 O(n·V) —— 对 ~2B 字符的全语料学合并要 36 小时。Rust 实现在同一规模下是分钟级。

**这也是 2 天训 1B 的硬前提**：字符级要 31 TFLOPS（贴着峰值，不可达），BPE 后降到 17.4。
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

__all__ = ["compression", "encode_file", "train_tokenizer"]


def train_tokenizer(
    files: list[Path], out_dir: Path, *, vocab_size: int = 32000, min_frequency: int = 2
) -> ByteLevelBPETokenizer:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(f) for f in files],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<s>", "</s>"],
    )
    tokenizer.save_model(str(out_dir))
    return tokenizer


def encode_file(tokenizer: ByteLevelBPETokenizer, src: Path, dst: Path) -> dict:
    """把一行一段的语料编码成一行一个 id 列表（JSONL），并返回压缩统计。"""
    lines = src.read_text(encoding="utf-8").splitlines()
    chars = sum(len(t) for t in lines)
    tokens = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as handle:
        for text in lines:
            ids = tokenizer.encode(text).ids
            tokens += len(ids)
            handle.write(",".join(str(i) for i in ids) + chr(10))
    return compression(chars, tokens)


def compression(chars: int, tokens: int) -> dict:
    ratio = tokens / max(chars, 1)
    return {
        "chars": chars,
        "tokens": tokens,
        "tokens_per_char": round(ratio, 4),
        "chars_per_token": round(1 / ratio, 3) if ratio else None,
        "saved_pct": round((1 - ratio) * 100, 1),
    }
