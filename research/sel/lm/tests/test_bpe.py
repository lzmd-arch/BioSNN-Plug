"""守卫：Rust 分词器可训练、可复现，压缩比为正。"""

from __future__ import annotations

from pathlib import Path

from research.sel.lm.bpe import compression, encode_file, train_tokenizer

SAMPLE = [
    "今天天气不错，我们出去走走吧。",
    "这家餐厅的环境很好，但是价格有点贵。",
    "你觉得这部电影怎么样？我觉得一般般。",
] * 40


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(SAMPLE), encoding="utf-8")
    return path


def test_trainer_and_encoder(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    tok = train_tokenizer([src], tmp_path / "tok", vocab_size=600, min_frequency=1)
    stats = encode_file(tok, src, tmp_path / "ids.txt")
    assert stats["tokens"] > 0
    assert stats["tokens_per_char"] < 1.0, "压缩比应小于 1（BPE 若没合并会等于 1）"


def test_compression_helper() -> None:
    out = compression(1000, 400)
    assert out["tokens_per_char"] == 0.4
    assert out["chars_per_token"] == 2.5
    assert out["saved_pct"] == 60.0
