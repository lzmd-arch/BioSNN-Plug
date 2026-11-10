"""守卫：`train` 的合并序列可复现，`encode` 与训练时的切分一致。"""

from __future__ import annotations

from research.sel.lm.bpe import encode, train

BASE = 0xE000


def test_train_is_deterministic() -> None:
    text = "abcabcabcabc" * 3
    assert train(text, 10, base=BASE) == train(text, 10, base=BASE)


def test_train_produces_requested_merges() -> None:
    text = "ababababababab"
    merges = train(text, 5, base=BASE)
    assert len(merges) == 5


def test_encode_uses_the_learned_merges() -> None:
    text = "abababababab"
    merges = train(text, 5, base=BASE)
    ids = encode(list(text), merges, BASE)
    assert len(ids) < len(text), "编码没有压缩"
    assert all(isinstance(i, int) for i in ids)
