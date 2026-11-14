"""守卫：千问 32k 子集的分词桥。

这条链最危险的失败方式是**不报错**——把本地下标当成千问 id 去解码，每个 token 都
解成另一个字，服务照样返回 200。所以下面几条断言钉的是"映射是双向一致的"，
而不是"能不能跑"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

QWEN_DIR = Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2")

pytestmark = pytest.mark.skipif(
    not (QWEN_DIR / "subset_32k.json").exists(),
    reason="子集目录不存在（数据在 D:/AxoData，不入库）",
)


@pytest.fixture(scope="module")
def tok():
    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    return SubsetTokenizer(QWEN_DIR)


def test_local_ids_stay_inside_the_subset(tok) -> None:
    """编码产物必须落在 0–31999，否则模型的嵌入表会越界（或者静默取错行）。"""
    ids = tok.encode("北岸一号站的采集箱每五分钟往岸上推一次数据包，推失败会重试三次。").ids
    assert ids
    assert all(0 <= index < tok.vocab_size for index in ids)


def test_every_byte_token_is_reachable(tok) -> None:
    """256 个单字节 token 必须全部在子集里——它们是子集外 token 的唯一退路。"""
    assert len(tok.byte_local) == 256
    assert all(0 <= index < tok.vocab_size for index in tok.byte_local)


def test_round_trip_is_lossless_on_in_vocabulary_text(tok) -> None:
    text = "第一个版本用千问的三万二千个 token 子集。"
    assert tok.decode(tok.encode(text).ids) == text


def test_decoding_uses_qwen_and_not_the_raw_index(tok) -> None:
    """**这条是防"把本地下标当千问 id"的**：两边同一位置的字必须不同，否则测不出来。"""
    ids = tok.encode("的记忆").ids
    decoded = tok.decode(ids)
    raw = tok._tokenizer.decode(ids)  # 故意用本地 id 当千问 id
    assert decoded == "的记忆"
    assert raw != decoded, "本地 id 与千问 id 恰好相同，这条守卫失去意义，需另选样本"


def test_oov_falls_back_to_bytes_and_still_decodes(tok) -> None:
    """子集外的字走字节回退：编码不报错，解码回来还是同一个字符。"""
    rare = "𠀀"  # 扩展 B 区，几乎不可能在 32k 子集里
    ids = tok.encode(rare).ids
    assert ids
    assert all(0 <= index < tok.vocab_size for index in ids)
    assert tok.decode(ids) == rare


def _fake_subset(directory: Path, ids: list[int]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "subset_32k.json"
    target.write_text(json.dumps({"ids": ids}), encoding="utf-8")
    return target


def test_fingerprint_tracks_content(tmp_path) -> None:
    """**换一份子集，指纹必须变**——否则服务端的校验是摆设。"""
    from research.sel.lm.subset_tokenizer import subset_fingerprint

    a = _fake_subset(tmp_path / "a", [1, 2, 3])
    b = _fake_subset(tmp_path / "b", [1, 2, 4])
    assert subset_fingerprint(a) != subset_fingerprint(b)
    assert subset_fingerprint(a) == subset_fingerprint(a)


def test_verify_subset_accepts_a_match(tmp_path) -> None:
    from research.sel.lm.subset_tokenizer import subset_fingerprint, verify_subset

    path = _fake_subset(tmp_path / "q", [5, 6, 7])
    verify_subset(subset_fingerprint(path), tmp_path / "q")  # 不抛就算过


def test_verify_subset_refuses_a_mismatch(tmp_path) -> None:
    """检查点是用 A 子集训的，服务端拿了 B 子集 ⇒ **必须拒绝启动**。

    放过去的后果不是报错，而是每个 id 都指向另一个字，输出全是错的但看起来正常。
    """
    from research.sel.lm.subset_tokenizer import subset_fingerprint, verify_subset

    a = _fake_subset(tmp_path / "a", [1, 2, 3])
    _fake_subset(tmp_path / "b", [9, 9, 9])
    with pytest.raises(SystemExit):
        verify_subset(subset_fingerprint(a), tmp_path / "b")


def test_verify_subset_only_warns_for_old_checkpoints(tmp_path) -> None:
    """旧检查点没记指纹时**不能假装校验过**，但也别把服务拦死——只告警。"""
    from research.sel.lm.subset_tokenizer import verify_subset

    _fake_subset(tmp_path / "q", [1])
    verify_subset("", tmp_path / "q")  # 不抛


def test_partial_multibyte_tokens_keep_their_bytes(tok) -> None:
    """**这次事故的直接回归测试。**

    子集外、且字节单独不是合法 UTF-8 的 token（只覆盖某个汉字的一部分），
    回退必须落到**它自己的字节**上。旧写法会得到 `EF BF BD`（替换字符）的三份，
    长度和内容都不对，而且不报错。
    """
    checked = 0
    for qwen_id, piece in tok._bytes_of_qwen.items():
        if qwen_id in tok._from_qwen:
            continue
        try:
            piece.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            continue
        assert tok.to_local([qwen_id]) == [tok.byte_local[byte] for byte in piece], piece
        checked += 1
        if checked >= 300:
            break
    assert checked >= 50, f"没找到足够的样本（{checked}），这条守卫等于没跑"
