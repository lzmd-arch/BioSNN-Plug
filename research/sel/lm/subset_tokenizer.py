"""千问 32k 子集的分词桥：**本地 id（0–31999）↔ 千问 id**。

模型只认识子集下标，而分词器认识千问的全词表（151,936）。两边差一个映射，
这一步做错了不会报错——只会把每个 token 解成另一个字。所以它单独成一个模块，
并且有守卫。

子集外的千问 token 走**字节回退**：拆成 UTF-8 字节，每个字节映射到子集里对应的
字节 token（`build_qwen_subset.py` 会保证 256 个字节 token 全在子集里）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 式的字节→可见 unicode 映射；千问用的就是这套。"""
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, [chr(c) for c in chars], strict=True))


def subset_fingerprint(path: Path) -> str:
    """子集文件的指纹（前 16 位）。

    **模型的 id 空间由它定义**：换了一份 `subset_32k.json`，同一个整数就指向另一个
    千问 token，读出的权重整块错位，而且不报错。所以它必须跟着检查点走。
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def verify_subset(expected: str, qwen_dir: Path) -> None:
    """检查点记的子集指纹与这份子集目录是否一致。不一致就拒绝启动。

    ``expected`` 为空表示旧检查点没有记——那只能告警，不能假装校验过。
    """
    if not expected:
        print(
            "WARN：检查点里没有 subset_sha256（旧检查点），无法确认 --qwen 是否配套",
        )
        return
    actual = subset_fingerprint(Path(qwen_dir) / "subset_32k.json")
    if actual != expected:
        raise SystemExit(
            f"子集不配套：检查点记的是 {expected}，--qwen 指向的是 {actual}。"
            "同一个整数在两边指向不同的千问 token，输出会整块错位且不报错，因此拒绝启动。"
        )


class SubsetTokenizer:
    """对外与 ``ByteLevelBPETokenizer`` 同形：``encode(text).ids`` 与 ``decode(ids)``。

    ``encode`` 返回的是**本地子集 id**，``decode`` 收的也是本地 id——服务端不需要
    知道映射存在。
    """

    def __init__(self, qwen_dir: Path):
        from tokenizers import ByteLevelBPETokenizer

        root = Path(qwen_dir)
        self.ids: list[int] = json.loads((root / "subset_32k.json").read_text(encoding="utf-8"))[
            "ids"
        ]
        self._to_qwen = dict(enumerate(self.ids))
        self._from_qwen = {qwen: local for local, qwen in enumerate(self.ids)}
        self._tokenizer = ByteLevelBPETokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
        vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        to_byte = bytes_to_unicode()
        #: 256 个单字节 token 在子集里的下标；子集外的 token 靠它落地
        self.byte_local = [self._from_qwen[vocab[to_byte[b]]] for b in range(256)]
        #: 反向表：本地下标 → 它是哪个字节（不是字节 token 的就不在表里）
        self._byte_of_local = {local: byte for byte, local in enumerate(self.byte_local)}
        #: 每个千问 token 自己的字节。词表键是"字节→可见字符"的映射结果，
        #: 反过来查就能精确还原字节（`decode` 全靠它）。
        inverse = {visible: byte for byte, visible in to_byte.items()}
        self._bytes_of_qwen = {
            qwen_id: bytes(inverse[ch] for ch in text if ch in inverse)
            for text, qwen_id in vocab.items()
        }
        self.vocab_size = len(self.ids)

    def to_local(self, qwen_ids: list[int]) -> list[int]:
        """千问 id → 本地下标。子集外的走**原始字节**回退。"""
        local: list[int] = []
        for qwen_id in qwen_ids:
            slot = self._from_qwen.get(qwen_id)
            if slot is None:
                # **用原始字节，不要 decode 再 encode**：子集外的 token 常常只覆盖
                # 一个多字节字符的一部分（如 `b" "`），先 decode 会得到
                # 替换字符，再 encode 就把原本的字节永久换掉了。
                local.extend(self.byte_local[byte] for byte in self._bytes_of_qwen[qwen_id])
            else:
                local.append(slot)
        return local

    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=self.to_local(self._tokenizer.encode(text).ids))

    def decode(self, local_ids: list[int]) -> str:
        """本地下标 → 文本。**按字节拼**，不逐 token 交给分词器。

        为什么必须这样：千问的词表里有**只覆盖多字节字符一部分**的 token
        （例如「子」会被切成 `53497` + 单字节 `0x90`）。逐个 token 解码时前者是非法
        UTF-8，分词器各自吐一个 ``�``——实测拿到的是 ``' �集'`` 而不是 `' 子集'`。
        把每个 token 还原成它自己的字节、攒到最后一起解，就没有这个问题。
        """
        buffer = bytearray()
        for local in local_ids:
            buffer.extend(self._bytes_of_qwen[self._to_qwen[int(local)]])
        return buffer.decode("utf-8", "replace")
