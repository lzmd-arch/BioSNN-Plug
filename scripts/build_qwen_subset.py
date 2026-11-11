"""把千问的分词器从 GGUF 里取出来，并建一个 **32k 子集**。

为什么要子集而不是换词表：**改千问的词表会把千问弄坏，而我们没有修它的办法**
（重造嵌入等于让 40 层看到从未见过的输入，修复需要反向传播）。

子集相反：**千问一个字都不动**，我们只在它自己的 token 空间里挑 32,000 个常用的用。
于是读出宽度保住了（20,833），而且两边的分布落在同一个空间里 —— logits 能直接相加。

本脚本产出：
1. `qwen_tok/{vocab.json,merges.txt}` —— 从 GGUF 还原的分词器；
2. `subset_32k.json` —— 32,000 个高频 token id；
3. 实测的 **chars/token**（全量 vs 子集+字节回退），与现役 BPE 的 1.685 直接可比。

用法::

    uv run --with gguf python scripts/build_qwen_subset.py --sample-chars 200000000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_GGUF = Path("D:/Projects/ollama-models/import/Qwen3-4B-Q4_K_M.gguf")
OUT = Path("D:/AxoData/sel/lm/corpus/qwen_tok")
CORPUS = Path("D:/AxoData/sel/lm/corpus/skypile/data")


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


def extract(gguf_path: Path, out: Path):
    """从 GGUF 还原 `ByteLevelBPETokenizer` 需要的两份文件。"""
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    # **`parts` 是分片、`data` 才是条目**：151,936 个 token 摊在 303,877 个 parts 里。
    # 直接用 parts 会把词表抽成 30 万条且全部错位——自检里表现为"字节回退 0 个"这种假数。
    field_tokens = reader.get_field("tokenizer.ggml.tokens")
    tokens = [bytes(field_tokens.parts[i]).decode("utf-8", "replace") for i in field_tokens.data]
    field = reader.get_field("tokenizer.ggml.merges")
    merges = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]
    out.mkdir(parents=True, exist_ok=True)
    vocab = {token: index for index, token in enumerate(tokens)}
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges) + "\n", encoding="utf-8", newline="\n")
    print(f"分词器：{len(tokens):,} 个 token，{len(merges):,} 条合并", flush=True)
    return len(tokens)


def texts(files: list[Path]):
    for path in files:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text") or row.get("content") or ""
                if text:
                    yield text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="还原千问分词器并建 32k 子集")
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--sample-chars", type=int, default=200_000_000)
    ap.add_argument("--size", type=int, default=32000)
    args = ap.parse_args(argv)

    from tokenizers import ByteLevelBPETokenizer

    extract(args.gguf, args.out)
    tokenizer = ByteLevelBPETokenizer(str(args.out / "vocab.json"), str(args.out / "merges.txt"))
    probe = "北岸一号站的采集箱每五分钟往岸上推一次数据包，推失败会重试三次。"
    print(f"自检：{tokenizer.encode(probe).tokens}", flush=True)

    files = sorted(args.corpus.glob("*.jsonl"))[:6]
    counts: Counter[int] = Counter()
    chars = 0
    for text in texts(files):
        counts.update(tokenizer.encode(text).ids)
        chars += len(text)
        if chars >= args.sample_chars:
            break
    tokens = sum(counts.values())
    print(
        f"统计样本：{chars / 1e9:.2f}B 字 / {tokens / 1e9:.2f}B token"
        f" ⇒ 全量词表 {chars / tokens:.3f} 字/token",
        flush=True,
    )

    keep = [index for index, _ in counts.most_common(args.size)]
    # **256 个单字节 token 必须全在**：它们是子集外 token 的字节回退路径。
    # 实测 63 个控制字节（0x01–0x05 之类）落在 32k 之外，回退就无法落地。
    vocab_map = json.loads((args.out / "vocab.json").read_text(encoding="utf-8"))
    to_byte = bytes_to_unicode()
    must = [vocab_map[to_byte[b]] for b in range(256) if to_byte[b] in vocab_map]
    missing = [index for index in must if index not in set(keep)]
    if missing:
        keep = keep[: args.size - len(missing)] + missing
        print(f"强制保留 {len(missing)} 个字节 token（从尾部换入）", flush=True)
    subset = set(keep)
    print(
        f"取前 {len(keep):,} 个 token，覆盖样本的 "
        f"{sum(counts[i] for i in keep) / tokens * 100:.2f}%",
        flush=True,
    )

    # 子集 + 字节回退：不在子集里的 token 拆成字节级 token（它们几乎必然在子集里）
    byte_ids = [
        index
        for index, token in enumerate(
            json.loads((args.out / "vocab.json").read_text(encoding="utf-8"))
        )
        if len(token) == 6 and token.startswith("<0x")
    ]
    byte_set = set(byte_ids)
    fallback = len(byte_set - subset)
    print(f"字节回退 token {len(byte_set):,} 个，其中 {fallback:,} 个不在子集里", flush=True)

    kept_tokens = sum(counts[i] for i in keep)
    extra = 0
    for index, count in counts.items():
        if index in subset:
            continue
        token = tokenizer.decode([index])
        extra += count * max(len(token.encode("utf-8")), 1)
    restricted = kept_tokens + extra
    print(
        f"子集 + 字节回退：{chars / restricted:.3f} 字/token"
        f"（全量 {chars / tokens:.3f}，现役 BPE 1.685）",
        flush=True,
    )
    print(f"等效 token 数增加 {restricted / tokens - 1:.1%}", flush=True)

    (args.out / f"subset_{args.size // 1000}k.json").write_text(
        json.dumps(
            {
                "size": args.size,
                "ids": keep,
                "chars_per_token_full": round(chars / tokens, 4),
                "chars_per_token_subset": round(chars / restricted, 4),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {args.out / f'subset_{args.size // 1000}k.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
