"""把语料重编成**千问 32k 子集**的 id 流（uint16），供 C-1 与将来的 logits 融合使用。

两个要点：

1. **存储用重映射后的本地 id**：千问的 id 最大到 151,935，装不进 uint16。
   子集里的 32,000 个 token 重映射到 0..31,999，映射表落盘，融合时用得上；
2. **子集外的 token 按字节拆**：千问没有 `<0xNN>` 形式的字节 token（和 GPT-2 一样），
   每个字节就是词表里的一个单字节 token —— 它们必然在子集里，所以拆分总能落地。

用法::

    uv run --with gguf python scripts/build_qwen_subset_corpus.py --chars 200000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

QWEN = Path("D:/AxoData/sel/lm/corpus/qwen_tok")
CORPUS = Path("D:/AxoData/sel/lm/corpus/skypile/data")
OUT = Path("D:/AxoData/sel/lm/corpus/qwen32k")


def bytes_to_unicode() -> dict[int, str]:
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
    ap = argparse.ArgumentParser(description="语料 -> 千问 32k 子集 id 流（uint16）")
    ap.add_argument("--qwen", type=Path, default=QWEN)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--chars", type=int, default=200_000_000)
    ap.add_argument("--holdout-chars", type=int, default=5_000_000)
    args = ap.parse_args(argv)

    from tokenizers import ByteLevelBPETokenizer

    payload = json.loads((args.qwen / "subset_32k.json").read_text(encoding="utf-8"))
    subset_ids: list[int] = payload["ids"]
    remap = {qwen_id: local for local, qwen_id in enumerate(subset_ids)}

    tokenizer = ByteLevelBPETokenizer(str(args.qwen / "vocab.json"), str(args.qwen / "merges.txt"))
    vocab = json.loads((args.qwen / "vocab.json").read_text(encoding="utf-8"))
    b2u = bytes_to_unicode()
    byte_ids = {index: vocab[b2u[index]] for index in range(256)}
    # 检查要拿**字符串**去查，不能拿 id 去查（一次写成 `token not in vocab`，
    # 而 token 是 int、vocab 的键是 str ⇒ 256 个全被误报成"缺失"）。
    missing = [b for b in range(256) if b2u[b] not in vocab]
    if missing:
        raise SystemExit(f"单字节 token 缺失：{missing[:5]}")

    absent = [b for b in range(256) if byte_ids[b] not in remap]
    if absent:
        raise SystemExit(f"字节 token 不在 32k 子集里（{len(absent)} 个），回退无法落地：{absent[:5]}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "id_map.json").write_text(
        json.dumps({"qwen_ids": subset_ids, "note": "local id = 列表下标"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    files = sorted(args.corpus.glob("*.jsonl"))
    written = chars = tokens = out_of_subset = 0
    budget = args.chars + args.holdout_chars
    train_path = args.out / "corpus.u16"
    holdout_path = args.out / "holdout.u16"
    handle = train_path.open("wb")
    holdout = holdout_path.open("wb")
    try:
        for text in texts(files):
            ids = tokenizer.encode(text).ids
            local: list[int] = []
            for qwen_id in ids:
                slot = remap.get(qwen_id)
                if slot is not None:
                    local.append(slot)
                else:
                    out_of_subset += 1
                    piece = tokenizer.decode([qwen_id]).encode("utf-8", "ignore")
                    for byte in piece:
                        local.append(remap[byte_ids[byte]])
            np.asarray(local, dtype=np.uint16).tofile(handle if chars < args.chars else holdout)
            written += len(local)
            chars += len(text)
            tokens += len(ids)
            if written and written % 20_000_000 < len(local):
                print(f"  {written / 1e6:.0f}M token（{chars / 1e9:.2f}B 字）", flush=True)
            if chars >= budget:
                break
    finally:
        handle.close()
        holdout.close()

    stats = {
        "chars": chars,
        "qwen_tokens": tokens,
        "subset_tokens": written,
        "chars_per_token_qwen": round(chars / max(tokens, 1), 4),
        "chars_per_token_subset": round(chars / max(written, 1), 4),
        "out_of_subset_rate": round(out_of_subset / max(tokens, 1), 5),
        "holdout_chars": args.holdout_chars,
        "holdout_bytes": holdout_path.stat().st_size if holdout_path.exists() else 0,
        "subset_size": len(subset_ids),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
