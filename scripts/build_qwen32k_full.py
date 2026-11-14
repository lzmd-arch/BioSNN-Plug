"""把**全量语料**编成千问 32k 子集的 id 流（uint16），多进程按片并行。

与 `build_qwen_subset_corpus.py` 的区别只有一条：那个是单进程、只做样本；
这个是全量、按片并行，产物可以直接喂给 `run_1b`。

存储用**重映射后的本地 id**（千问 id 最大 151,935，装不进 uint16）。
映射表 `id_map.json` 与 `subset_32k.json` 一起构成"我们的 id ↔ 千问 id"的桥。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from corpus_io import iter_texts, shards

# 脚本以 `python scripts/xxx.py` 的方式跑时，sys.path[0] 是 scripts/，仓库根不在里面。
# 显式加进去，才能复用研究侧的 `subset_fingerprint`——**别在这里另写一份哈希实现**，
# 两份迟早不一致，而指纹不一致是静默的（就是这条链要防的那种错）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.sel.lm.subset_tokenizer import SubsetTokenizer, subset_fingerprint

QWEN = Path("D:/AxoData/sel/lm/corpus/qwen_tok")
CORPUS = Path("D:/AxoData/sel/lm/corpus/skypile/data")
OUT = Path("D:/AxoData/sel/lm/corpus/qwen32k_full")


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


def encode_one(job: tuple[str, str, str]) -> tuple[str, int, int]:
    src, dst, qwen_dir = job
    root = Path(qwen_dir)
    # **映射只有一份实现**：`SubsetTokenizer`。这里曾经自带一份，
    # 而且它的字节回退是 `decode([qwen_id]).encode(...)`——对只覆盖半个多字节字符的
    # token（如 b"\xa0\x80"）会先变成替换字符、再把原始字节永久换掉。
    # 那会让建出来的语料里约 1% 的位置是 U+FFFD 而不是真字，而且不报错。
    bridge = SubsetTokenizer(root)

    chars = 0
    local_tokens = 0
    with Path(dst).open("wb") as handle:
        for text in iter_texts(Path(src)):
            local = bridge.encode(text).ids
            np.asarray(local, dtype=np.uint16).tofile(handle)
            chars += len(text)
            local_tokens += len(local)
    return Path(src).name, chars, local_tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="全量语料 -> 千问 32k 子集 id 流（并行）")
    ap.add_argument("--qwen", type=Path, default=QWEN)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    args = ap.parse_args(argv)

    files = shards(args.corpus)
    if not files:
        raise SystemExit(f"没找到语料：{args.corpus}")
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(parts / (f.stem + ".u16")), str(args.qwen)) for f in files]
    print(f"{len(files)} 片，{args.workers} 进程", flush=True)

    started = time.time()
    chars = tokens = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, c, t) in enumerate(pool.map(encode_one, jobs), 1):
            chars += c
            tokens += t
            rate = tokens / max(time.time() - started, 1)
            print(
                f"  [{index}/{len(files)}] {name} 累计 {tokens / 1e9:.2f}B 本地 token"
                f"（{rate / 1e3:.0f}k tok/s）",
                flush=True,
            )

    target = args.out / "corpus.u16"
    with target.open("wb") as dst:
        for part in sorted(parts.glob("*.u16")):
            with part.open("rb") as src:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
            part.unlink()
    parts.rmdir()
    stats = {
        "chars": chars,
        "local_tokens": tokens,
        # **id 空间的身份**：这份流里的整数只有在同一份子集下才有意义。
        # 记下指纹，训练端 (`--subset`) 与服务端 (`--qwen`) 才校验得起来。
        "subset_dir": str(args.qwen),
        "subset_sha256": subset_fingerprint(args.qwen / "subset_32k.json"),
        "bin_bytes": target.stat().st_size,
        "files": len(files),
        "subset_size": 32000,
        "note": "本地 id 是 subset_32k.json 里的下标；融合时用 id_map 换回千问 id",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
