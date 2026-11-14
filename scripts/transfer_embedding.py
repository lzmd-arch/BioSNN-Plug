"""把千问的词嵌入**几何**搬到我们 BPE 词表上，用作 `emb(V,d)` 的初始化。

只搬几何、不搬权重：我们的模型与 transformer 没有同构层，唯一能对上的是
"字词在语义空间里怎么摆"这件事。

流程：GGUF 里的 `token_embd` → 按我们的 BPE 单元查表（GPT-2 字节映射）→
未命中的按最长前缀拆开取平均 → 去中心 PCA 降到 d 维 → 缩放到与 `input_std` 同量级。

用法::

    uv run --with gguf python scripts/transfer_embedding.py --dim 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_GGUF = Path("D:/Projects/ollama-models/import/Qwen3-4B-Q4_K_M.gguf")
DEFAULT_BPE = Path("D:/AxoData/sel/lm/corpus/skypile_tok/bpe")
DEFAULT_OUT = Path("D:/AxoData/sel/lm/corpus/skypile_tok/emb_init.pt")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 式的字节→可见 unicode 映射；Qwen 的词表用的就是这套编码。"""
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="千问词嵌入几何 -> 我们的 emb 初始化")
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--bpe", type=Path, default=DEFAULT_BPE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--std", type=float, default=0.12, help="与 TokenEProp 的 input_std 一致")
    args = ap.parse_args(argv)

    import gguf
    from tokenizers import ByteLevelBPETokenizer

    reader = gguf.GGUFReader(str(args.gguf))
    field = reader.get_field("tokenizer.ggml.tokens")
    qwen_tokens = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]
    qwen_index = {t: i for i, t in enumerate(qwen_tokens)}
    print(f"千问词表 {len(qwen_tokens):,}", flush=True)

    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    print(f"token_embd {tuple(tensor.shape)} type={tensor.tensor_type}", flush=True)
    table = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    table = np.asarray(table, dtype=np.float32).reshape(tensor.shape[1], tensor.shape[0])
    print(f"反量化完成 {table.shape} {table.nbytes / 1e9:.2f} GB", flush=True)

    tokenizer = ByteLevelBPETokenizer(str(args.bpe / "vocab.json"), str(args.bpe / "merges.txt"))
    vocab = json.loads((args.bpe / "vocab.json").read_text(encoding="utf-8"))
    mapping = bytes_to_unicode()

    def keys_for(text: str) -> list[str]:
        """最长前缀贪心拆成若干千问键；整串命中就直接一个。"""
        raw = text.encode("utf-8")
        whole = "".join(mapping[b] for b in raw)
        if whole in qwen_index:
            return [whole]
        out: list[str] = []
        start = 0
        while start < len(raw):
            for end in range(len(raw), start, -1):
                piece = "".join(mapping[b] for b in raw[start:end])
                if piece in qwen_index:
                    out.append(piece)
                    start = end
                    break
            else:
                start += 1  # 认不出的字节直接跳过
        return out

    size = len(vocab)
    rows = np.zeros((size, table.shape[1]), dtype=np.float32)
    hit = miss = 0
    for token, index in vocab.items():
        text = tokenizer.decode([tokenizer.token_to_id(token)])
        keys = keys_for(text)
        ids = [qwen_index[k] for k in keys if k in qwen_index]
        if len(keys) == 1 and ids:
            hit += 1
        else:
            miss += 1
        if ids:
            rows[index] = table[ids].mean(axis=0)
    print(f"整串命中 {hit:,} / 拆分 {miss:,}", flush=True)

    used = np.abs(rows).sum(axis=1) > 0
    print(f"有向量 {int(used.sum()):,} / {size:,}", flush=True)
    mean = rows[used].mean(axis=0)
    rows -= mean
    matrix = torch.from_numpy(rows)
    # 去中心 PCA：SVD 取前 dim 个右奇异向量
    _, _, basis = torch.pca_lowrank(matrix, q=args.dim, center=False, niter=4)
    projected = matrix @ basis
    scale = projected[used].std().item()
    projected = projected * (args.std / max(scale, 1e-6))
    projected[~torch.from_numpy(used)] = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"emb": projected.to(torch.float32), "dim": args.dim, "std": args.std}, args.out)
    print(
        f"写入 {args.out}：{tuple(projected.shape)}，std {projected[used].std():.4f}，"
        f"零向量 {int((~torch.from_numpy(used)).sum())} 行",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
