"""千问 0.5B vs 淘宝 MNN-Pocket-0.3B：按**我们的用途**比，不按榜单比。

三条对我们的用途真正相关的量：

1. **中文质量**：留出文本上的 **bits/char**（按字符归一，跨分词器可比）；
2. **可训性**：8 GB 上的训练吞吐与峰值显存；
3. **体积/词表**：参数量、层数、词表大小（词表直接决定条件偏置的成本）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = Path("D:/AxoData/sel/lm/corpus/fineweb2_u16/corpus.u16")
TOKROOT = Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2")
MODELS = {
    "qwen2.5-0.5b": "D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct",
    "mnn-pocket-0.3b": "D:/AxoData/models/hf/MNN-Pocket-0.3B",
}


def corpus_text(n: int = 4000) -> str:
    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    a = np.memmap(TEXT, dtype=np.uint16, mode="r")
    return SubsetTokenizer(TOKROOT).decode(a[7_000_000 : 7_000_000 + n * 2].tolist())


@torch.no_grad()
def bits_per_char(model, tok, text: str, dev: str = "cuda") -> float:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    nll = 0.0
    for i in range(0, ids.shape[1] - 257, 256):
        chunk = ids[:, i : i + 257]
        out = model(chunk[:, :-1])
        lp = torch.log_softmax(out.logits.float(), dim=-1)
        nll += float(-lp.gather(-1, chunk[:, 1:].unsqueeze(-1)).sum())
    return nll / max(len(text), 1) / np.log(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=3000)
    args = ap.parse_args()
    text = corpus_text(args.chars)
    rows = []
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        cfg = model.config
        bpc = bits_per_char(model, tok, text)
        # 训练吞吐：5 步，batch 2 × 256
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        x = torch.randint(0, 1000, (2, 256)).cuda()
        t0 = time.time()
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            model(x, labels=x).logits.float().mean().backward()
            opt.step()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append(
            {
                "model": name,
                "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                "layers": cfg.num_hidden_layers,
                "hidden": cfg.hidden_size,
                "vocab": cfg.vocab_size,
                "bits_per_char": round(bpc, 3),
                "train_tok_per_s": round(5 * 512 / dt, 0),
                "peak_GiB": round(peak, 2),
            }
        )
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    Path("sweep_results/model_pick.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
