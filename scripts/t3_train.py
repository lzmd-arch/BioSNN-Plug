"""T-3：在 8 GB 上用 **NF4 4-bit 底座 + 只训 `lm_head`** 改千问权重。

**不是 LoRA**：不加低秩适配器，训的是词表上每个 token 的输出投影。
预注册：`research/sel/prereg-v2-t3-weight-training.md`
判据：ECE < 0.0425（不引入温度）/ 固定 T=1 的平均 log p +0.05 / top-1 不掉。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "D:/AxoData/models/hf/Qwen3-4B"
CORPUS = Path("D:/AxoData/sel/lm/corpus/fineweb2_u16/corpus.u16")
TOKROOT = Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2")
OUT = Path("sweep_results/t3")


def load_text(n_chars: int, start_tok: int) -> str:
    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    st = SubsetTokenizer(TOKROOT)
    a = np.memmap(CORPUS, dtype=np.uint16, mode="r")
    return st.decode(a[start_tok : start_tok + n_chars * 2].tolist())


def chunks(tok, text: str, n: int, length: int) -> torch.Tensor:
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n * length + 1
    if len(ids) < need:
        raise SystemExit(f"语料不够：{len(ids)} < {need}")
    blocks = ids[: need - 1].reshape(n, length)
    return blocks


@torch.no_grad()
def evaluate(model, blocks: torch.Tensor, dev: str) -> dict:
    model.eval()
    logps, hits = [], []
    for i in range(0, len(blocks), 4):
        x = blocks[i : i + 4].to(dev)
        out = model(x, use_cache=False).logits[:, :-1].float()
        tgt = x[:, 1:]
        lp = torch.log_softmax(out, dim=-1)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        logps.append(tok_lp.reshape(-1).cpu())
        hits.append((out.argmax(-1) == tgt).float().reshape(-1).cpu())
    lp = torch.cat(logps)
    hit = torch.cat(hits)
    # ECE：按 top-1 的置信度分箱
    conf = torch.exp(lp)
    bins = torch.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        m = (conf >= bins[b]) & (conf < bins[b + 1])
        if m.sum() == 0:
            continue
        ece += float(m.float().mean()) * abs(float(conf[m].mean()) - float(hit[m].mean()))
    return {
        "mean_logp": float(lp.mean()),
        "top1": float(hit.mean()),
        "ece": ece,
        "n_tokens": int(lp.numel()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-blocks", type=int, default=48)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    qnt = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=qnt, device_map={"": 0}, trust_remote_code=True
    )
    for p in model.parameters():
        p.requires_grad = False
    head = model.lm_head
    head.weight.requires_grad = True  # **唯一可训的东西**
    if head.bias is not None:
        head.bias.requires_grad = True
    print(
        f"可训参数 {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M",
        flush=True,
    )

    train_text = load_text(400_000, 500_000)
    held_text = load_text(60_000, 9_000_000)
    torch.manual_seed(0)
    tr = chunks(tok, train_text, args.steps * args.batch * args.accum, args.length)
    he = chunks(tok, held_text, args.eval_blocks, args.length)
    print(f"训练块 {tr.shape} / 评测块 {he.shape}", flush=True)

    before = evaluate(model, he, dev)
    print("训练前", json.dumps(before, ensure_ascii=False), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    t0 = time.time()
    k = 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            x = tr[k].unsqueeze(0).to(dev)
            k += 1
            out = model(x, use_cache=False).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(
                out.reshape(-1, out.shape[-1]), x[:, 1:].reshape(-1)
            )
            (loss / args.accum).backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step} loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)

    after = evaluate(model, he, dev)
    out = {
        "steps": args.steps,
        "lr": args.lr,
        "before": before,
        "after": after,
        "d_logp": after["mean_logp"] - before["mean_logp"],
        "d_top1": after["top1"] - before["top1"],
        "d_ece": after["ece"] - before["ece"],
    }
    (OUT / "train.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
