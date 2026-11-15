"""把人格**训进权重**：0.5B 全参微调（不是 LoRA），只在 assistant 段计算损失。"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.chatfmt import render

MODEL = "D:/AxoData/models/hf/Qwen2.5-0.5B-Instruct"


def set_seed(seed: int) -> None:
    """**每个 run 必须播种**：不播的话"三种子"只是"三次跑"，拿不到可复现的区间。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


OUT = Path("sweep_results/persona_train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=OUT / "data.jsonl")
    ap.add_argument("--base", default=MODEL, help="底座模型目录（④ 换底座的入口）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save", type=Path, default=OUT / "ckpt")
    ap.add_argument("--resume", default=None, help="从这个检查点**补训**（不是从底座重来）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x]

    ids, labs = [], []
    for r in rows:
        prompt = render(tok, r["messages"][:1], add_generation_prompt=True)
        full = render(tok, r["messages"], add_generation_prompt=False)
        pi = tok(prompt, add_special_tokens=False).input_ids
        fi = tok(full, add_special_tokens=False).input_ids[: args.max_len]
        lab = [-100] * min(len(pi), len(fi)) + fi[min(len(pi), len(fi)) :]
        ids.append(torch.tensor(fi))
        labs.append(torch.tensor(lab))

    src = args.resume or args.base
    print(f"起点：{'补训自 ' + src if args.resume else '底座 ' + src}", flush=True)
    print(f"种子：{args.seed}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0, ntok, step = time.time(), 0, 0
    for _ in range(args.epochs):
        order = torch.randperm(len(ids))
        for i in range(0, len(order) - args.batch, args.batch):
            idx = order[i : i + args.batch]
            n_len = max(int(ids[j].numel()) for j in idx)
            x = torch.full((len(idx), n_len), tok.pad_token_id, dtype=torch.long)
            y = torch.full((len(idx), n_len), -100, dtype=torch.long)
            for k, j in enumerate(idx):
                x[k, : ids[j].numel()] = ids[j]
                y[k, : labs[j].numel()] = labs[j]
            x, y = x.cuda(), y.cuda()
            opt.zero_grad(set_to_none=True)
            loss = model(x, labels=y).loss
            loss.backward()
            opt.step()
            step += 1
            ntok += int((y != -100).sum())
            if step % 25 == 0:
                print(
                    f"  step {step} loss {float(loss):.4f} "
                    f"{time.time() - t0:.0f}s {ntok / (time.time() - t0):.0f} tok/s",
                    flush=True,
                )
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break
    dt = time.time() - t0
    args.save.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    out = {
        "base": args.base,
        "seed": args.seed,
        "steps": step,
        "seconds": round(dt, 1),
        "tokens": ntok,
        "tok_per_s": round(ntok / dt, 1),
        "epochs": args.epochs,
        "lr": args.lr,
    }
    (OUT / "train_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
