"""在**带传感器的合成任务**上训练小模型——这是"局部学习 + 独占输入"的最小闭环。

任务：序列 = 「前缀 token ... → 数字 token」，数字由传感器向量决定（温度→个位）。
三臂只差传感器注入方式：`zero`（不给）/ `add`（加性）/ `gate`（门控）。

判据：`add` 与 `gate` 的 top-1 必须显著高于 `zero`；两者之间再分高下。
**训练与推理全程无 autograd**（与主网络同一条纪律）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.sel.lm.hierarchical import build_huffman_paths, path_probabilities
from research.sel.lm.model import TokenEProp

PREFIX = ["当前温度"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("zero", "add", "gate"), required=True)
    ap.add_argument("--subset", type=Path, default=Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2/subset_32k.json"))
    ap.add_argument("--counts", type=Path, default=Path("sweep_results/lm_10m/counts_32000.npy"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/sensor_task"))
    args = ap.parse_args(argv)

    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    tok = SubsetTokenizer(args.subset.parent)
    counts = torch.from_numpy(np.load(args.counts)).float()
    paths = build_huffman_paths(counts)
    vocab = 32000
    n_rec = 156
    model = TokenEProp(
        vocab, n_rec, recurrence="none", readout_taus=(2.0, 128.0), tree=paths,
        tree_signal_scale=3.0, tree_packed=True, tree_init_weight=False,
        embed_rank=256, sensor_dim=16, lr_in=1e-3, lr_out=0.5,
        generator=torch.Generator().manual_seed(0), device=torch.device("cpu"),
    )
    model.sensor_mode = "add" if args.arm != "gate" else "gate"
    digit_ids = [tok.encode(str(d)).ids[0] for d in range(10)]
    prefix_ids = tok.encode(PREFIX[0]).ids

    rng = np.random.default_rng(0)
    dev = model.device
    t0 = time.time()
    for step in range(1, args.steps + 1):
        temp = rng.integers(0, 10, size=args.batch)
        sensors = torch.zeros(args.batch, 16)
        sensors[:, 0] = torch.tensor(temp / 9.0, dtype=torch.float32)
        model.sensors = None if args.arm == "zero" else sensors.to(dev)
        trace = model.new_trace(args.batch)
        seq = prefix_ids + [digit_ids[0]]
        for i in range(len(seq) - 1):
            tok_t = torch.full((args.batch,), seq[i], dtype=torch.long, device=dev)
            tgt = torch.tensor([digit_ids[int(t)] for t in temp], device=dev) if i == len(seq) - 2 else torch.full((args.batch,), seq[i + 1], dtype=torch.long, device=dev)
            model.step(tok_t, tgt, trace, update=True, all_valid=True)
        model.sensors = None
        if step % 250 == 0:
            # 评测：同一任务、新样本
            hit = n = 0
            with torch.no_grad():
                for _ in range(20):
                    t2 = rng.integers(0, 10, size=args.batch)
                    s2 = torch.zeros(args.batch, 16); s2[:, 0] = torch.tensor(t2 / 9.0)
                    model.sensors = None if args.arm == "zero" else s2.to(dev)
                    tr = model.new_trace(args.batch)
                    for i in range(len(prefix_ids) - 1):
                        model.step(torch.full((args.batch,), prefix_ids[i], dtype=torch.long, device=dev),
                                   torch.full((args.batch,), prefix_ids[i + 1], dtype=torch.long, device=dev),
                                   tr, update=False, all_valid=True)
                    model.sensors = None
                    # 用树上 10 个数字 token 的路径概率当分类器
                    full = path_probabilities(model._features(tr), model.tree.weight, model.tree.paths)
                    probs = full[:, torch.tensor(digit_ids, device=dev)]
                    pred = probs.argmax(1).cpu().numpy()
                    hit += int((pred == t2).sum()); n += args.batch
            print(f"  [{args.arm}] {step} 步  评测 top-1 {hit/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    out = {"arm": args.arm, "steps": args.steps, "batch": args.batch}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.arm}.json").write_text(json.dumps(out) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
