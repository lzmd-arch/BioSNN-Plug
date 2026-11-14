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
    ap.add_argument("--lr-sensor", type=float, default=0.05)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/sensor_task"))
    args = ap.parse_args(argv)

    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    tok = SubsetTokenizer(args.subset.parent)
    counts = torch.from_numpy(np.load(args.counts)).float()
    paths = build_huffman_paths(counts)
    vocab = 32000
    TEXT_N = 156
    POOL = 16            # **每个传感器模态一个独立池**，不占文本神经元
    n_rec = TEXT_N + POOL
    model = TokenEProp(
        vocab, n_rec, recurrence="none", readout_taus=(2.0, 128.0), tree=paths,
        tree_signal_scale=3.0, tree_packed=True, tree_init_weight=False,
        embed_rank=256, sensor_dim=16, lr_in=1e-3, lr_out=0.5,
        generator=torch.Generator().manual_seed(0), device=torch.device("cpu"),
    )
    # **功能分离**：文本只驱动前 TEXT_N 个神经元，传感器只驱动后 POOL 个。
    # 不做分离时，传感器的 1e-2 量级分量挤不进由 O(1) 文本电流决定的发放模式——
    # 这是今天六条排除之后剩下的那条解释。
    with torch.no_grad():
        # 文本→神经元的路径可能是 proj（低秩）、也可能是 w_in（稠密查找），两者都要挡住
        if getattr(model, "proj", None) is not None:
            model.proj[:, TEXT_N:] = 0.0
        if getattr(model, "w_in", None) is not None:
            model.w_in[:, TEXT_N:] = 0.0
        sp = torch.zeros(model.sensor_proj.shape)
        sp[:, TEXT_N:] = torch.randn(POOL, POOL, generator=torch.Generator().manual_seed(7)) * 3.0
        model.sensor_proj = sp                       # 传感器只写自己的池
    model.sensor_mode = "add" if args.arm != "gate" else "gate"
    model.lr_sensor = args.lr_sensor
    #: **非零初始化**：零初始化 + 比文本路径弱几个数量级的梯度 = 慢启动，
    #: 通道可能从来没被真正启动过。按 n_rec 的标准差给一个可比的初始尺度。
    if args.init_scale > 0:
        g = torch.Generator().manual_seed(7)
        model.sensor_proj = torch.randn(model.sensor_proj.shape, generator=g) * (
            args.init_scale / (model.n_rec ** 0.5)
        )
    digit_ids = [tok.encode(str(d)).ids[0] for d in range(10)]
    prefix_ids = tok.encode(PREFIX[0]).ids

    rng = np.random.default_rng(0)
    dev = model.device
    #: 线性头：**评测必须用一个能表达这个任务的头**。树读出的概率被语言先验主导，
    #: 实测在五种配置下 top-1 恒为 0.109（= 常数预测），分辨不出任何差别。
    head_w = torch.zeros(model.readout_width, 10)
    buf_x: list[torch.Tensor] = []
    buf_y: list[torch.Tensor] = []
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
        feats = model._features(trace)
        buf_x.append(feats.detach()); buf_y.append(torch.tensor(temp))
        if len(buf_x) > 200:
            buf_x.pop(0); buf_y.pop(0)
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
                    # **kNN：非线性、无参数、不吃语言先验**——线性探针只能测"线性可解码"，
                    # 树读出被先验淹没，两者都不干净。kNN 是唯一不带这两条假设的探针。
                    # **只看传感器池的维**：全特征里 156 个恒定的文本维会把 16 个有信息的维稀释掉
                    # （实测：全特征最近邻 [1,0,4,4,2,4,8,8,7,8]，只看池 [1,0,3,4,5,4,7,8,7,8]）。
                    X = torch.stack(buf_x[-50:])[:, -2 * POOL :].reshape(-1, 2 * POOL)
                    Y = torch.stack(buf_y[-50:]).reshape(-1)
                    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
                    Q = model._features(tr)[:, -2 * POOL :]
                    Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-9)
                    sim = Q @ X.T
                    topk = sim.topk(5, dim=1).indices
                    votes = Y[topk]
                    pred = torch.mode(votes, dim=1).values.cpu().numpy()
                    hit += int((pred == t2).sum()); n += args.batch
            print(f"  [{args.arm}] {step} 步  评测 top-1 {hit/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    out = {"arm": args.arm, "steps": args.steps, "batch": args.batch}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.arm}.json").write_text(json.dumps(out) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
