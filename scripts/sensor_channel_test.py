"""传感器通道的**连通性**测试：通道是活的，方向要学。

本机没有配对的（传感器, 文本）数据，所以这里**不宣称学到任何语义**。它只回答三件事：

1. **零向量/无向量时逐位无副作用**（早先已经用守卫钉死，这里在真检查点上再验一次）；
2. **给了非零投影和非零向量，分布会动**——通道真的接在分布上；
3. **不同的传感器向量给出不同的偏移**（KL 不一样）——即通道是可区分的，不是一个常量偏置。

方向（哪个通道该把分布推向哪个词）**必须由数据学出来**，不是手设的。
本脚本故意给一个随机投影，就是为了说明"有偏移"和"偏移对"是两件事。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sel.lm.hierarchical import build_huffman_paths
from research.sel.lm.model import TokenEProp


def load(ckpt: Path, sensor_dim: int, device):
    blob = torch.load(ckpt, map_location="cpu")
    cfg = blob["config"]
    vocab = int(cfg["vocab"])
    n_rec = max(int(float(cfg["params"])) // vocab // 2, 64)
    counts_path = ckpt.parent / f"counts_{vocab}.npy"
    if counts_path.exists():
        import numpy as np

        counts = torch.from_numpy(np.load(counts_path)).float()
    else:
        counts = torch.ones(vocab)
    model = TokenEProp(
        vocab,
        n_rec,
        recurrence="none",
        readout_taus=(2.0, 128.0),
        tree=build_huffman_paths(counts).to(device),
        tree_signal_scale=float(cfg["tree_signal_scale"]),
        tree_packed=True,
        embed_rank=int(cfg["embed_rank"]),
        tree_init_weight=False,
        sensor_dim=sensor_dim,
        generator=torch.Generator().manual_seed(0),
        device=device,
    )
    for key, value in blob["tensors"].items():
        if key == "__tree_weight":
            model.tree.weight = value.to(device)
        elif key == "__tree_bias":
            model.tree.bias = value.to(device)
        else:
            setattr(model, key, value.to(device))
    from tokenizers import ByteLevelBPETokenizer

    bpe = Path("D:/AxoData/sel/lm/corpus/skypile_tok/bpe")
    tokenizer = ByteLevelBPETokenizer(str(bpe / "vocab.json"), str(bpe / "merges.txt"))
    return model, tokenizer


@torch.no_grad()
def probs(model, tokenizer, context: str) -> torch.Tensor:
    ids = tokenizer.encode(context).ids[-256:]
    trace = model.new_trace(1)
    for step in range(max(len(ids) - 1, 0)):
        model.step(
            torch.tensor([ids[step]], device=model.device),
            torch.tensor([ids[step + 1]], device=model.device),
            trace,
            update=False,
            all_valid=True,
        )
    return model._token_probabilities(model._features(trace))[0]


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p * (p.clamp(min=1e-12).log() - q.clamp(min=1e-12).log())).sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="传感器通道连通性测试（不宣称语义）")
    ap.add_argument("--ckpt", type=Path, default=Path("sweep_results/lm_1b/run05b_ckpt.pt"))
    ap.add_argument("--sensor-dim", type=int, default=4)
    ap.add_argument("--context", default="北岸一号站的采集箱每")
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/sensor_channel.json"))
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load(args.ckpt, args.sensor_dim, device)
    base = probs(model, tokenizer, args.context)

    # 1) 零向量：必须逐位相同
    model.sensors = torch.zeros(args.sensor_dim, device=device)
    zero = probs(model, tokenizer, args.context)
    identical = torch.equal(base, zero)

    # 2) 随机投影 + 几个不同的传感器向量
    gen = torch.Generator().manual_seed(0)
    model.sensor_proj = (torch.randn(args.sensor_dim, model.n_rec, generator=gen) * args.scale).to(
        device
    )
    rows = []
    for vector in (
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 3.0],
    ):
        model.sensors = torch.tensor(vector, device=device)
        shifted = probs(model, tokenizer, args.context)
        rows.append(
            {
                "vector": vector,
                "kl_vs_zero": round(kl(shifted, base), 4),
                "argmax_changed": int(shifted.argmax()) != int(base.argmax()),
            }
        )
    model.sensors = None

    print(f"上下文 {args.context!r}")
    print(f"  零向量逐位相同：{identical}")
    for row in rows:
        print(
            f"  {row['vector']} → KL {row['kl_vs_zero']:.4f}，argmax 变了 {row['argmax_changed']}"
        )
    distinct = len({row["kl_vs_zero"] for row in rows}) == len(rows)
    print(f"\n判据 T-1（零向量无副作用）：{'过' if identical else '不过'}")
    print(f"判据 T-2（通道可区分，不同向量 → 不同 KL）：{'过' if distinct else '不过'}")
    print("**方向（哪个通道推向哪个词）必须由数据学出来，本测试不宣称任何语义。**")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"identical_at_zero": identical, "rows": rows}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
