"""把 SNN 的特征当作**第三个召回通道**，在 Axo 的同一套金标准上量 hit@K。

对比对象（本机实测，`tools/bench_recall.py`）：

| 通道 | hit@1 | hit@5 |
| :--- | ---: | ---: |
| 纯词法（SDR） | **0.8961** | 0.9351 |
| 词法 + semantic(nomic 768d) | 0.8312 | 0.9091 |

**判据不是"高不高"，是"在词法必输的那一类上有没有赢"**：`paraphrase` 子集（n=5）词法只有 0.20，
但 n=5 太小 ⇒ 结论只能当信号，不能当判决。

用法::

    uv run python scripts/eval_recall_snn.py --ckpt sweep_results/lm_1b/xxx_ckpt.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
GOLD = Path("D:/Projects/BioSNN-MemoryTerminal/benchmarks/gold")
BPE = Path("D:/AxoData/sel/lm/corpus/skypile_tok/bpe")

#: 本机实测的对照（`tools/bench_recall.py`，2026-11-12）
BASELINE = {"lexical": (0.8961, 0.9351), "lexical+semantic": (0.8312, 0.9091)}


def encode(model, tokenizer, text: str, max_len: int = 256) -> torch.Tensor:
    """把一段文本压成一个向量：**各位置读出特征的均值**。"""
    ids = tokenizer.encode(text).ids[:max_len]
    if len(ids) < 2:
        return torch.zeros(model.readout_width, device=model.device)
    trace = model.new_trace(1)
    device = model.device
    acc = torch.zeros(model.readout_width, device=device)
    for step, token in enumerate(ids[:-1]):
        model.step(
            torch.tensor([token], device=device),
            torch.tensor([ids[step + 1]], device=device),
            trace,
            update=False,
            all_valid=True,
        )
        acc += model._features(trace)[0]
    return acc / (len(ids) - 1)


def load_model(ckpt: Path, device):
    from research.sel.lm.hierarchical import build_huffman_paths
    from research.sel.lm.model import TokenEProp

    blob = torch.load(ckpt, map_location="cpu")
    cfg = blob["config"]
    vocab = int(cfg["vocab"])
    width = int(float(cfg["params"])) // vocab
    n_rec = max(width // 2, 64)
    counts_path = ckpt.parent / f"counts_{vocab}.npy"
    if counts_path.exists():
        import numpy as np

        counts = torch.from_numpy(np.load(counts_path)).float()
    else:
        counts = torch.ones(vocab)
    paths = build_huffman_paths(counts).to(device)
    model = TokenEProp(
        vocab,
        n_rec,
        recurrence="none",
        readout_taus=(2.0, 128.0),
        tree=paths,
        tree_signal_scale=float(cfg["tree_signal_scale"]),
        tree_packed=True,
        embed_rank=int(cfg["embed_rank"]),
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
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SNN 通道的召回评测")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--bpe", type=Path, default=BPE)
    ap.add_argument("--out", type=Path, default=REPO / "sweep_results" / "lm_recall_snn")
    ap.add_argument("--tag", default="snn_channel")
    args = ap.parse_args(argv)

    from tokenizers import ByteLevelBPETokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    tokenizer = ByteLevelBPETokenizer(str(args.bpe / "vocab.json"), str(args.bpe / "merges.txt"))

    memories = [
        json.loads(x) for x in (args.gold / "memories.jsonl").read_text("utf-8").splitlines()
    ]
    queries = [json.loads(x) for x in (args.gold / "queries.jsonl").read_text("utf-8").splitlines()]
    print(f"语料 {len(memories)} 条 / 查询 {len(queries)} 条", flush=True)

    with torch.no_grad():
        mem_mat = torch.stack([encode(model, tokenizer, m["content"]) for m in memories])
        mem_mat = torch.nn.functional.normalize(mem_mat, dim=1)
        per_cat = defaultdict(lambda: [0, 0])
        hits1 = hits5 = 0
        rr = 0.0
        counted = 0
        for q in queries:
            if not q.get("expect"):
                continue
            vec = torch.nn.functional.normalize(encode(model, tokenizer, q["query"]), dim=0)
            scores = mem_mat @ vec
            order = torch.argsort(scores, descending=True).tolist()
            ranked = [memories[i]["id"] for i in order]
            want = set(q["expect"])
            pos = next((r for r, rid in enumerate(ranked) if rid in want), None)
            hit1 = int(pos == 0)
            hit5 = int(pos is not None and pos < 5)
            hits1 += hit1
            hits5 += hit5
            rr += (1.0 / (pos + 1)) if pos is not None else 0.0
            counted += 1
            cell = per_cat[q.get("category", "?")]
            cell[0] += hit1
            cell[1] += 1
    n = max(counted, 1)
    result = {
        "channel": "snn",
        "n": counted,
        "hit@1": round(hits1 / n, 4),
        "hit@5": round(hits5 / n, 4),
        "mrr": round(rr / n, 4),
        "by_category": {k: [v[0], v[1]] for k, v in sorted(per_cat.items())},
        "baseline": BASELINE,
    }
    print(
        f"SNN 通道：hit@1 {result['hit@1']:.4f}  hit@5 {result['hit@5']:.4f}  "
        f"MRR {result['mrr']:.4f}  (n={counted})",
        flush=True,
    )
    for key, (hit, total) in result["by_category"].items():
        print(f"  {key:16s} hit@1 {hit}/{total}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
