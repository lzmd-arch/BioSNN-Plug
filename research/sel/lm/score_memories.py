"""用局部规则模型给**真实记忆库**打"惊讶度"分，并与真实结局变量对照。

动机：这个模型当聊天模型差 20pp，但当"**这个人自己的文本分布模型**"未必不够用。
惊讶度（逐条 CE）作为"值不值得记"的打分器，不需要模型会聊天、不需要 1B、不需要 GPU。

**只读**：打开的是 `D:/AxoData/memory/snapshots/` 下的只读快照，主库一个字都不碰。

用法::

    uv run python -m research.sel.lm.score_memories --updates 300 --folds 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from research.sel.lm.model import TokenEProp

OUT = Path(__file__).resolve().parents[3] / "sweep_results" / "lm_eprop_memory"
SNAPSHOT = "D:/AxoData/memory/snapshots/mem-20261110T100005.db"


def load_rows(path: str) -> list[dict]:
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = [
        {
            "id": r[0],
            "kind": r[1],
            "tier": r[2],
            "content": r[3],
            "access_count": r[4],
            "reinforce_count": r[5],
            "weight": r[6],
        }
        for r in conn.execute(
            "select memory_id, kind, tier, content, access_count, reinforce_count, weight"
            " from memories where content is not null and length(content) > 0"
        )
    ]
    conn.close()
    return rows


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(ch, unk) for ch in text[:256]]


def train_and_score(
    texts: list[str], held: list[int], *, n_rec: int, updates: int, seed: int
) -> dict[int, float]:
    """在**不含 held** 的文本上训练，然后给 held 里每条算逐 token CE。"""
    keep = [t for i, t in enumerate(texts) if i not in set(held)]
    chars = sorted({c for t in keep for c in t})
    vocab = {c: i + 1 for i, c in enumerate(chars)}
    vocab["<unk>"] = len(vocab)
    train = [encode(t, vocab) for t in keep]
    train = [s for s in train if len(s) >= 4]
    model = TokenEProp(
        len(vocab),
        n_rec,
        fan_in=32,
        tau=20.0,
        tau_read=8.0,
        input_std=0.12,
        lr_in=1e-3,
        lr_out=0.5,
        recurrence="none",
        readout_taus=(2.0, 128.0),
        generator=torch.Generator().manual_seed(seed),
        device="cpu",
    )
    gen = torch.Generator().manual_seed(seed + 7)
    for _ in range(updates):
        order = torch.randint(0, len(train), (16,), generator=gen).tolist()
        width = max(len(train[i]) for i in order)
        tokens = torch.full((16, width), vocab["<unk>"], dtype=torch.long)
        valid = torch.zeros(16, width, dtype=torch.bool)
        for row, idx in enumerate(order):
            seq = train[idx]
            tokens[row, : len(seq)] = torch.tensor(seq)
            valid[row, : len(seq)] = True
        model.run(tokens, model.new_trace(16), valid=valid)
    scores: dict[int, float] = {}
    for i in held:
        seq = encode(texts[i], vocab)
        if len(seq) < 2:
            continue
        tokens = torch.tensor(seq).unsqueeze(0)
        valid = torch.ones(1, len(seq), dtype=torch.bool)
        stats = model.evaluate(tokens, model.new_trace(1), valid=valid)
        scores[i] = float(stats["ce"])
    return scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真实记忆库的惊讶度打分")
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--n-rec", type=int, default=512)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="memory_surprisal")
    args = ap.parse_args(argv)

    rows = load_rows(args.snapshot)
    texts = [r["content"] for r in rows]
    print(f"记忆 {len(rows)} 条；平均 {sum(len(t) for t in texts) / len(texts):.1f} 字", flush=True)

    scores: dict[int, float] = {}
    for k in range(args.folds):
        held = list(range(k, len(rows), args.folds))
        part = train_and_score(texts, held, n_rec=args.n_rec, updates=args.updates, seed=k)
        scores.update(part)
        print(
            f"  fold {k}: 训练 {len(rows) - len(held)} 条，打分 {len(part)} 条，"
            f"中位 CE {sorted(part.values())[len(part) // 2]:.3f}",
            flush=True,
        )

    scored = [(rows[i], scores[i]) for i in scores]
    ce = torch.tensor([s for _, s in scored], dtype=torch.float32)
    print(
        f"\n惊讶度 CE：中位 {float(ce.median()):.3f}，"
        f"四分位 {[round(float(x), 3) for x in ce.quantile(torch.tensor([0.25, 0.75]))]}",
        flush=True,
    )

    def corr(key: str) -> float | None:
        try:
            v = torch.tensor([float(r[key] or 0) for r, _ in scored], dtype=torch.float32)
        except (TypeError, ValueError):
            return None  # tier 这类是字符串，不参与相关
        if float(v.std()) == 0:
            return None
        return float(torch.corrcoef(torch.stack([ce, v]))[0, 1])

    report = {
        "n_memories": len(rows),
        "n_scored": len(scored),
        "median_ce": float(ce.median()),
        "mean_ce": float(ce.mean()),
        "corr": {k: corr(k) for k in ("access_count", "reinforce_count", "weight", "tier")},
    }
    print("\n与真实结局变量的相关：")
    for k, v in report["corr"].items():
        print(f"  {k:16s} {'—（无方差）' if v is None else f'{v:+.3f}'}", flush=True)
    uniform = math.log(max(len({c for t in texts for c in t}), 2))
    print(
        "参考：本库词表下均匀分布的 CE 约 "
        + f"{uniform:.3f}"
        + "；中位 "
        + f"{float(ce.median()):.3f}"
        + " 相对均匀降了 "
        + f"{(1 - float(ce.median()) / uniform) * 100:.1f}"
        + "%"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
