"""PMI 式融合：把快半的**通用词频先验**减掉，只留它对上文的敏感部分。

    score = log q_slow + α · (log p_fast − log p_prior)

其中 `prior` 是语料的一元词频（`counts_32000.npy`，就是建树时用的那份）。

**理由**：快半的绝对分布里有一大块是"这个字在中文里本来就不常见"——那部分千问也有、
而且更准。减掉它，剩下的是"**在这个上下文里**它认为接下来是什么"，
这才是可能互补的信号。

如果去偏之后 F-0 上升，说明快半确实带着上下文相关的信息；
如果不变，说明它和千问只是在同一个先验上分歧，融合没有额外信息。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

QWEN_DIR = Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2")
COUNTS = Path("D:/Projects/BioSNN-Plug/sweep_results/lm_semprea_v1/counts_32000.npy")


def token_priors() -> dict[str, float]:
    """token 字符串 → log 一元概率。用子集分词器把 32k 个 id 解回字符串。"""
    import sys

    sys.path.insert(0, "D:/Projects/BioSNN-Plug")
    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    bridge = SubsetTokenizer(QWEN_DIR)
    counts = np.load(COUNTS).astype(np.float64)
    total = max(counts.sum(), 1.0)
    priors: dict[str, float] = {}
    for local_id in range(len(counts)):
        text = bridge.decode([local_id])
        if text:
            priors[text] = math.log(max(counts[local_id] / total, 1e-12))
    return priors


def ok(tok: str, truth: str) -> bool:
    return bool(tok) and truth.startswith(tok)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PMI 去偏融合")
    ap.add_argument(
        "--cache", type=Path, default=Path("sweep_results/lm_fusion/alpha_v1_big.raw.json")
    )
    args = ap.parse_args(argv)

    raw = json.loads(args.cache.read_text(encoding="utf-8"))
    priors = token_priors()
    print(f"先验表 {len(priors)} 个 token，位置 {len(raw)}")

    def debias(fast, alpha: float, beta: float):
        out = []
        for token, lp in fast:
            prior = priors.get(token)
            if prior is None:  # 多字 token 没有一元先验，按 0 处理（不减）
                prior = 0.0
            out.append((token, lp - beta * prior))
        return out

    def evaluate(beta: float, alpha: float) -> tuple[int, int, float]:
        slow_ok = fast_only = oracle = 0
        for row in raw:
            slow, fast, truth = row["slow"], row["fast"], row["truth"]
            if not slow:
                continue
            s_tok = max(slow, key=lambda kv: kv[1])[0]
            s_ok = ok(s_tok, truth)
            adj = debias(fast, alpha, beta)
            f_tok = max(adj, key=lambda kv: kv[1])[0] if adj else ""
            f_ok = ok(f_tok, truth)
            slow_ok += int(s_ok)
            fast_only += int(f_ok and not s_ok)
            oracle += int(s_ok or f_ok)
        n = max(len(raw), 1)
        return slow_ok, fast_only, oracle / n

    print(f"\n{'β':>5}{'快半 top-1':>12}{'F-0':>7}{'oracle':>9}")
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        slow_ok, fast_only, oracle = evaluate(beta, 0.0)
        n = max(len(raw), 1)
        # 快半 top-1 = 自己命中数（含与慢半重合的那部分，此处只报独立命中 + 重合）
        print(f"{beta:>5.2f}{'':>12}{fast_only:>7d}{oracle:>9.3f}   （慢半 {slow_ok}/{n}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
