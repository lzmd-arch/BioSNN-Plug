"""V2 的 S0：条件偏置在真实骨架上到底有没有头寸。

预注册：`research/sel/prereg-v2-s0-conditioning.md`

两步：
    python scripts/v2_s0.py collect     # 打本地 llama-server，落原始 logprobs
    python scripts/v2_s0.py analyze     # 用落盘的原始数据跑 B0/B2/B1/D 四臂

**骨架上不加任何东西**：所有臂都在同一份 Qwen logprobs 上比较。
"""

# ruff: noqa: N803, N806  —— 这是矩阵记号，A/P/Q/W 是本文件的可读性承重件

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

HOST = "http://127.0.0.1:8081"
CORPUS = Path("D:/AxoData/sel/lm/corpus/fineweb2_u16/corpus.u16")
TOKROOT = Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2")
OUT = Path("sweep_results/v2_s0")
RAW_TAG = ""
N_PROBS = 100
TOP_K = 20


def _post(path: str, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def collect(positions: int, stride: int, start: int) -> dict:
    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    tok = SubsetTokenizer(TOKROOT)
    a = np.memmap(CORPUS, dtype=np.uint16, mode="r")
    text = tok.decode(a[start : start + stride * positions + 4000].tolist())
    raw_pieces = _post("/tokenize", {"content": text, "with_pieces": True})["tokens"]
    # 有的 piece 是字节数组（一个汉字被切成多字节 token）；统一成 bytes 再按前缀解码
    parts = [
        t["piece"].encode() if isinstance(t["piece"], str) else bytes(t["piece"])
        for t in raw_pieces
    ]
    rows = []
    truth_text = [b.decode("utf-8", errors="replace") for b in parts]
    for i in range(1, min(len(parts) - 1, stride * positions), stride):
        try:
            prompt = b"".join(parts[:i]).decode("utf-8")
        except UnicodeDecodeError:
            continue  # 前缀落在多字节 token 中间 —— 跳过，不拼半个字
        truth = truth_text[i]
        r = _post(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": N_PROBS,
                "temperature": 0,
            },
        )
        cp = r.get("completion_probabilities") or []
        if not cp:
            continue
        top = cp[0]["top_logprobs"]
        rows.append(
            {
                "pos": i,
                "prompt_tail": prompt[-40:],
                "truth": truth,
                "top": [[t["token"], float(t["logprob"])] for t in top],
            }
        )
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} 个位置", flush=True)
    return {"host": HOST, "positions": len(rows), "rows": rows}


def _features(row: dict) -> np.ndarray:
    """预注册 §二 的 8 维 p——全部推理期可算。"""
    top = row["top"]
    lps = np.array([lp for _, lp in top], dtype=np.float64)
    tail = row["prompt_tail"]
    n = max(len(tail), 1)
    punct = sum(ch in "，。、；：！？,.!?;:" for ch in tail) / 8.0
    rep = (len(tail) - len(set(tail))) / n
    digit = sum(ch.isdigit() for ch in tail) / 8.0
    p = lps - lps.max()
    prob = np.exp(p)
    prob /= prob.sum()
    ent = float(-(prob * np.log(prob + 1e-12)).sum())
    return np.array(
        [
            row["pos"] / 512.0,
            float(lps[0]),
            ent,
            float(lps[0] - lps[1]) if len(lps) > 1 else 0.0,
            punct,
            rep,
            digit,
            1.0,
        ],
        dtype=np.float64,
    )


def _setup(rows: list[dict], k: int):
    """统一候选集，返回 (logq[n,C], y[n], P[n,8]) 与词表。"""
    vocab = sorted({t for r in rows for t, _ in r["top"][:k]} | {r["truth"] for r in rows})
    idx = {t: i for i, t in enumerate(vocab)}
    n, C = len(rows), len(vocab)
    logq = np.full((n, C), -50.0)
    # **逐位置归一化**：候选集 = 该位置的 top-k ∪ {真值}，与预注册 §二 一致。
    mask = np.zeros((n, C), dtype=bool)
    y = np.zeros(n, dtype=np.int64)
    P = np.zeros((n, 8))
    cover = 0
    for i, r in enumerate(rows):
        floor = min(lp for _, lp in r["top"]) - 5.0
        logq[i] = floor
        for t, lp in r["top"][:k]:
            logq[i, idx[t]] = lp
            mask[i, idx[t]] = True
        y[i] = idx[r["truth"]]
        if abs(logq[i, y[i]] - floor) > 1e-9:
            cover += 1
        mask[i, y[i]] = True
        P[i] = _features(r)
    return logq, y, P, vocab, cover / n, mask


def _logsoftmax(z: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        z = np.where(mask, z, -1e9)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def _proj(r: int, seed: int = 7) -> np.ndarray:
    """8 → r 的**冻结**随机投影：让可学参数只有 A ∈ R^{r×C}。"""
    return np.random.default_rng(seed).normal(0, 1.0 / math.sqrt(8), (8, r))


def _fit(logq, y, P, r, mask=None, l2=1e-2, epochs=20, lr=0.5, seed=0):
    n, C = logq.shape
    W = _proj(r)
    Q = P @ W  # (n, r)
    A = np.zeros((r, C))
    for _ in range(epochs):
        lp = _logsoftmax(logq + Q @ A, mask)
        pr = np.exp(lp)
        pr[np.arange(n), y] -= 1.0
        A = A - lr * (Q.T @ pr / n + l2 * A)
    return A


def _score(logq, y, P, A=None, temp=1.0, mask=None):
    z = logq / temp
    if A is not None:
        z = z + (P @ _proj(A.shape[0])) @ A
    lp = _logsoftmax(z, mask)
    return float(lp[np.arange(len(y)), y].mean())


def analyze(k: int = TOP_K, r: int = 2, seeds: int = 5) -> dict:
    name = f"raw_{RAW_TAG}.json" if RAW_TAG else "raw.json"
    raw = json.loads((OUT / name).read_text(encoding="utf-8"))
    rows = raw["rows"]
    logq, y, P, vocab, cover, mask = _setup(rows, k)
    n = len(rows)
    rng = np.random.default_rng(12345)
    order = rng.permutation(n)
    cut = int(n * 0.7)
    tr, te = order[:cut], order[cut:]

    out = {
        "positions": n,
        "vocab": len(vocab),
        "truth_coverage": round(cover, 4),
        "train": len(tr),
        "holdout": len(te),
        "r": r,
        "k": k,
    }
    mte, mtr = mask[te], mask[tr]
    out["B0"] = _score(logq[te], y[te], P[te], mask=mte)

    best_t, best = 1.0, -1e9
    for t in np.linspace(0.5, 2.0, 31):
        s = _score(logq[tr], y[tr], P[tr], temp=t, mask=mtr)
        if s > best:
            best, best_t = s, t
    out["B2"] = {
        "logp": _score(logq[te], y[te], P[te], temp=best_t, mask=mte),
        "temp": float(best_t),
    }

    rand = []
    for sd in range(seeds):
        A = np.random.default_rng(sd).normal(0, 1e-3, (r, logq.shape[1]))
        rand.append(_score(logq[te], y[te], P[te], A=A, mask=mte))
    out["B1"] = {"logp": float(np.mean(rand)), "best": float(max(rand)), "all": rand}

    A = _fit(logq[tr], y[tr], P[tr], r, mask=mtr)
    d_lp = _score(logq[te], y[te], P[te], A=A, mask=mte)
    out["D"] = {"logp": d_lp, "gain_vs_B0": d_lp - out["B0"]}

    real = out["D"]["gain_vs_B0"]
    perm = []
    for _ in range(200):
        perm_idx = np.random.default_rng(len(perm)).permutation(len(tr))
        Ai = _fit(logq[tr], y[tr], P[tr][perm_idx], r, mask=mtr)
        perm.append(_score(logq[te], y[te], P[te], A=Ai, mask=mte) - out["B0"])
    out["S0-2_p"] = float((np.array(perm) >= real).mean())
    out["perm_mean"] = float(np.mean(perm))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "analyze"))
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--start", type=int, default=500_000)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "collect":
        d = collect(args.positions, args.stride, args.start)
        (OUT / "raw.json").write_text(
            json.dumps(d, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps({k: v for k, v in d.items() if k != "rows"}, ensure_ascii=False))
        return 0
    global RAW_TAG
    RAW_TAG = args.tag
    out = analyze(args.k, args.r)
    (OUT / f"s0{("_"+RAW_TAG) if RAW_TAG else ""}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
