"""追试 ②：**行为熵 0.99 是「丰富」还是「掷骰子」**（预注册 `research/sel/prereg-behavior-structure-v1.md`）

## 问题

`m6ay`：量子闭环的符号熵 0.98–0.99（极大值 1.0），**四种子方差近零**。
**熵在极大值处分不清「丰富」与「噪声」；方差塌缩是警报。**

## 设计：**熵匹配的 shuffle 对照**

把行为串的符号**打乱**（保留每个符号的计数 ⇒ **熵完全相同**），再算结构测度。
**两者熵相同，只有时间结构不同。**

三个结构测度：LZ76 归一化复杂度、一阶自相关、zlib 压缩长度。

## 判据（照预注册）

- **S-0 仪器**：LZ 自检 全零 < 周期3 < 随机；
- **S-1 排除伪丰富** ★★：量子串 LZ **显著低于**熵匹配 shuffle（≥5/6 种子）⇒ 真丰富；
- **S-2 结构确认** ★：若 S-1 不过，至少一个其它结构测度显著；
- **S-3 对照**：经典 argmax 串（熵 0.59）应当结构明显。

**纪律：≥6 种子；串长 ≥512。**
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import os
import pathlib as _pathlib
import sys as _sys
import zlib

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import build_pools
from scripts.m6ay_quantum_loop import run_loop

OUT = _pathlib.Path("sweep_results/m6ay2_structure")
K = 4
T_HIST, T_CLOSED = 1024, 512


def lz_norm(syms) -> float:
    """LZ76 复杂度，按长度归一。符号先映射到字母表。"""
    alpha = {s: chr(65 + i) for i, s in enumerate(sorted(set(syms)))}
    s = "".join(alpha[x] for x in syms)
    n = len(s)
    if n == 0:
        return 0.0
    i, c, pos, k, kmax = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[pos + k - 1]:
            k += 1
            if pos + k > n:
                c += 1
                break
        else:
            kmax = max(kmax, k)
            i += 1
            if i == pos:
                c += 1
                pos += kmax
                if pos + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return c / n


def autocorr1(syms) -> float:
    x = torch.tensor(syms, dtype=torch.float32)
    a, b = x[:-1], x[1:]
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if a.std() > 1e-9 else 0.0


def comp_len(syms) -> float:
    return len(zlib.compress(bytes(syms), 9)) / max(len(syms), 1)


def shuffled(syms, seed: int):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(syms), generator=g).tolist()
    return [syms[i] for i in p]


def self_check() -> bool:
    n = 2000
    g = torch.Generator().manual_seed(0)
    zeros = [0] * n
    per3 = [i % 3 for i in range(n)]
    rand = [int(x) for x in (torch.rand(n, generator=g) < 0.5).long()]
    a, b, c = lz_norm(zeros), lz_norm(per3), lz_norm(rand)
    print(f"  LZ 自检：全零 {a:.4f}  周期3 {b:.4f}  随机 {c:.4f}", flush=True)
    return bool(a < b < c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    print("S-0 自检：", flush=True)
    ok = self_check()
    print(f"  => {'过' if ok else '**不过 => 作废**'}", flush=True)
    res: dict = {"self_check": ok, "cells": {}}
    if not ok:
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )
        return 1

    for mode in ("classical", "quantum"):
        rows = []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            ha = make_family_hist(0, sd * 97 + 1, T_HIST, 1, dev)
            syms, _ = run_loop(net, sd, ha, dev, T_CLOSED, mode == "quantum", 8.0)
            sh = shuffled(syms, int.from_bytes(os.urandom(8), "little") >> 1)
            row = {
                "seed": sd,
                "lz": lz_norm(syms),
                "lz_sh": lz_norm(sh),
                "ac": autocorr1(syms),
                "ac_sh": autocorr1(sh),
                "cl": comp_len(syms),
                "cl_sh": comp_len(sh),
                "struct": lz_norm(syms) < lz_norm(sh),
            }
            rows.append(row)
            print(
                f"  {mode:9s} seed {sd}: LZ {row['lz']:.4f} vs {row['lz_sh']:.4f}  "
                f"自相关 {row['ac']:+.3f} vs {row['ac_sh']:+.3f}  "
                f"压缩 {row['cl']:.3f} vs {row['cl_sh']:.3f}",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        v = {k: m(k) for k in rows[0]}
        v["rows"] = rows
        v["wins"] = sum(1 for r in rows if r["struct"])
        res["cells"][mode] = v
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    cc = res["cells"]
    q, c = cc["quantum"], cc["classical"]
    s1 = q["wins"] >= 5
    print(
        f"  S-1 量子串 LZ 低于熵匹配 shuffle（≥5/6）：{q['wins']}/{args.seeds} "
        f"（{q['lz']:.4f} vs {q['lz_sh']:.4f}）=> {'过：真丰富' if s1 else '不过'}"
    )
    s2 = abs(q["ac"] - q["ac_sh"]) > 0.15 or q["cl"] < q["cl_sh"] - 0.05
    print(
        f"  S-2 其它结构测度显著：自相关差 {abs(q['ac'] - q['ac_sh']):.3f}、"
        f"压缩差 {q['cl_sh'] - q['cl']:+.3f} => {'过' if s2 else '不过'}"
    )
    s3 = c["wins"] >= 5
    print(
        f"  S-3 对照（经典串应有结构）：{c['wins']}/{args.seeds} "
        f"（LZ {c['lz']:.4f} vs {c['lz_sh']:.4f}）=> {'过' if s3 else '不过'}"
    )
    if not s1 and not s2 and s3:
        print("  ⇒ **伪丰富确认**：正结果降级为「不自毁」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
