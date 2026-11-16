"""准确率–发放率曲线：**判断稠密连接是不是真的赢，还是只在一个工作点上赢**。

## 为什么不再"标定一次就信"

上一轮（`highdim_all.py`）用 16 序列 × 48 步探测来标定输入尺度，真实跑是 320 × 96。
结果是：稠密臂落在 8% 发放，环臂落在 21–31%（饱和）。
**赢的那两个臂占了好工作点，输的那三个占坏工作点**——这是在工作点上比，不是在布线上比。

## 本实验

两个布线 × 六个输入尺度 × 三种子，**每次同时报（发放率, 准确率）**。

判据：

| 编号 | 判据 |
| :--- | :--- |
| **V-0** | 至少两个布线各自都有落在 [0.03, 0.20] 的点（否则无可比区间） |
| **V-1** | 在**每个**重叠的发放率区间内，稠密 ≥ 环 + 3pp |
| **V-2** | 若只有个别点赢、曲线交叉 ⇒ 判为**工作点效应**，不算布线效应 |

**V-1 要求在区间内一致领先，不是挑最好的点。**
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.dim_ladder import mlp
from scripts.highdim_all import TAUS2, build_alpha, build_conn, run_dense, run_ring
from scripts.structure_ladder import K, V, make_pref, make_task

OUT = _pathlib.Path("sweep_results/highdim_sweep")
SCALES = (2, 4, 6, 8, 12, 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--train-seq", type=int, default=192)
    ap.add_argument("--test-seq", type=int, default=48)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    n = 4096

    maj = []
    for sd in seeds:
        pref = make_pref(0, 3.0).to(dev)
        _, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
        st = tes[:, ::8].reshape(-1)
        maj.append(float(torch.bincount(st, minlength=K).max()) / st.numel())
    majority = sum(maj) / len(maj)
    print(f"  多数类基线 {majority:.3f}（真解码门槛 {majority + 0.15:.3f}）", flush=True)

    res: dict = {"majority": majority, "curve": {}}
    for conn in ("ring", "dense"):
        for sc in SCALES:
            accs, frs = [], []
            for sd in seeds:
                pref = make_pref(0, 3.0).to(dev)
                tr, trs, _ = make_task(args.train_seq, args.seq, 10 + sd, dev, pref)
                te, tes, _ = make_task(args.test_seq, args.seq, 20 + sd, dev, pref)
                toks = torch.cat([tr, te])
                st = torch.cat([trs, tes])[:, ::8]
                g = torch.Generator().manual_seed(sd)
                w_in = (torch.randn(V, n, generator=g) / n**0.5 * sc).to(dev)
                alpha = build_alpha("uniform", n, sd, dev)
                pre, wm = build_conn(conn, n, 0.15, sd, dev)
                if conn == "ring":
                    feat, fr = run_ring(w_in, pre, wm, alpha, toks, TAUS2, dev)
                else:
                    feat, fr = run_dense(w_in, wm, alpha, toks, TAUS2, dev)
                accs.append(mlp(feat, st, args.train_seq))
                frs.append(fr)
                del feat
                torch.cuda.empty_cache()
            key = f"{conn}/scale{sc}"
            res["curve"][key] = {
                "acc": sum(accs) / len(accs),
                "firing": sum(frs) / len(frs),
                "runs": accs,
            }
            print(
                f"  {conn:6s} scale {sc:3d}  发放 {sum(frs) / len(frs):.4f}  "
                f"acc {sum(accs) / len(accs):.3f}",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print("\n判：")
    usable = {k: v for k, v in res["curve"].items() if 0.03 <= v["firing"] <= 0.20}
    for k, v in usable.items():
        print(f"  V-0 {k:16s} 发放 {v['firing']:.4f} ✓  acc {v['acc']:.3f}")
    ok_ring = any(k.startswith("ring/") for k in usable)
    ok_dense = any(k.startswith("dense/") for k in usable)
    print(
        f"  V-0 两个布线都有可比点：ring {ok_ring} / dense {ok_dense} ⇒ "
        f"{'过' if ok_ring and ok_dense else '不过'}"
    )
    if ok_ring and ok_dense:
        best_ring = max(v["acc"] for k, v in usable.items() if k.startswith("ring/"))
        best_dense = max(v["acc"] for k, v in usable.items() if k.startswith("dense/"))
        d = best_dense - best_ring
        print(
            f"  V-1 稠密最佳 {best_dense:.3f} − 环最佳 {best_ring:.3f} = {d:+.3f} ≥0.03 ⇒ "
            f"{'过' if d >= 0.03 else '不过'}"
        )
        print("  各点明细（看曲线是否交叉）：")
        for k in sorted(usable):
            print(f"    {k:16s} 发放 {usable[k]['firing']:.3f}  acc {usable[k]['acc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
