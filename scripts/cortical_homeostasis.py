"""稳态补偿：损伤后重新标定输入增益，看功能能否被救回（M5 的第一个可测形式）。

预注册：`research/sel/prereg-homeostatic-compensation-v1.md`

v8 测出：少 25% 神经元，活动塌两个数量级（0.081 → 0.004）。
生物学上神经元损伤后会**上调兴奋性**维持活动（稳态可塑性）。
本轮把补偿当自变量：

    nocomp  输入尺度保持完整时的值（v8 的行为）
    comp    重新标定输入尺度，使发放回到 0.08

**补偿的实现是"全局输入增益"**，是真实稳态机制（逐神经元阈值/兴奋性）的粗糙替身。
"""

from __future__ import annotations

import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.cortical_fold import N, build_fold
from scripts.instrument import calibrate, mlp_probe
from scripts.memory_depth import K_CUE, make_delay_task, run_md

OUT = _pathlib.Path("sweep_results/cortical_homeostasis")
LEVELS = (0.25, 0.50)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=1536)
    ap.add_argument("--test-seq", type=int, default=384)
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"N": N, "levels": list(LEVELS), "cells": {}}

    majors = []
    for sd in seeds:
        _, cue = make_delay_task(args.train_seq + args.test_seq, args.seq, 10 + sd, dev)
        trp, tep = cue[: args.train_seq], cue[args.train_seq :]
        top = int(torch.bincount(trp, minlength=K_CUE).argmax())
        majors.append(float((tep == top).float().mean()))
    floor = sum(majors) / len(majors)
    res["majority"] = floor
    print(f"  底板(多数类) {floor:.3f}", flush=True)

    out = {"0.00": {"nocomp": [], "comp": [], "sham": []}}
    frs = {"0.00": {"nocomp": [], "comp": [], "sham": []}}
    for lv in LEVELS:
        out[f"{lv:.2f}"] = {"nocomp": [], "comp": [], "sham": []}
        frs[f"{lv:.2f}"] = {"nocomp": [], "comp": [], "sham": []}

    for sd in seeds:
        w_in, pre, wrec = build_fold(sd, dev, False)
        cal_toks, cal_cue = make_delay_task(32, args.seq, 30 + sd, dev)

        def _rate(s, _a=None, _w=w_in, _p=pre, _rw=wrec, _t=cal_toks, _c=cal_cue):
            return run_md(_w * s, _p, _rw, _t, _c, dev, 0.0, 0.0, 0.0, alive=_a)[1]

        ok_cal, sc0, _pfr = calibrate(_rate)
        if not ok_cal:
            print(f"  seed {sd} ✗ 完整网络标定失败", flush=True)
            continue

        tr, cue_tr = make_delay_task(args.train_seq, args.seq, 10 + sd, dev)
        te, cue_te = make_delay_task(args.test_seq, args.seq, 20 + sd, dev)
        toks = torch.cat([tr, te])
        cue = torch.cat([cue_tr, cue_te])

        for lv in (0.0, *LEVELS):
            key = f"{lv:.2f}"
            if lv == 0.0:
                alive = None
            else:
                g = torch.Generator().manual_seed(7000 + sd * 31 + int(lv * 100))
                alive = (torch.rand(N, generator=g) >= lv).float().to(dev)

            # nocomp：沿用完整网络的尺度
            feat, fr = run_md(w_in * sc0, pre, wrec, toks, cue, dev, 0.0, 0.0, 0.0, alive=alive)
            out[key]["nocomp"].append(mlp_probe(feat[:, 1], cue, args.train_seq, K_CUE)[0])
            frs[key]["nocomp"].append(fr)
            del feat
            torch.cuda.empty_cache()

            if lv == 0.0:
                out[key]["comp"].append(out[key]["nocomp"][-1])
                frs[key]["comp"].append(frs[key]["nocomp"][-1])
                out[key]["sham"].append(out[key]["nocomp"][-1])
                frs[key]["sham"].append(frs[key]["nocomp"][-1])
                continue

            # sham：网络完好，只把同样这些位置的**读出**置零
            feat, fr = run_md(
                w_in * sc0, pre, wrec, toks, cue, dev, 0.0, 0.0, 0.0, alive=None, read_mask=alive
            )
            out[key]["sham"].append(mlp_probe(feat[:, 1], cue, args.train_seq, K_CUE)[0])
            frs[key]["sham"].append(fr)
            del feat
            torch.cuda.empty_cache()

            # comp：损伤后重新标定，把发放对回目标
            ok_c, sc_c, _f = calibrate(lambda s, _a=alive: _rate(s, _a))
            if not ok_c:
                print(f"  seed {sd} lv {key} ✗ 补偿标定失败", flush=True)
                # **必须 append 占位**，否则 comp 与 nocomp 列表错位，zip 会错配
                out[key]["comp"].append(float("nan"))
                frs[key]["comp"].append(float("nan"))
                continue
            feat, fr = run_md(w_in * sc_c, pre, wrec, toks, cue, dev, 0.0, 0.0, 0.0, alive=alive)
            out[key]["comp"].append(mlp_probe(feat[:, 1], cue, args.train_seq, K_CUE)[0])
            frs[key]["comp"].append(fr)
            del feat
            torch.cuda.empty_cache()

        print(f"  seed {sd} 完成", flush=True)
        del toks, cue, w_in, pre, wrec
        torch.cuda.empty_cache()

    for cond in ("nocomp", "comp", "sham"):
        res["cells"][cond] = {
            "acc": {
                k: (sum(out[k][cond]) / len(out[k][cond]) if out[k][cond] else float("nan"))
                for k in out
            },
            "runs": {k: out[k][cond] for k in out},
            "firing": {
                k: (sum(frs[k][cond]) / len(frs[k][cond]) if frs[k][cond] else float("nan"))
                for k in frs
            },
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    thresh = floor + 0.10
    for cond in ("nocomp", "comp", "sham"):
        accs = [round(cc[cond]["acc"][k], 3) for k in ("0.00", "0.25", "0.50")]
        fcs = [round(cc[cond]["firing"][k], 4) for k in ("0.00", "0.25", "0.50")]
        print(f"  {cond:7s} 峰值 {accs}  发放 {fcs}")

    c0 = cc["nocomp"]["acc"]["0.00"] >= thresh and 0.03 <= cc["nocomp"]["firing"]["0.00"] <= 0.20
    print(f"  C-0 0% 峰值 ≥ {thresh:.3f}、工作点 => {'过' if c0 else '不过 => 作废'}")
    if not c0:
        return 0
    c1 = all(0.03 <= cc["comp"]["firing"][k] <= 0.20 for k in ("0.25", "0.50"))
    print(f"  C-1 补偿把发放拉回窗内 => {'过' if c1 else '不过'}")
    c2 = cc["comp"]["acc"]["0.50"] >= thresh
    print(f"  C-2 补偿后 50% 损伤仍可用 => {'过' if c2 else '不过'}")
    diffs = [
        b - a
        for a, b in zip(cc["nocomp"]["runs"]["0.50"], cc["comp"]["runs"]["0.50"], strict=False)
    ]
    pos = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    print(f"  C-3 50% 损伤时 comp-nocomp 逐种子 {[round(d, 4) for d in diffs]}")
    print(f"        同号 {max(pos, n - pos)}/{n} => {'过' if max(pos, n - pos) == n else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
