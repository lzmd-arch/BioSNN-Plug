"""损伤下的简并：正常时等价的接线，损伤后是否分家。

预注册：`research/sel/prereg-degeneracy-lesion-v1.md`

到目前五次都测到「不同接线 -> 同一功能」（γ 改 10%、折叠改 21%，功能都平）。
简并的价值**不在正常功能里，在损伤下**。本轮测这个。

两套接线（`cortical_fold`）：`sheet`（薄片距离）与 `space`（3 维距离）。
损伤：随机 0% / 25% / 50% 神经元不再放电。指标：线索保持的峰值解码准确率。
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

OUT = _pathlib.Path("sweep_results/cortical_lesion")
LEVELS = (0.0, 0.25, 0.50)


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

    for arm, use_3d in (("sheet", False), ("space", True)):
        rows = {f"{lv:.2f}": [] for lv in LEVELS}
        frs = {f"{lv:.2f}": [] for lv in LEVELS}
        for sd in seeds:
            w_in, pre, wrec = build_fold(sd, dev, use_3d)
            cal_toks, cal_cue = make_delay_task(32, args.seq, 30 + sd, dev)

            def _rate(s, _w=w_in, _p=pre, _rw=wrec, _t=cal_toks, _c=cal_cue):
                return run_md(_w * s, _p, _rw, _t, _c, dev, 0.0, 0.0, 0.0)[1]

            ok_cal, sc, _pfr = calibrate(_rate)
            if not ok_cal:
                print(f"  {arm:5s} seed {sd} ✗ 标定失败", flush=True)
                continue
            tr, cue_tr = make_delay_task(args.train_seq, args.seq, 10 + sd, dev)
            te, cue_te = make_delay_task(args.test_seq, args.seq, 20 + sd, dev)
            toks = torch.cat([tr, te])
            cue = torch.cat([cue_tr, cue_te])
            for lv in LEVELS:
                if lv == 0.0:
                    alive = None
                else:
                    g = torch.Generator().manual_seed(7000 + sd * 31 + int(lv * 100))
                    alive = (torch.rand(N, generator=g) >= lv).float().to(dev)
                feat, fr = run_md(w_in * sc, pre, wrec, toks, cue, dev, 0.0, 0.0, 0.0, alive=alive)
                acc, _ = mlp_probe(feat[:, 1], cue, args.train_seq, K_CUE)
                rows[f"{lv:.2f}"].append(acc)
                frs[f"{lv:.2f}"].append(fr)
                del feat
                torch.cuda.empty_cache()
            del toks, cue, w_in, pre, wrec
            torch.cuda.empty_cache()
        res["cells"][arm] = {
            "acc": {k: (sum(v) / len(v) if v else float("nan")) for k, v in rows.items()},
            "runs": rows,
            "firing": {k: (sum(v) / len(v) if v else float("nan")) for k, v in frs.items()},
        }
        curve = [round(res["cells"][arm]["acc"][f"{lv:.2f}"], 3) for lv in LEVELS]
        fcs = [round(res["cells"][arm]["firing"][f"{lv:.2f}"], 4) for lv in LEVELS]
        print(f"  {arm:5s} 峰值 {curve}  发放 {fcs}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    if "sheet" not in cc or "space" not in cc:
        print("  ✗ 有效臂不足")
        return 1
    thresh = floor + 0.10
    # **只在 0% 那一档**要求工作点 —— 损伤本来就会降低发放
    l0 = all(cc[a]["acc"]["0.00"] >= thresh for a in cc) and all(
        0.03 <= v["firing"]["0.00"] <= 0.20 for v in cc.values()
    )
    print(f"  L-0 0% 损伤峰值 ≥ {thresh:.3f}、工作点 => {'过' if l0 else '不过 => 作废'}")
    if not l0:
        return 0
    l1 = all(cc[a]["acc"][lv] >= thresh for a in cc for lv in ("0.25", "0.50"))
    print(
        f"  L-1 25%/50% 损伤后仍 ≥ {thresh:.3f} => {'过' if l1 else '不过'}  "
        f"(sheet {[round(cc['sheet']['acc'][k], 3) for k in ('0.25', '0.50')]}, "
        f"space {[round(cc['space']['acc'][k], 3) for k in ('0.25', '0.50')]})"
    )
    diffs = [
        b - a
        for a, b in zip(cc["sheet"]["runs"]["0.50"], cc["space"]["runs"]["0.50"], strict=False)
    ]
    pos = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    print(f"  L-2 50% 损伤时 space-sheet 逐种子 {[round(d, 4) for d in diffs]}")
    print(f"        同号 {max(pos, n - pos)}/{n} => {'过' if max(pos, n - pos) == n else '不过'}")
    for lv in ("0.25", "0.50"):
        ds = [b - a for a, b in zip(cc["sheet"]["runs"][lv], cc["space"]["runs"][lv], strict=False)]
        print(f"        （{lv} 效应量 {(sum(ds) / len(ds)) * 100:+.2f}pp）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
