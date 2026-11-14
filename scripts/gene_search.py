"""B-4「基因」：外循环搜架构，内循环冻结。

把架构超参当基因组（池大小 / 池膜 τ / 读出时间尺度数 / 驱动尺度 / 激素路由），
**随机抽样 + 留出评测**，权重一次都不更新（`lr-sensor=0`，纯编码）。

判据（预注册 `research/sel/bio-layers-v1.md` §三 B-4）：搜到的最好个体相对手设计基线 < +3pp 则关闭。
"""

from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

OUT = Path("sweep_results/gene_search")
POOL_SIZES = (8, 16, 32)
TAUS = (5, 20, 80)
READOUTS = ("2,128", "2,8,32,128")
SCALES = (0.5, 1.0, 2.0, 4.0)
ROUTES = (0, 3)
BASELINE = {"pool_size": 16, "tau_pool": 20, "readout": "2,128", "scale": 2.0, "route": 0}


def run(cfg: dict, pool: str = "temp", steps: int = 200, seed: int = 0) -> dict | None:
    cmd = [
        sys.executable,
        "-m",
        "research.sel.lm.train_sensor_task",
        "--arm",
        "add",
        "--pools",
        pool,
        "--encoder",
        "design",
        "--lr-sensor",
        "0",
        "--clip",
        "0",
        "--steps",
        str(steps),
        "--eval-every",
        str(steps),
        "--eval-batches",
        "10",
        "--pool-size",
        str(cfg["pool_size"]),
        "--readout-taus",
        cfg["readout"],
        "--pool-scales",
        f"{pool}:{cfg['scale']}",
        "--tag",
        "gene",
        "--seed",
        str(seed),
    ]
    if cfg["tau_pool"] != 20:
        cmd += ["--tau-pool", str(cfg["tau_pool"])]
    if cfg["route"]:
        cmd += ["--route", str(cfg["route"])]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if not line:
        return None
    d = json.loads(line[-1])
    return {
        "cfg": cfg,
        "top1": d["hit"][pool],
        "rate": d["rate"][pool],
        "cv": d["rate_cv"][pool],
        "seed": seed,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    space = [
        {"pool_size": a, "tau_pool": b, "readout": c, "scale": d, "route": e}
        for a, b, c, d, e in itertools.product(POOL_SIZES, TAUS, READOUTS, SCALES, ROUTES)
    ]
    rng = random.Random(20261114)
    sample = rng.sample(space, 48)
    if BASELINE not in sample:
        sample = [BASELINE, *sample[:47]]
    rows = []
    for i, cfg in enumerate(sample):
        r = run(cfg)
        if r is None:
            continue
        rows.append(r)
        print(f"  {i + 1:2d}/48  top1 {r['top1']:.3f}  {cfg}", flush=True)
    rows.sort(key=lambda x: -x["top1"])
    base = next((r["top1"] for r in rows if r["cfg"] == BASELINE), None)
    out = {
        "baseline": base,
        "best": rows[0],
        "n": len(rows),
        "gain_pp": (rows[0]["top1"] - base) * 100 if base is not None else None,
        "top10": rows[:10],
    }
    (OUT / "search.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "top10"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
