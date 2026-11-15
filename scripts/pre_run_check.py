"""长跑前的机器检查：**训练期间这张卡上不许跑别的东西**。

`plan-v1-training-budget.md`（2026-11-15）写死的第一条硬约束：

| 条件 | 吞吐 |
| :--- | ---: |
| GPU 空闲 | **241 tok/s** |
| 同卡有别的进程 | **16 tok/s**（掉 **15×**） |

2026-11-16 一整天我都在争用状态下跑训练（实测 28–95 tok/s），把 48 小时的账算错了。
本脚本把这条约束变成**可执行的检查**，而不是靠记性。

用法：

    python scripts/pre_run_check.py            # 只看
    python scripts/pre_run_check.py --strict   # 有占用就返回非零，长跑脚本可以 `&&` 起来
"""

from __future__ import annotations

import argparse
import subprocess
import sys

#: 这些进程会吃显存，跑长训前应当关掉
OFFENDERS = ("llama-server", "ollama", "python", "chrome", "msedge", "wechat", "qq")
#: 但 python 里可能就有我们自己，所以单独提示、不直接判红


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-busy", action="store_true", help="明知有占用也放行")
    ap.add_argument("--min-free-mib", type=int, default=3000, help="至少要留这么多空闲显存")
    args = ap.parse_args()

    ok = True
    smi = _run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader"]
    )
    if not smi.strip():
        print("✗ 拿不到 nvidia-smi 输出")
        return 1
    total, used, free = (int(x.split()[0]) for x in smi.strip().split(","))
    print(f"显存：总 {total} MiB / 已用 {used} / 空闲 {free}")
    if free < args.min_free_mib:
        print(f"  ✗ 空闲 {free} MiB < {args.min_free_mib} MiB —— 长跑会掉进换页区")
        ok = False
    else:
        print(f"  ✓ 空闲 ≥ {args.min_free_mib} MiB")

    apps = _run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    ).strip()
    hits = []
    if apps:
        for line in apps.splitlines():
            low = line.lower()
            for name in OFFENDERS:
                if name in low:
                    hits.append(line.strip())
                    break
    print(f"吃显存的进程：{len(apps.splitlines()) if apps else 0} 个")
    for h in hits[:10]:
        print(f"  ⚠ {h}")
    if hits:
        print("  ↑ 长训前必须关掉这些（训练期间这张卡上不许跑别的东西，15× 的差别）")
        ok = False

    print("\n判：", "✅ 可以开长跑" if ok else "❌ 先清干净再跑")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
