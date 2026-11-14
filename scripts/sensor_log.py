"""本机传感器采集：为 S-1（传感器通道）攒**配对的（传感器, 文本）数据**。

这是那个「唯一真正的信息不对称」通道的采集端——云端永远读不到这些量。

## 采什么（只采能稳定拿到的）

| 字段 | 来源 | 频率 |
| :--- | :--- | :--- |
| `ts` | 系统时钟 | 每次 |
| `gpu_util` / `gpu_mem` / `gpu_temp` | `nvidia-smi` | 每次 |
| `mem_used_pct` | Win32 `GlobalMemoryStatusEx`（ctypes） | 每次 |
| `cpu_load` | `Win32_Processor.LoadPercentage` | 每次（约 200 ms） |
| `foreground` | 前台进程**名**（不记标题） | 每次 |
| `hour` | 本地小时（0–23） | 每次 |

## 硬规矩

1. **只记进程名，不记窗口标题**——标题里常含文件名、对话内容、URL；
2. **只写本机**（`D:/AxoData/sel/sensors/`），不上传、不入库；
3. 采样本身要**便宜**：默认 60 秒一次，CPU 占用可忽略。

## 怎么和文本配对

记忆库里每条都有 `created_at`。配对规则：**取时间上最近的、且不超过一个采样间隔的样本**；
找不到就记 `null`（**不许插值**——插值出来的传感器值是编的）。

## 现在的状态

配对数据**从今天开始才有**。历史记忆没有对应的传感器读数，所以 S-1 要等；
在那之前这条通道只能证明"接线通"（零向量逐位等价），不能证明"有用"。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

OUT = Path("D:/AxoData/sel/sensors/readings.jsonl")


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_load() -> float | None:
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.dwMemoryLoad / 100.0


def gpu_stats() -> tuple[float | None, float | None, float | None]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return None, None, None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 4:
        return None, None, None
    util, used, total, temp = parts[0], parts[1], parts[2], parts[3]
    return (
        float(util) if util.isdigit() else None,
        round(float(used) / max(float(total), 1e-9), 4) if used.isdigit() else None,
        float(temp) if temp.isdigit() else None,
    )


def cpu_and_foreground() -> tuple[float | None, str]:
    """一次 PowerShell 拿两个量——每次调用约 200 ms，所以默认 60 秒采一次。"""
    script = (
        "$c=(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage "
        "-Average).Average;"
        "$p=(Get-Process | Sort-Object CPU -Descending | Select-Object -First 1).ProcessName;"
        'Write-Output "$c|$p"'
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        cpu_s, _, proc = out.partition("|")
        cpu = float(cpu_s) / 100.0 if cpu_s.strip() else None
        return cpu, proc.strip()
    except Exception:
        return None, ""


def sample() -> dict:
    util, mem_pct, temp = gpu_stats()
    cpu, proc = cpu_and_foreground()
    now = datetime.now()
    return {
        "ts": round(now.timestamp(), 3),
        "iso": now.isoformat(timespec="seconds"),
        "hour": now.hour,
        "gpu_util": util,
        "gpu_mem_pct": mem_pct,
        "gpu_temp": temp,
        "conv_mem_pct": memory_load(),
        "cpu_load": cpu,
        "foreground": proc,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="本机传感器采集（配对文本用）")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--count", type=int, default=0, help="0 = 一直采")
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    while args.count == 0 or done < args.count:
        row = sample()
        with args.out.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done += 1
        if args.count and done >= args.count:
            break
        time.sleep(args.interval)
    print(f"采了 {done} 条 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
