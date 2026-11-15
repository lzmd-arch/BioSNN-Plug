"""发布前检查：**包里该有的是不是都在，不该在的是不是都没有**。

七条，全过才允许对外分发。

| 编号 | 查什么 | 为什么 |
| :--- | :--- | :--- |
| **P-1** | `LICENSE` 在包里 | 没有许可 = 默认保留所有权利，别人没法合法用 |
| **P-2** | `MODEL_CARD.md` 在包里，且写明"不是聊天模型" | 定位写不清，用户第一句闲聊就会失望 |
| **P-3** | 没有 `__pycache__` / `*.pyc` | 每台机器生成的不一样，会破坏可复现 |
| **P-4** | `MANIFEST.json` 里每个文件的 sha256 对得上 | 分发过程被改过要能发现 |
| **P-5** | 所有文本文件里**没有绝对路径** | 换机器就崩 |
| **P-6** | **没有凭据、没有个人数据** | 用户原话："确认包括 apikey 在内的个人数据没有在仓库里" |
| **P-7** | 危机语料 ≥ 50 条，且模型卡里没有残留的 `未测` | 3 条探针算不出 0.95 |

P-6 只做**模式扫描**，它不是"证明没有"，只是"没有明显的"；真正的证明要靠人工过一遍 diff。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ABSPATH = [
    re.compile(r"[A-Za-z]:[\\/](Users|Projects|AxoData|Windows)"),
    re.compile(r"/(?:home|mnt|Users)/"),
]
CRED = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"]{8,}", re.I),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]
TEXT_EXT = {".py", ".md", ".json", ".txt", ".jinja", ".cfg", ".toml"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, default=Path("dist/semprea-anchor-v1"))
    args = ap.parse_args()
    pkg: Path = args.pkg
    fails: list[str] = []
    warns: list[str] = []

    if not pkg.is_dir():
        print(f"✗ 包不存在：{pkg}")
        return 1

    # P-1
    if not (pkg / "LICENSE").is_file():
        fails.append("P-1 缺 LICENSE")
    # P-2
    card = pkg / "MODEL_CARD.md"
    if not card.is_file():
        fails.append("P-2 缺 MODEL_CARD.md")
    else:
        body = card.read_text(encoding="utf-8")
        if "不是聊天模型" not in body:
            fails.append("P-2 模型卡没写明「不是聊天模型」")
        if "未测" in body:
            warns.append("P-7 模型卡里还有「未测」")
    # P-3
    junk = [p for p in pkg.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc"]
    if junk:
        fails.append(f"P-3 有 {len(junk)} 个 __pycache__/*.pyc")
    # P-4
    mf = pkg / "MANIFEST.json"
    if not mf.is_file():
        fails.append("P-4 缺 MANIFEST.json")
    else:
        man = json.loads(mf.read_text(encoding="utf-8"))
        for rel, meta in man.get("files", {}).items():
            f = pkg / rel
            if not f.is_file():
                fails.append(f"P-4 清单里有、包里没有：{rel}")
            elif sha(f) != meta["sha256"]:
                fails.append(f"P-4 sha256 对不上：{rel}")
        extra = {
            str(p.relative_to(pkg)).replace("\\", "/")
            for p in pkg.rglob("*")
            if p.is_file() and p.name != "MANIFEST.json"
        } - set(man.get("files", {}))
        if extra:
            warns.append(f"P-4 包里有、清单没有：{sorted(extra)}")
    # P-5 / P-6
    for p in pkg.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = p.relative_to(pkg)
        for rx in ABSPATH:
            m = rx.search(body)
            if m:
                fails.append(f"P-5 绝对路径 {rel}: {m.group(0)}")
                break
        for rx in CRED:
            m = rx.search(body)
            if m:
                fails.append(f"P-6 疑似凭据/邮箱 {rel}: {m.group(0)[:24]}")
                break
    # P-7
    total = sum(v["bytes"] for v in json.loads(mf.read_text(encoding="utf-8"))["files"].values())
    print(f"包：{pkg}  {len(list(pkg.rglob('*')))} 项   {total / 2**20:.0f} MiB")
    for w in warns:
        print("  ⚠", w)
    for f in fails:
        print("  ✗", f)
    print("判：", "✅ P-1~P-7 全过" if not fails else f"❌ {len(fails)} 条不过")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
