"""发布前禁词扫描：**公开仓不得出现「生命」「意识」**（本地不受限，见用户 2026-11-17 政策）。

用法：
    python scripts/check_publish_firewall.py          # 扫全仓
    python scripts/check_publish_firewall.py --staged # 只扫已暂存的改动

**政策（用户 2026-11-17）**：
- 「千岁」「Semprea」**不是**禁词；
- 「生命」「意识」在**公开仓**禁用，**本地不受限**；
- 所以这是一道**发布闸门**，不是写代码时的禁令 —— 本地写、发布前扫。
"""

from __future__ import annotations

import argparse
import pathlib as _pathlib
import subprocess
import sys

BANNED = ("生命", "意识")
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "dist", "build", ".ruff_cache"}
TEXT_EXT = {".md", ".py", ".json", ".txt", ".yml", ".yaml", ".toml", ".html", ".css", ".js"}
#: **治理文档豁免**：它们必须能写出禁词才能禁止它们。
#: 这些文件本身在「红线清单（不发布）」里，不是发布物。
WHITELIST = {
    "docs/consciousness-framework.md",
    "docs/consciousness-framework.ja.md",
    "docs/adr/ADR-0036-consciousness-claims-discipline.md",
    "docs/adr/ADR-0036-consciousness-claims-discipline.en.md",
    "docs/adr/ADR-0036-consciousness-claims-discipline.ja.md",
    "scripts/check_publish_firewall.py",
    "AGENTS.md",
}


def iter_files(root: _pathlib.Path, staged: bool):
    """**扫描对象 = 会被发布的文件集**（`git ls-files`），不是文件系统。

    2026-11-17 修正：原先用 `rglob` 扫盘，把 `.gitignore` 里的本地留档也算成公开内容，
    于是"把原始输出移出版本控制"这一步在扫描里看不见。防火墙要量的是**仓库内容**。
    """
    cmd = ["git", "diff", "--cached", "--name-only"] if staged else ["git", "ls-files"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=root).stdout
    for line in out.splitlines():
        p = root / line.strip()
        if not p.is_file() or p.suffix not in TEXT_EXT:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--root", type=_pathlib.Path, default=_pathlib.Path("."))
    args = ap.parse_args()
    hits: list[tuple[str, int, str]] = []
    for p in iter_files(args.root, args.staged):
        if p.as_posix().removeprefix("./") in WHITELIST:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for w in BANNED:
                if w in line:
                    hits.append((str(p), i, line.strip()[:90]))
    if not hits:
        print("✓ 禁词扫描通过（生命/意识 零命中）")
        return 0
    print(f"✗ 命中 {len(hits)} 处（公开前必须清除；本地可保留）：")
    for f, i, line in hits[:40]:
        print(f"  {f}:{i}  {line}")
    if len(hits) > 40:
        print(f"  …… 另有 {len(hits) - 40} 处")
    return 1


if __name__ == "__main__":
    sys.exit(main())
