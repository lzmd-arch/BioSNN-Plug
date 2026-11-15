"""把 V1 打成一个**能在任何机器上跑**的目录。

零绝对路径、零宿主依赖：模型、人格、persona-core 全在包内，用户数据由 `--root` 决定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "sweep_results/persona_train/ckpt_v4"
CORE = REPO / "packages/persona-core/src/persona_core"
SHELL = REPO / "packages/semprea-anchor/src/semprea_anchor"
PERSONA = REPO / "packages/persona-core/src/persona_core/persona.md"
FUSION = REPO / "research/sel/lm/fusion_model.py"
MODULATOR = REPO / "sweep_results/state_qa/modulator_v4_hand.pt"
DEMO = REPO / "packages/semprea-anchor/demo_state.py"
LICENSE = REPO / "LICENSE"
CARD = REPO / "packages/semprea-anchor/MODEL_CARD.md"

VERSIONS = {"model": "Semprea-Anchor-0.5B", "anchor": "v1", "persona": 8, "birth": "2026-11-16"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "dist/semprea-anchor-v1")
    ap.add_argument("--ckpt", type=Path, default=CKPT, help="用哪个检查点（默认 ckpt_v4）")
    ap.add_argument("--modulator", type=Path, default=MODULATOR)
    args = ap.parse_args()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copytree(args.ckpt, out / "model")
    shutil.copytree(CORE, out / "persona_core", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(SHELL, out / "semprea_anchor", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(PERSONA, out / "persona.md")
    shutil.copy2(FUSION, out / "fusion_model.py")
    shutil.copy2(DEMO, out / "demo_state.py")
    shutil.copy2(LICENSE, out / "LICENSE")
    if CARD.is_file():
        shutil.copy2(CARD, out / "MODEL_CARD.md")
    else:
        print("警告：没有 MODEL_CARD.md —— 公开发布前必须补")
    if args.modulator.is_file():
        shutil.copy2(args.modulator, out / "modulator.pt")
    else:
        print("警告：找不到调制器权重，demo_state 会跑不了")

    # 运行时会在包内生成 __pycache__，不能在清单里留下它（每台机器都不一样）
    for junk in list(out.rglob("__pycache__")) + list(out.rglob("*.pyc")):
        shutil.rmtree(junk) if junk.is_dir() else junk.unlink()

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[str(p.relative_to(out)).replace("\\", "/")] = {
                "bytes": p.stat().st_size,
                "sha256": _sha(p),
            }
    manifest = {
        "versions": VERSIONS,
        "files": files,
        "entry": "python -m semprea_anchor",
        "python": ">=3.10",
        "deps": ["torch", "transformers"],
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    total = sum(v["bytes"] for v in files.values())
    print(f"打包完成 → {out}")
    print(f"{len(files)} 个文件，{total / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
