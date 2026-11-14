"""把 Semprea-4.5B 的**一版**打成一个可发布、可核对的包。

为什么要有这个脚本，而不是手敲一堆 `cp`：

1. **"包里必须有哪几件"是写死的**（`plan-semprea-versions.md` §二）——少一件，V2 就接不上，
   或者用户在干净机器上起不来。手抄迟早漏；
2. **每一件都要有 sha256**。发布出去的东西，下载回来不能核对等于没发；
3. **许可没核实就不许标 `publishable`**。这是 §九 T-5，脚本负责把它变成机器可读的字段，
   而不是某个人记得要检查。

用法::

    uv run python scripts/package_semprea.py --version 1.0 --dry-run
    uv run python scripts/package_semprea.py --version 1.0 --out D:/Semprea-4.5B/dist/v1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path("D:/AxoData/sel/lm")
DEFAULT_OUT = Path("D:/Projects/Semprea-4.5B/dist")

#: 包里**必须**有的三件（另外的由本脚本生成）。
#: 这是 `plan-semprea-versions.md` §二那张表的机器版本。
#:
#: **`id_map.json` 不在里面**：它只是 `subset_32k.json` 的反向索引
#: （`ids[i]` 就是本地 id `i` 对应的千问 id），重复存一份必然有一天两边对不上。
#: 打包时**从 subset 现生成**一份自描述的 id_map，让下游不用猜。
REQUIRED = {
    "ckpt": "快半检查点",
    "tree": "冻结的树（V2 续训的承重件）",
    "subset": "32k 子集在千问词表里的 id 序列",
}

#: 只在发布时存在的东西；缺了**不阻止打包**，但 manifest 里会写 null。
OPTIONAL = {
    "user_head": "用户头冷启动快照（.npz）",
    "gguf": "慢半的千问 GGUF（体积大，通常放 ModelScope）",
    "emb_init": "词嵌入几何迁移的产物",
}

#: 这些路径出现过就说明有人把**专有件**塞进包里了（气体传感器插件在仓外且专有）。
FORBIDDEN_NAMES = ("gassensor", "gas_sensor", "BioSNN-GasSensor")

#: 运行层的能力声明（B4）。**它是契约，不是说明**：
#: 下游（Axo 或任何 OpenAI 客户端）靠它判断"这个包能不能对话、能不能出向量"。
#: 改了运行层的档位或端点，这里必须同步改，否则 manifest 会说谎。
RUNTIME_DECL = {
    "name": "semprea",
    "version": "1.0.0",
    "entry": "python -m semprea --config semprea.toml",
    "default_port": 18740,
    "source": "runtime/（运行层与权重层分开发布，B3）",
    "endpoints": [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/health",
        "/v1/models",
    ],
    "slow_backends": ["llama-server（包内/系统）", "ollama", "openai-compatible"],
    "tiers": ["full", "full-cpu", "chat-only", "fast-only", "offline"],
    "capabilities": {"chat": "需要慢半", "embeddings": "需要快半", "fusion": "需要两半同时在"},
    "alpha_zero_contract": "alpha=0 时运行时原样转发慢半的响应字节（T-3）",
    "self_contained": {
        "fast_half": "runtime/semprea/fast/（原样 vendor 自 BioSNN-Plug，保真由 tools/verify_fast_parity.py 核对）",
        "launch": "[fast] ckpt 给了就由运行层拉起快半子进程；torch 不进主层",
        "watchdog": "父进程死亡 ⇒ 子进程随 stdin EOF 退出（防孤儿占端口与显存）",
    },
    "requires": {
        "python": ">=3.11（主运行层零第三方依赖；自带快半的内核需要 torch + tokenizers）",
        "fast_half_service": "torch + tokenizers（只在跑快半的那个进程里）",
        "slow_half": "llama-server / Ollama / 任意 OpenAI 兼容端点",
    },
    "weight_dtype": "auto（GPU⇒bf16，CPU⇒fp32；GPU+fp32 是慢 30 倍的组合，不许 auto 落到它上面）",
    "deployment_note": (
        "两半都在同一张 8 GB GPU 上时，快半读出权重（1.9 GB fp32）把显存顶到换页边缘，"
        "逐 token 融合每切一次 1.1–1.5 秒。改 bf16（0.95 GB）后降到 0.02–0.03 秒，"
        "top-1 10/10 不变、最大 logprob 差 0.0071。运行层的 weight_dtype=auto 因此规定"
        "GPU 用 bf16、CPU 用 fp32。详见 runtime/README.md。"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
            ).strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def collect(args) -> tuple[dict[str, Path], list[str]]:
    """返回 (存在的件 -> 路径, 缺件说明)。专有件一出现就直接拒。"""
    wanted = dict(REQUIRED)
    wanted.update(OPTIONAL)
    found: dict[str, Path] = {}
    missing: list[str] = []
    for key, label in wanted.items():
        raw = getattr(args, key, None)
        path = Path(raw) if raw else None
        if path is None:
            if key in OPTIONAL:
                continue
            missing.append(f"{key}（{label}）：没有给路径")
            continue
        if not path.exists():
            if key in OPTIONAL:
                continue
            missing.append(f"{key}（{label}）：{path} 不存在")
            continue
        lowered = str(path).lower()
        for bad in FORBIDDEN_NAMES:
            if bad.lower() in lowered:
                raise SystemExit(f"拒绝打包：{path} 看起来是专有件（{bad}），不许进模型仓")
        found[key] = path
    return found, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打包 Semprea-4.5B 的一版")
    ap.add_argument("--version", required=True, help="语义版本，如 1.0")
    ap.add_argument(
        "--ckpt", type=Path, default=REPO / "sweep_results/lm_semprea_v1/semprea_v1_ckpt.pt"
    )
    ap.add_argument(
        "--tree", type=Path, default=REPO / "sweep_results/lm_semprea_v1/semprea_v1_tree.pt"
    )
    ap.add_argument(
        "--subset",
        type=Path,
        default=DATA / "corpus/qwen_tok/subset_32k.json",
        help="32k 子集在千问词表里的 id 序列",
    )
    ap.add_argument("--user-head", type=Path, default=None)
    ap.add_argument("--gguf", type=Path, default=None)
    ap.add_argument("--emb-init", type=Path, default=None)
    ap.add_argument("--eval-json", type=Path, default=None, help="评测结果（写进 model card）")
    ap.add_argument(
        "--served-by", default="", help="服务启动参数原样记下，如 'serve_snn.py --steps 100000'"
    )
    ap.add_argument(
        "--corpus-license",
        default="",
        help="语料的许可（如 CC-BY-4.0）。**留空则 publishable=false**",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="只检查缺什么，不落盘")
    args = ap.parse_args(argv)

    found, missing = collect(args)
    if missing:
        print("缺件：", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1

    files = {}
    for key, path in found.items():
        size = path.stat().st_size
        files[key] = {
            "name": path.name,
            "bytes": size,
            "sha256": sha256(path),
            "source": str(path),
        }
        print(f"  {key:<10} {size / 1e6:>10.1f} MB  {files[key]['sha256'][:16]}  {path.name}")

    publishable = bool(args.corpus_license.strip())
    manifest = {
        "name": "Semprea-4.5B",
        "version": args.version,
        "format": 1,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_on": f"{platform.system()} {platform.machine()}",
        "source_commit": git_commit(),
        "parts": {
            "slow": {"name": "Qwen3-4B", "license": "Apache-2.0", "frozen": True},
            "fast": {"name": "snn-0.5b", "license": "Apache-2.0", "frozen": False},
        },
        "files": files,
        "served_by": args.served_by,
        "licensing": {
            "code": "Apache-2.0",
            "corpus": args.corpus_license.strip(),
            "note": "" if publishable else "语料许可未核实：只可放私有仓（plan §九 T-5）",
        },
        "publishable": publishable,
        "runtime": RUNTIME_DECL,
        "eval": {},
    }
    if args.eval_json and Path(args.eval_json).exists():
        manifest["eval"] = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=1)[:1200])
        print(f"\n[dry-run] publishable={publishable}")
        return 0

    target = args.out / f"v{args.version}"
    target.mkdir(parents=True, exist_ok=True)
    for path in found.values():
        shutil.copy2(path, target / path.name)
    # id_map 从 subset 现生成：`ids[i]` 就是本地 id `i` 对应的千问 id。
    subset = json.loads(found["subset"].read_text(encoding="utf-8"))
    id_map = target / "id_map.json"
    id_map.write_text(
        json.dumps({"note": "local_id -> qwen_id", "ids": list(subset["ids"])}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # **分词器本体必须进包**：`subset_32k.json` 只是一张 id 表，真正的
    # ByteLevel BPE 由 `vocab.json` + `merges.txt` 决定。少了它们，服务在干净机器上
    # 连分词器都建不起来——T-4（包自足）正是这么抓出来的。
    for name in ("vocab.json", "merges.txt"):
        source = found["subset"].parent / name
        if not source.exists():
            raise SystemExit(f"包不完整：{source} 不存在，子集分词器建不起来")
        shutil.copy2(source, target / name)
        files[name.replace(".", "_")] = {
            "name": name,
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "source": str(source),
        }
    files["idmap"] = {
        "name": id_map.name,
        "bytes": id_map.stat().st_size,
        "sha256": sha256(id_map),
        "source": "generated from subset",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    total = sum(item["bytes"] for item in files.values())
    print(f"\n打包到 {target}：{len(files)} 件，{total / 1e9:.2f} GB")
    print(f"publishable = {publishable}")
    if not publishable:
        print("⚠ 语料许可未核实 ⇒ 只放私有仓；核实后加 --corpus-license 重打", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
