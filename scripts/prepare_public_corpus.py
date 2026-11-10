"""公开语料：下载、清洗、切成项目 schema（数据不入库）。

计划书 §12.1 的纪律照旧：**本仓库不镜像数据**，仓库里只有这个脚本与它的登记信息；
语料落到仓外 ``D:/AxoData/sel/lm/v1/public/``，脚本打印每个产物的 sha256 与统计量。

## 首份语料：KdConv（Apache-2.0）

THU-CoAI 的中文多领域知识驱动对话（ACL 2020），4.5K 段对话 / 86K 句，三领域 film / music / travel。
选它的三件事：

1. **规模正好**：清洗后约 1.7M 字，落在单臂 ~2–3.5 小时的预算里；
2. **自带逐句知识标注**：每条助手回复挂着 `{name, attrname, attrvalue}` 三元组——
   后面"知识探针"的唯一现成来源；
3. **许可干净**：Apache-2.0，且数据只在仓外落盘。

## 产出

- ``dialogue.jsonl``：``{"session": "train"|"test", "user": ..., "assistant": ...}``。
  **直接用官方的 train/test 切分**，折名就是 split 名——`run_r0 --sessions test`
  会拿 train 训、test 测，**runner 一行都不用改**。
- ``kb.jsonl``：``{"split", "name", "attrname", "attrvalue"}``，去重后的知识三元组，供探针用。
- ``manifest.json``：两个产物的 sha256、字数、唯一字数、每对均长（**原始值**；
  模型词表是训练时按频率截的，另报）。

用法::

    uv run python scripts/prepare_public_corpus.py            # 下载 + 转换 + 打印统计
    uv run python scripts/prepare_public_corpus.py --verify-only   # 只重算哈希
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("D:/AxoData/sel/lm/v1/public")
BASE = "https://raw.githubusercontent.com/thu-coai/KdConv/master/data"
DOMAINS = ("film", "music", "travel")
SPLITS = ("train", "test")


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """把一段对话切成 (user, assistant, attrs)，按 user→assistant 交替。"""
    messages = conversation.get("messages", [])
    out = []
    for index in range(0, len(messages) - 1, 2):
        user = (messages[index].get("message") or "").strip()
        reply = messages[index + 1]
        assistant = (reply.get("message") or "").strip()
        if not user or not assistant:
            continue
        out.append((user, assistant, reply.get("attrs") or []))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="公开语料下载与转换（KdConv）")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--corpus", choices=("kdconv",), default="kdconv")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = args.out_root / args.corpus / "raw"
    out_dir = args.out_root / args.corpus
    manifest_path = out_dir / "manifest.json"

    if args.verify_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, entry in manifest["files"].items():
            actual = _sha256(out_dir / name)
            status = "OK" if actual == entry["sha256"] else "MISMATCH"
            print(f"{status} {name} {actual}")
        return 0

    rows: list[dict] = []
    triples: dict[tuple[str, str, str, str], None] = {}
    for domain in DOMAINS:
        for split in SPLITS:
            path = raw_dir / domain / f"{split}.json"
            _download(f"{BASE}/{domain}/{split}.json", path)
            conversations = json.loads(path.read_text(encoding="utf-8"))
            kept = 0
            for conversation in conversations:
                for user, assistant, attrs in _pairs(conversation):
                    rows.append({"session": split, "user": user, "assistant": assistant})
                    kept += 1
                    for attr in attrs:
                        name = (attr.get("name") or "").strip()
                        relation = (attr.get("attrname") or "").strip()
                        value = (attr.get("attrvalue") or "").strip()
                        if name and relation and value:
                            triples[(split, name, relation, value)] = None
            print(f"[{domain}/{split}] 段 {len(conversations)}，对 {kept}", flush=True)

    dialogue_path = out_dir / "dialogue.jsonl"
    dialogue_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    kb_path = out_dir / "kb.jsonl"
    kb_path.write_text(
        "".join(
            json.dumps(
                {"split": split, "name": name, "attrname": relation, "attrvalue": value},
                ensure_ascii=False,
            )
            + "\n"
            for (split, name, relation, value) in sorted(triples)
        ),
        encoding="utf-8",
        newline="\n",
    )

    lengths = [len(row["user"]) + len(row["assistant"]) for row in rows]
    chars = sum(lengths)
    unique = len({char for row in rows for char in row["user"] + row["assistant"]})
    per_split = {split: sum(1 for row in rows if row["session"] == split) for split in SPLITS}
    manifest = {
        "corpus": "kdconv",
        "license": "Apache-2.0",
        "source": "https://github.com/thu-coai/KdConv",
        "files": {
            "dialogue.jsonl": {
                "sha256": _sha256(dialogue_path),
                "bytes": dialogue_path.stat().st_size,
            },
            "kb.jsonl": {"sha256": _sha256(kb_path), "bytes": kb_path.stat().st_size},
        },
        "stats": {
            "pairs": len(rows),
            "pairs_per_split": per_split,
            "chars": chars,
            "chars_per_pair": round(chars / max(len(rows), 1), 2),
            "chars_per_pair_median": statistics.median(lengths),
            "unique_chars_raw": unique,
            "kb_triples": len(triples),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=1))
    print(f"写入 {dialogue_path}")
    print(f"写入 {kb_path}")
    print(f"写入 {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
