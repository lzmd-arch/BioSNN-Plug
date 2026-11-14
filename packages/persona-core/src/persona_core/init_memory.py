"""**权重外记忆**的落地命令：给任何用户、任何宿主程序把记忆库建出来。

## 为什么要在打包时内置这条命令

融合模型的能力分两半：**权重里的是通用能力**（常识/伦理/心理学），
**权重外的是这个用户的记忆与人格**。权重可以随包分发，记忆不能——
所以包必须带一条"把记忆那半边建起来"的命令，否则用户拿到的只是一半。

## 通用性

这里**不 import 任何宿主程序**（不 import axo），只依赖标准库。
数据根由 `--root` 给定，默认 `~/persona-core`。任何产品都能调用：

    python -m persona_core init-memory --root /path/to/data
    python -m persona_core manifest  --root /path/to/data

第二条把"她由哪些文件组成"写成 MANIFEST.json（逐件 sha256），供打包与核对。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

#: 记忆库的表结构。**与 memcore 的 schema 同源**——那边的 `_SCHEMA` 是权威，
#: 这里复制一份是为了让本包不依赖宿主仓库；`tests` 里有一条守卫比对两边的列名。
SCHEMA = """
CREATE TABLE IF NOT EXISTS spaces (
    space_id   TEXT PRIMARY KEY,
    dim        INTEGER NOT NULL,
    model_ref  TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    memory_id       TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    tier            TEXT NOT NULL,
    content         TEXT NOT NULL,
    taint           TEXT NOT NULL,
    created_at      REAL NOT NULL,
    last_access_ts  REAL NOT NULL,
    last_decay_ts   REAL NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    reinforce_count INTEGER NOT NULL DEFAULT 0,
    weight          REAL NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0,
    space_id        TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    source_ref      TEXT NOT NULL DEFAULT '',
    meta_json       TEXT NOT NULL DEFAULT '{}',
    lex_idx         BLOB,
    lex_w           BLOB,
    lex_width       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memories_tier    ON memories(tier);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    space_id  TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL,
    added_at  REAL NOT NULL,
    PRIMARY KEY (memory_id, space_id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_space ON embeddings(space_id);
"""

#: 内置人格锚的来源指针。宿主程序可以换名字，`init-memory` 只保证**有一条**。
BUILTIN_REF = "persona://builtin"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _persona_text(package_dir: Path | None) -> str:
    """包里带的 persona.md；没有就用一句最小的占位（**不编造人格**）。"""
    package_dir = package_dir or Path(__file__).resolve().parent
    if package_dir is not None:
        md = package_dir / "persona.md"
        if md.is_file():
            return md.read_text(encoding="utf-8").strip()
    return "你是这个终端里的陪伴实体。记忆存在本地，不属于任何模型。"


def init_memory(root: Path, *, persona: str | None = None, package_dir: Path | None = None) -> dict:
    """建目录、建库、写一条内置人格锚、写实例身份。**幂等**：重复跑不会重复写。"""
    root = Path(root).expanduser()
    mem_dir = root / "memory"
    ident_dir = root / "identity"
    mem_dir.mkdir(parents=True, exist_ok=True)
    ident_dir.mkdir(parents=True, exist_ok=True)

    db = mem_dir / "mem.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)

    text = persona if persona is not None else _persona_text(package_dir)
    now = time.time()
    row = con.execute(
        "select memory_id, content from memories where kind='persona_anchor' and source_ref=?",
        (BUILTIN_REF,),
    ).fetchone()
    if row is None:
        con.execute(
            "insert into memories (memory_id, kind, tier, content, taint, created_at,"
            " last_access_ts, last_decay_ts, access_count, reinforce_count, weight, pinned,"
            " source_ref, meta_json) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                uuid.uuid4().hex,
                "persona_anchor",
                "longterm",
                text,
                "user",
                now,
                now,
                now,
                0,
                0,
                1.0,
                1,
                BUILTIN_REF,
                json.dumps({"persona_version": 1}),
            ),
        )
        created = True
    else:
        created = False
    con.commit()
    con.close()

    inst = ident_dir / "instance.json"
    if not inst.is_file():
        inst.write_text(
            json.dumps(
                {
                    "instance_id": uuid.uuid4().hex,
                    "created_at": now,
                    "birth_date": "2026-11-16",
                    "age_at_birth": 20,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return {
        "root": str(root),
        "db": str(db),
        "instance": str(inst),
        "persona_created": created,
        "memories": _count(db),
    }


def _count(db: Path) -> int:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return int(con.execute("select count(*) from memories").fetchone()[0])
    finally:
        con.close()


def manifest(root: Path) -> dict:
    """把"她由哪些文件组成"写成 MANIFEST.json（逐件 sha256）。"""
    root = Path(root).expanduser()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[str(path.relative_to(root)).replace("\\", "/")] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    out = {"root": str(root), "files": files, "built_at": time.time()}
    (root / "MANIFEST.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return {"root": str(root), "n_files": len(files), "manifest": str(root / "MANIFEST.json")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="persona_core")
    ap.add_argument("cmd", choices=("init-memory", "manifest"))
    ap.add_argument("--root", default=str(Path.home() / "persona-core"))
    ap.add_argument("--persona", default=None, help="覆盖内置人格正文（一般不用）")
    args = ap.parse_args(argv)
    pkg = Path(__file__).resolve().parent
    if args.cmd == "init-memory":
        print(
            json.dumps(
                init_memory(Path(args.root), persona=args.persona, package_dir=pkg),
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(manifest(Path(args.root)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
