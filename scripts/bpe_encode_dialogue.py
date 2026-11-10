"""把对话语料编码成 **BPE token 串**，喂给现有 `run_r0`（它只认「一串 token」）。

做法：每个 BPE token 映射成**一个独立码点**（从 U+E000 起），于是"token 串"就是"字符串"，
`run_r0` 一行不用改。这样 top-10 就是 **10 个 BPE 单元**（平均 1.7 字），
而不是 10 个字符 —— 这才是"10 格候选条"真正该看的数。

产物：``dialogue_bpe.jsonl``（同 schema：session / user / assistant），
以及 ``token_units.json``（每个 token 解码后的文本，用于算多字单元占比）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sel.lm.bpe import train_tokenizer

PUA = 0xE000


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对话语料 -> BPE token 串")
    ap.add_argument(
        "--src", type=Path, default=Path("D:/AxoData/sel/lm/v1/public/kdconv/dialogue.jsonl")
    )
    ap.add_argument("--out-dir", type=Path, default=Path("D:/AxoData/sel/lm/v1/public/kdconv_bpe"))
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--sessions", default="test", help="留作 test 的会话名")
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.src.read_text(encoding="utf-8").splitlines()]
    held = {s for s in args.sessions.split(",") if s}
    train_texts = [r["user"] + r["assistant"] for r in rows if r["session"] not in held]
    print(f"总 {len(rows)} 条，训练用 {len(train_texts)} 条", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample = args.out_dir / "train_sample.txt"
    sample.write_text("\n".join(train_texts), encoding="utf-8", newline="\n")
    tokenizer = train_tokenizer([sample], args.out_dir / "bpe", vocab_size=args.vocab)

    vocab = tokenizer.get_vocab()
    id_to_code = {i: chr(PUA + i) for i in range(len(vocab))}
    units = {str(PUA + i): tokenizer.decode([i]) for i in range(len(vocab))}
    (args.out_dir / "token_units.json").write_text(
        json.dumps(units, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    def to_codes(text: str) -> str:
        return "".join(id_to_code[i] for i in tokenizer.encode(text).ids)

    out_path = args.out_dir / "dialogue_bpe.jsonl"
    chars = tokens = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            coded = {k: to_codes(row[k]) for k in ("user", "assistant") if row.get(k)}
            if not coded:
                continue
            chars += len(row.get("user", "")) + len(row.get("assistant", ""))
            tokens += len(coded.get("user", "")) + len(coded.get("assistant", ""))
            handle.write(
                json.dumps({"session": row["session"], **coded}, ensure_ascii=False) + chr(10)
            )
    print(f"写入 {out_path}")
    print(
        json.dumps(
            {"chars": chars, "tokens": tokens, "chars_per_token": round(chars / max(tokens, 1), 3)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
