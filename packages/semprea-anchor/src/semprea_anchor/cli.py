"""Semprea-Anchor 的可分发命令行：**在任何机器上建库、问身份、看状态**。

三条"换台机器也能跑"的硬要求：

1. **零绝对路径**——模型与人格都在包内，用 ``__file__`` 相对定位；用户数据根由 ``--root`` 给；
2. **有 GPU 用 GPU，没有就 CPU**——``--device auto`` 是默认值；
3. **不需要任何宿主程序**——只依赖 ``transformers`` / ``torch`` 这两个公开包。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
MODEL_NAME = "Semprea-Anchor-0.5B"


def _find_model() -> Path:
    """**不写死路径**：从本文件往上找最近的 `model/`。

    开发布局是 `packages/semprea-anchor/model/`，分发布局是 `semprea-anchor-v1/model/`，
    同一个 CLI 要在两种布局下都找得到。
    """
    for base in (PKG, *PKG.parents[:3]):
        cand = base / "model"
        if cand.is_dir():
            return cand
    return PKG.parent / "model"


DEFAULT_MODEL = _find_model()


def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load(model_dir: Path, dev: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(dev).eval()
    return tok, model


def cmd_init(args) -> int:
    from persona_core import init_memory

    info = init_memory(Path(args.root))
    print(json.dumps(info, ensure_ascii=False))
    return 0


def cmd_state(args) -> int:
    from persona_core import SelfState

    s = SelfState(
        backbone=MODEL_NAME,
        memory_count=_count(args.root),
        persona_version=int(args.persona_version),
        persona_in_weights=True,
        online=False,
    )
    print(s.render())
    print([round(x, 3) for x in s.to_vector()])
    return 0


def _count(root: str) -> int:
    import sqlite3

    db = Path(root) / "memory" / "mem.db"
    if not db.is_file():
        return 0
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return int(con.execute("select count(*) from memories").fetchone()[0])
    finally:
        con.close()


def cmd_ask(args) -> int:
    import torch
    from persona_core import Influence, SelfState

    dev = _device(args.device)
    tok, model = _load(Path(args.model), dev)
    st = SelfState(
        backbone=MODEL_NAME,
        memory_count=_count(args.root),
        persona_version=int(args.persona_version),
        persona_in_weights=True,
        online=False,
    )
    block = Influence(self_state=st).render_prompt_block()
    msgs = []
    if args.show_state:
        msgs.append({"role": "system", "content": block})
    msgs.append({"role": "user", "content": args.prompt})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip())
    return 0


def cmd_manifest(args) -> int:
    from persona_core import manifest

    print(json.dumps(manifest(Path(args.root)), ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="semprea-anchor")
    ap.add_argument("--root", default=str(Path.home() / "semprea"))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--persona-version", default="8")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="建记忆库与实例身份").set_defaults(fn=cmd_init)
    sub.add_parser("state", help="打印她此刻的状态").set_defaults(fn=cmd_state)
    sub.add_parser("manifest", help="写 MANIFEST.json").set_defaults(fn=cmd_manifest)
    p_ask = sub.add_parser("ask", help="问一句")
    p_ask.add_argument("prompt")
    p_ask.add_argument("--max-new-tokens", type=int, default=64)
    p_ask.add_argument(
        "--show-state", action="store_true", help="把状态块也塞进提示词（默认不塞：身份在权重里）"
    )
    p_ask.set_defaults(fn=cmd_ask)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
