"""本地编码服务：把训好的 SNN 包成一个**独立进程 + 本地 HTTP**，供 Axo 调用。

为什么是独立进程：Axo 有一条既定红线——**torch 不进它的依赖**（语音组件因此独立）。
本服务与 Ollama 完全同构：常驻、本地、不联网。

只实现两个端点。**不做对话**：这个模型记不住超过 ~64 步、没有对齐，
把它套成聊天后端等于在接口层面承诺一件它做不到的事。

::

    uv run python -m research.sel.lm.serve_snn --ckpt <检查点> --device cpu --port 18734
    curl -s localhost:18734/health
    curl -s localhost:18734/encode -d '{"texts": ["北岸一号站"]}'
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
BPE = Path("D:/AxoData/sel/lm/corpus/skypile_tok/bpe")


def load_model(ckpt: Path, bpe: Path, device: torch.device):
    """按检查点里记的配置重建模型。**形状对不上就报错，不静默降级。**"""
    from tokenizers import ByteLevelBPETokenizer

    from research.sel.lm.hierarchical import build_huffman_paths
    from research.sel.lm.model import TokenEProp

    blob = torch.load(ckpt, map_location="cpu")
    cfg = blob["config"]
    vocab = int(cfg["vocab"])
    width = int(float(cfg["params"])) // vocab
    n_rec = max(width // 2, 64)
    counts_path = ckpt.parent / f"counts_{vocab}.npy"
    if counts_path.exists():
        import numpy as np

        counts = torch.from_numpy(np.load(counts_path)).float()
    else:
        counts = torch.ones(vocab)
    paths = build_huffman_paths(counts).to(device)
    model = TokenEProp(
        vocab,
        n_rec,
        recurrence="none",
        readout_taus=(2.0, 128.0),
        tree=paths,
        tree_signal_scale=float(cfg["tree_signal_scale"]),
        tree_packed=True,
        embed_rank=int(cfg["embed_rank"]),
        tree_init_weight=False,
        generator=torch.Generator().manual_seed(0),
        device=device,
    )
    for key, value in blob["tensors"].items():
        if key == "__tree_weight":
            model.tree.weight = value.to(device)
        elif key == "__tree_bias":
            model.tree.bias = value.to(device)
        else:
            setattr(model, key, value.to(device))

    tokenizer = ByteLevelBPETokenizer(str(bpe / "vocab.json"), str(bpe / "merges.txt"))
    return model, tokenizer, int(blob["step"])


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], max_len: int = 256) -> torch.Tensor:
    """逐条编码（各自的 trace），返回 ``(n, M)`` 的 **L2 归一化** 矩阵。

    向量取**各位置读出特征的均值**——这是这个模型唯一有的句子级表示。
    """
    out = torch.zeros(len(texts), model.readout_width, dtype=torch.float32)
    for row, text in enumerate(texts):
        ids = tokenizer.encode(text).ids[:max_len]
        if len(ids) < 2:
            continue
        trace = model.new_trace(1)
        acc = torch.zeros(model.readout_width, device=model.device)
        for step in range(len(ids) - 1):
            model.step(
                torch.tensor([ids[step]], device=model.device),
                torch.tensor([ids[step + 1]], device=model.device),
                trace,
                update=False,
                all_valid=True,
            )
            acc += model._features(trace)[0]
        out[row] = (acc / (len(ids) - 1)).cpu()
    norms = out.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return out / norms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="本地 SNN 编码服务")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bpe", type=Path, default=BPE)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--port", type=int, default=18734)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, tokenizer, step = load_model(args.ckpt, args.bpe, device)
    space = f"snn/tree-0.5b-step{step}@{model.readout_width}"
    state = {"ready": True, "device": str(device), "step": step, "space_id": space}
    lock = threading.Lock()
    print(f"就绪：{space}，设备 {device}，端口 {args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *exc) -> None:  # 安静
            return

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ready": True, **state, "dim": model.readout_width})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/encode":
                self._send(404, {"error": "unknown path"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                texts = json.loads(self.rfile.read(length) or b"{}").get("texts") or []
                if not isinstance(texts, list):
                    raise ValueError("texts 必须是字符串数组")
            except Exception as exc:
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if not texts:
                self._send(200, {"vectors": [], "dim": model.readout_width, "space_id": space})
                return
            with lock:  # 单模型的 trace 不共享，串行化最省心
                matrix = encode_texts(model, tokenizer, texts)
            self._send(
                200,
                {
                    "vectors": matrix.tolist(),
                    "dim": model.readout_width,
                    "space_id": space,
                    "normalized": True,
                },
            )

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
