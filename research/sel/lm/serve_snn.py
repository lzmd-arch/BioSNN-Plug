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
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
#: 束宽。`/complete` 每层只保留这么多节点；0 = 退回全词表的精确实现。
BEAM = 64

BPE = Path("D:/AxoData/sel/lm/corpus/skypile_tok/bpe")


def load_model(
    ckpt: Path,
    bpe: Path,
    device: torch.device,
    sensor_dim: int = 0,
    qwen: Path | None = None,
):
    """按检查点里记的配置重建模型。**形状对不上就报错，不静默降级。**"""
    from tokenizers import ByteLevelBPETokenizer

    from research.sel.lm.hierarchical import build_huffman_paths
    from research.sel.lm.model import TokenEProp
    from research.sel.lm.subset_tokenizer import SubsetTokenizer

    blob = torch.load(ckpt, map_location="cpu")
    cfg = blob["config"]
    if qwen is not None:
        # **子集必须与检查点配套**：同一个整数在两张子集表里指向不同的 token。
        from research.sel.lm.subset_tokenizer import verify_subset

        verify_subset(str(cfg.get("subset_sha256", "")), qwen)
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
        #: Semprea-4.5B 的传感器通道；0 = 关闭（逐位等同旧模型）
        sensor_dim=sensor_dim,
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

    if qwen is not None:
        # **子集模型必须配子集分词器**：模型吐出的是 0–31999 的本地下标，
        # 拿千问的原生分词器去解会得到完全不同的字，而且不会报错。
        tokenizer = SubsetTokenizer(qwen)
        if tokenizer.vocab_size != vocab:
            raise SystemExit(
                f"--qwen 的子集有 {tokenizer.vocab_size:,} 个 token，"
                f"而检查点是按 {vocab:,} 建的。两者必须一致，否则 id 整块错位。"
            )
    else:
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


@torch.no_grad()
def complete(model, tokenizer, context: str, k: int = 10, max_len: int = 256):
    """给一段上下文，返回**下一个 token 的 top-k**（id / 解码文本 / logprob）。

    **只做候选，不做生成**：这个模型记不住超过 ~64 步，把它包装成"会写句子"是在
    接口层面撒谎。调用方拿到的是一张分布表，怎么用由它决定。
    """
    ids = tokenizer.encode(context).ids[-max_len:]
    if not ids:
        return []
    trace = model.new_trace(1)
    device = model.device
    for step in range(len(ids) - 1):
        model.step(
            torch.tensor([ids[step]], device=device),
            torch.tensor([ids[step + 1]], device=device),
            trace,
            update=False,
            all_valid=True,
        )
    if len(ids) >= 2:
        probs = model._token_probabilities(model._features(trace))[0]
    else:
        probs = torch.full((model.n_vocab,), 1.0 / model.n_vocab, device=device)
    top = torch.topk(probs, min(k, probs.numel()))
    # **先取对数再 tolist()**：`top.values.tolist()` 给的是 Python float，
    # 再调 `.clamp` 会抛 AttributeError，而 http.server 会把连接直接掐掉——
    # 客户端看到的是"空正文"，不是 500。
    logprobs = torch.log(top.values.clamp(min=1e-12))
    return [
        {
            "id": int(index),
            "token": tokenizer.decode([int(index)]),
            "logprob": round(float(lp), 4),
        }
        for index, lp in zip(top.indices.tolist(), logprobs.tolist(), strict=True)
    ]


@torch.no_grad()
def score_sequence(model, tokenizer, context: str, continuation: str, max_len: int = 256) -> float:
    """给一段**续写**打分：teacher forcing 下的对数概率之和。

    这是 n-best 重排序要的接口——`/complete` 只给"下一个 token"，而重排序要评**整句**。
    小模型在序列级重排序上常常能提升大模型，因为它在评一致性，不是边际。
    """
    ids = tokenizer.encode(context).ids[-max_len:]
    cont = tokenizer.encode(continuation).ids
    if not ids or not cont:
        return float("-inf")
    trace = model.new_trace(1)
    device = model.device
    for step in range(len(ids) - 1):
        model.step(
            torch.tensor([ids[step]], device=device),
            torch.tensor([ids[step + 1]], device=device),
            trace,
            update=False,
            all_valid=True,
        )
    from research.sel.lm.hierarchical import path_logits

    total = 0.0
    last = ids[-1]
    for token in cont:
        center = torch.tensor([token], device=device)
        # **只算目标 token 那条路径**（O(depth)），不算整个词表（O(V·depth)）。
        # 全部保持 batch 维（1, K），不要 [0]——那样会和 bits/valid 的形状错开。
        # path_logits 返回 (logits, valid, safe) —— valid 就是 (B, K) 的路径掩码
        logits, valid, _safe = path_logits(
            model._features(trace), model.tree.weight, model.tree.paths, center, model.tree.bias
        )
        bits = model.tree.paths.bits.index_select(0, center)
        probs = torch.sigmoid(logits)
        probs = torch.where(bits > 0.5, probs, 1.0 - probs)
        total += float(torch.log(probs[valid].clamp(min=1e-12)).sum())
        model.step(
            torch.tensor([last], device=device),
            torch.tensor([token], device=device),
            trace,
            update=False,
            all_valid=True,
        )
        last = token
    return total


def build_child_map(paths):
    """深度 j 的节点 → 它在 j+1 的两个子节点（幂等，启动时建一次）。

    树是 ragged 的：某些 token 的路径在中途就结束了，那些节点本身就是叶子。
    """
    children: dict[int, dict[int, list[int]]] = {}
    depth = paths.nodes.shape[1]
    for j in range(depth - 1):
        layer: dict[int, list[int]] = {}
        parent = paths.nodes[:, j].tolist()
        child = paths.nodes[:, j + 1].tolist()
        for par, ch in zip(parent, child, strict=True):
            if par < 0 or ch < 0:
                continue
            slot = layer.setdefault(par, [])
            if ch not in slot:
                slot.append(ch)
        children[j] = layer
    return children


#: 树结构的派生表**必须缓存**——每次请求重建要 75 层 × 32,000 token，实测 2.8 秒。
_TREE_CACHE: dict[int, tuple] = {}


@torch.no_grad()
def complete_beam(model, tokenizer, context: str, k: int = 10, beam: int = 64, max_len: int = 256):
    """**树上束搜索**：每层只保留 top-B 节点，成本从 O(V·depth) 降到 O(B·depth)。

    这是 `/complete` 从 8.5–25 秒降到毫秒级的关键——它要的只是 top-k，
    而 `path_probabilities` 会把**整个词表**都算一遍。
    """
    paths = model.tree.paths
    weight = model.tree.weight
    bias = model.tree.bias
    ids = tokenizer.encode(context).ids[-max_len:]
    if not ids:
        return []
    trace = model.new_trace(1)
    device = model.device
    for step in range(len(ids) - 1):
        model.step(
            torch.tensor([ids[step]], device=device),
            torch.tensor([ids[step + 1]], device=device),
            trace,
            update=False,
            all_valid=True,
        )
    features = model._features(trace)  # (1, M)

    # 每个节点对应的 token（只有"路径在此结束"的节点才是叶子）
    depth = paths.nodes.shape[1]
    cached = _TREE_CACHE.get(id(paths))
    if cached is None:
        leaf_at: dict[tuple[int, int], int] = {}
        for token in range(len(paths)):
            length = int(paths.length[token])
            if 1 <= length <= depth:
                leaf_at[(int(paths.nodes[token, length - 1]), length - 1)] = token
        cached = (leaf_at, build_child_map(paths), depth)
        _TREE_CACHE[id(paths)] = cached
    leaf_at, children, depth = cached

    # 从根出发：根是第 0 层的那个节点
    roots = {int(n) for n in paths.nodes[:, 0].tolist() if n >= 0}
    beam_nodes = [(0.0, n, 0) for n in roots]  # (logp_sum, node, depth)
    results: list[tuple[float, int]] = []
    for level in range(depth):
        layer = children.get(level, {})
        # **每层一次批量矩阵乘**：逐个节点做 `@` 是上万次 Python 调用，实测占 2.9 s 的全部。
        pairs: list[tuple[float, int]] = []  # (父节点的 logp_sum, 子节点 id)
        for logp_sum, node, _ in beam_nodes:
            token = leaf_at.get((node, level))
            if token is not None:
                results.append((logp_sum, token))
            for child in layer.get(node, []):
                pairs.append((logp_sum, child))
        if not pairs:
            break
        child_ids = torch.tensor([c for _, c in pairs], device=device)
        logits = (features[0] @ weight.index_select(0, child_ids).T) + bias.index_select(
            0, child_ids
        )
        probs = torch.sigmoid(logits.clamp(-60.0, 60.0)).tolist()
        candidates: list[tuple[float, int, int]] = [
            (logp_sum + math.log(max(prob, 1e-12)), child, level + 1)
            for (logp_sum, child), prob in zip(pairs, probs, strict=True)
        ]
        if not candidates:
            break
        candidates.sort(key=lambda item: -item[0])
        beam_nodes = candidates[:beam]
    for logp_sum, node, depth_now in beam_nodes:
        token = leaf_at.get((node, depth_now - 1))
        if token is not None:
            results.append((logp_sum, token))

    results.sort(key=lambda item: -item[0])
    out = []
    for logp_sum, token in results[:k]:
        out.append(
            {
                "id": int(token),
                "token": tokenizer.decode([int(token)]),
                "logprob": round(logp_sum, 4),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="本地 SNN 编码服务")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bpe", type=Path, default=BPE)
    ap.add_argument(
        "--qwen",
        type=Path,
        default=None,
        help="千问 32k 子集目录（subset_32k.json/vocab.json/merges.txt）；给了就用子集分词器",
    )
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--port", type=int, default=18734)
    ap.add_argument(
        "--sensor-dim",
        type=int,
        default=16,
        help="传感器通道维度；投影头零初始化，所以没有向量时逐位无副作用",
    )
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, tokenizer, step = load_model(args.ckpt, args.bpe, device, args.sensor_dim, args.qwen)
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
            if self.path.rstrip("/") == "/complete":
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    context = str(body.get("context") or "")
                    k = int(body.get("k") or 10)
                    sensors = body.get("sensors")
                except Exception as exc:
                    self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                with lock:
                    # **传感器是这一整套里千问唯一拿不到的信息**：环境/多模态数据不出设备。
                    # 没有配对数据时传零向量，投影头也是零初始化 ⇒ 与不传逐位相同。
                    model.sensors = (
                        torch.tensor(sensors, dtype=torch.float32, device=model.device)
                        if sensors
                        else None
                    )
                    items = (
                        complete_beam(model, tokenizer, context, k)
                        if BEAM > 0
                        else complete(model, tokenizer, context, k)
                    )
                    model.sensors = None
                self._send(200, {"candidates": items, "space_id": space})
                return
            if self.path.rstrip("/") == "/score":
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    context = str(body.get("context") or "")
                    continuation = str(body.get("continuation") or "")
                except Exception as exc:
                    self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                with lock:
                    value = score_sequence(model, tokenizer, context, continuation)
                self._send(200, {"logprob": round(value, 4), "space_id": space})
                return
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
