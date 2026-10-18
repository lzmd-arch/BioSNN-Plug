"""在真实聊天记忆上评估召回：词法 / 语义 / 图通道。

**隐私边界：本脚本只输出聚合指标，不输出任何正文、查询文本或记忆内容。**
它用 SQLite 在线备份 API 把库复制到临时目录（原库只读），用完删除；查询向量由
本机 Ollama 的 ``nomic-embed-text`` 现算，不出设备。

数据：Axo 默认数据根下的 ``memory/mem.db`` 与 ``sessions/*.jsonl``。每条
``source_ref=turn://<session>/<seq>`` 的记忆与第 ``seq`` 条用户消息一一对应
（已用时间差核对：61/61 对齐）。

用法::

    uv run python scripts/axo_real_recall_probe.py
    uv run python scripts/axo_real_recall_probe.py --window 2 --graph-weight 0,0.1,0.25
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DEFAULT_DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", "")) / "axo"
OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class AdjacencyGraph:
    """``recall`` 的图通道只需要 ``node_ids``、``len``、``diffuse``。"""

    node_ids: tuple[str, ...]
    rows: np.ndarray
    targets: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.node_ids)

    def diffuse(self, base: np.ndarray, *, decay: float = 0.5) -> np.ndarray:
        scores = np.asarray(base, dtype=np.float64)
        if scores.shape != (len(self.node_ids),):
            raise ValueError(f"种子分形状应为 ({len(self.node_ids)},)，收到 {scores.shape}。")
        spread = np.zeros(len(self.node_ids), dtype=np.float64)
        np.maximum.at(spread, self.targets, scores[self.rows] * self.weights * decay)
        return spread


def _import_axo(src: Path):
    sys.path.insert(0, str(src))
    from memcore.recall import RecallConfig, recall
    from memcore.store import MemoryStore

    return RecallConfig, recall, MemoryStore


def _copy_db(src: Path) -> Path:
    """SQLite 在线备份，包含 WAL 里已提交的内容。"""
    directory = Path(tempfile.mkdtemp(prefix="axo_real_probe_"))
    destination = directory / "mem.db"
    source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(destination)
    source.backup(target)
    target.close()
    source.close()
    return directory


def _embed(texts: list[str], *, host: str, model: str, dim: int) -> np.ndarray:
    """用 urllib 直接调 Ollama，**显式禁用代理**（loopback 被系统代理接过一次）。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    batches = []
    for start in range(0, len(texts), 32):
        window = texts[start : start + 32]
        body = json.dumps({"model": model, "input": window}).encode("utf-8")
        request = urllib.request.Request(
            f"{host}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=120) as response:
            payload = json.load(response)
        matrix = np.asarray(payload["embeddings"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(window), dim):
            raise SystemExit(f"嵌入形状不对：{matrix.shape}，期望 {(len(window), dim)}")
        batches.append(matrix)
    matrix = np.concatenate(batches, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _load_queries(data_root: Path, items):
    """从会话 JSONL 取用户消息；用 turn_seq 对齐记忆。只返回查询文本与目标 id。"""
    sessions_dir = data_root / "sessions"
    session_files = sorted(sessions_dir.glob("*.jsonl"))
    if not session_files:
        raise SystemExit(f"{sessions_dir} 下没有会话文件。")

    queries = []
    for item in items:
        if not item.source_ref.startswith("turn://"):
            continue
        sequence = item.meta.get("turn_seq")
        if sequence is None:
            continue
        session_id = item.session_id
        path = sessions_dir / f"{session_id}.jsonl"
        if not path.is_file():
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        users = [row for row in rows if row.get("role") == "user"]
        index = int(sequence) - 1
        if not 0 <= index < len(users):
            continue
        queries.append(
            {
                "turn": int(sequence),
                "text": users[index]["content"],
                "target": item.memory_id,
                "session_id": session_id,
            }
        )
    return queries


def _semantic_graph(ids, vectors, *, top_m: int, min_sim: float) -> AdjacencyGraph:
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -np.inf)
    rows, targets, weights = [], [], []
    for row in range(len(ids)):
        order = np.argpartition(-sim[row], min(top_m, len(ids) - 1))[:top_m]
        for column in order:
            weight = float(sim[row, column])
            if weight >= min_sim:
                rows.append(row)
                targets.append(int(column))
                weights.append(weight)
    return AdjacencyGraph(
        node_ids=tuple(ids),
        rows=np.asarray(rows, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
    )


def _temporal_graph(items, *, window: int) -> AdjacencyGraph:
    ids = tuple(item.memory_id for item in items)
    index = {memory_id: row for row, memory_id in enumerate(ids)}
    turns = {}
    for item in items:
        if item.source_ref.startswith("turn://") and item.meta.get("turn_seq") is not None:
            # **会话内**轮次：不同会话的 turn_seq 会从 1 重新开始，不能跨会话连边。
            turns[(item.session_id, int(item.meta["turn_seq"]))] = item.memory_id
    rows, targets, weights = [], [], []
    for (session, source_turn), source in turns.items():
        for (other_session, target_turn), target in turns.items():
            if session != other_session or source == target:
                continue
            delta = abs(source_turn - target_turn)
            if 1 <= delta <= window:
                rows.append(index[source])
                targets.append(index[target])
                weights.append(1.0 / (1.0 + delta))
    return AdjacencyGraph(
        node_ids=ids,
        rows=np.asarray(rows, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
    )


def _merge_graphs(left: AdjacencyGraph, right: AdjacencyGraph) -> AdjacencyGraph:
    """按 id 合并两张图，同一个有向边取较大的权重。"""
    ids = tuple(dict.fromkeys([*left.node_ids, *right.node_ids]))
    index = {memory_id: row for row, memory_id in enumerate(ids)}
    weights: dict[tuple[int, int], float] = {}
    for graph in (left, right):
        for row, target, weight in zip(graph.rows, graph.targets, graph.weights, strict=True):
            key = (index[graph.node_ids[int(row)]], index[graph.node_ids[int(target)]])
            weights[key] = max(weights.get(key, 0.0), float(weight))
    rows = [key[0] for key in weights]
    targets = [key[1] for key in weights]
    values = list(weights.values())
    return AdjacencyGraph(
        node_ids=ids,
        rows=np.asarray(rows, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        weights=np.asarray(values, dtype=np.float64),
    )


def _rank(injected: tuple[str, ...], expected: set[str]) -> int:
    for position, memory_id in enumerate(injected, start=1):
        if memory_id in expected:
            return position
    return 0


def _metrics(ranks: list[int]) -> dict[str, float]:
    total = len(ranks)
    histogram = {str(rank): ranks.count(rank) for rank in range(0, 9)}
    return {
        "n": total,
        "hit@1": sum(1 for rank in ranks if rank == 1) / total if total else float("nan"),
        "hit@3": sum(1 for rank in ranks if 1 <= rank <= 3) / total if total else float("nan"),
        "hit@5": sum(1 for rank in ranks if 1 <= rank <= 5) / total if total else float("nan"),
        "hit@8": sum(1 for rank in ranks if 1 <= rank <= 8) / total if total else float("nan"),
        "mrr": sum(1.0 / rank for rank in ranks if rank) / total if total else float("nan"),
        "rank_histogram": histogram,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument("--model", default="nomic-embed-text:latest")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--window", type=int, default=2, help="关联目标的相邻轮数")
    parser.add_argument(
        "--task",
        default="association",
        choices=["primary", "association", "memory_to_adjacent", "semantic_neighbors"],
        help="primary = 用户消息找回本轮记忆；association = 排除本轮、找后 window 轮；"
        "memory_to_adjacent = 用本轮记忆正文找后 window 轮；"
        "semantic_neighbors = 以嵌入余弦的 top-k 为期望集（独立于时序图）",
    )
    parser.add_argument("--neighbors", type=int, default=5, help="semantic_neighbors 的 k")
    parser.add_argument(
        "--semantic-exclude-window",
        type=int,
        default=0,
        help="semantic_neighbors 里排除轮次差 ≤ 该值的记忆（0 = 不排除）",
    )
    parser.add_argument(
        "--channels",
        default="lexical,semantic,semantic+sem,lexical+temp,semantic+combined",
        help="逗号分隔的通道名",
    )
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--min-sim", type=float, default=0.20)
    parser.add_argument("--graph-weight", default="0,0.1,0.25")
    parser.add_argument("--min-score", default="0.10")
    parser.add_argument("--decay", default="0.25")
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_real_recall/scan.json"))
    args = parser.parse_args(argv)

    recall_config_cls, recall_fn, memory_store_cls = _import_axo(args.axo_src)
    source_db = args.data_root / "memory" / "mem.db"
    if not source_db.is_file():
        raise SystemExit(f"找不到真实库 {source_db}。")
    temporary = _copy_db(source_db)
    try:
        store = memory_store_cls(temporary / "mem.db")
        items = list(store.iter_items())
        queries = _load_queries(args.data_root, items)
        print(f"真实记忆 {len(items)} 条；与用户轮次对齐 {len(queries)} 条查询。")

        space = store.registry.all()[0]
        space_id = space.space_id
        dim = space.dim
        model = space_id.split("/", 1)[1].rsplit("@", 1)[0]
        print(f"嵌入空间 {space_id}（{dim} 维，模型 {model}）")

        vectors = _embed(
            [query["text"] for query in queries],
            host=args.host,
            model=model,
            dim=dim,
        )

        item_by_id = {item.memory_id: item for item in items}
        embedding_ids, embedding_matrix = store.embedding_matrix(space_id)
        vector_by_id = dict(zip(embedding_ids, embedding_matrix, strict=True))
        semantic_graph = _semantic_graph(
            embedding_ids, embedding_matrix, top_m=args.top_m, min_sim=args.min_sim
        )
        temporal_graph = _temporal_graph(items, window=args.window)
        combined_graph = _merge_graphs(semantic_graph, temporal_graph)

        weights = [float(value) for value in args.graph_weight.split(",") if value.strip()]
        min_scores = [float(value) for value in args.min_score.split(",") if value.strip()]
        decays = [float(value) for value in args.decay.split(",") if value.strip()]

        # 关联目标：本轮记忆之后 window 轮里存在的记忆。
        turn_to_id = {
            (item.session_id, int(item.meta["turn_seq"])): item.memory_id
            for item in items
            if item.source_ref.startswith("turn://") and item.meta.get("turn_seq") is not None
        }
        now = max(item.created_at for item in items) + 1.0

        task_queries = []
        for index, query in enumerate(queries):
            user_vector = vectors[index]
            target = query["target"]
            if args.task == "primary":
                expected = {target}
                exclude: set[str] = set()
                text = query["text"]
                memory_vector = None
            else:
                expected = set()
                for offset in range(1, args.window + 1):
                    adjacent = turn_to_id.get((query["session_id"], query["turn"] + offset))
                    if adjacent:
                        expected.add(adjacent)
                if not expected:
                    continue
                exclude = {target}
                if args.task == "memory_to_adjacent":
                    text = item_by_id[target].content
                    memory_vector = vector_by_id.get(target)
                elif args.task == "semantic_neighbors":
                    memory_vector = vector_by_id.get(target)
                    if memory_vector is None:
                        continue
                    similarities = embedding_matrix @ memory_vector
                    order = np.argsort(-similarities)
                    expected = set()
                    for position in order:
                        candidate = embedding_ids[int(position)]
                        if candidate == target:
                            continue
                        if args.semantic_exclude_window > 0:
                            candidate_item = item_by_id[candidate]
                            candidate_turn = int(candidate_item.meta.get("turn_seq", -(10**9)))
                            if (
                                candidate_item.session_id == query["session_id"]
                                and abs(candidate_turn - query["turn"])
                                <= args.semantic_exclude_window
                            ):
                                continue
                        expected.add(candidate)
                        if len(expected) >= args.neighbors:
                            break
                    if not expected:
                        continue
                    exclude = {target}
                    text = query["text"]
                else:
                    text = query["text"]
                    memory_vector = None
            task_queries.append(
                {
                    "text": text,
                    "user_vector": user_vector,
                    "memory_vector": memory_vector,
                    "expected": expected,
                    "exclude": exclude,
                }
            )
        print(f"任务 {args.task}：{len(task_queries)} 条查询（原 {len(queries)} 条）。")

        rows = []
        all_channels = [
            ("lexical", None, False),
            ("semantic", None, True),
            ("semantic+sem", semantic_graph, True),
            ("lexical+temp", temporal_graph, False),
            ("semantic+combined", combined_graph, True),
        ]
        selected_channels = {name.strip() for name in args.channels.split(",") if name.strip()}
        channels = [channel for channel in all_channels if channel[0] in selected_channels]
        first_channels_reported = False
        for channel, graph, use_semantic in channels:
            for graph_weight in weights:
                if graph is None and graph_weight != 0.0:
                    continue
                for min_score in min_scores:
                    for decay in decays:
                        config = recall_config_cls(
                            graph_weight=graph_weight,
                            graph_decay=decay,
                            min_score=min_score,
                        )
                        ranks = []
                        channel_sets = set()
                        for query in task_queries:
                            vector = None
                            if use_semantic:
                                vector = (
                                    query["memory_vector"]
                                    if args.task == "memory_to_adjacent"
                                    else query["user_vector"]
                                )
                            ledger = recall_fn(
                                store,
                                query_text=query["text"],
                                budget_tokens=10**9,
                                query_vector=vector,
                                space_id=space_id if vector is not None else "",
                                config=config,
                                exclude_ids=tuple(query["exclude"]),
                                now=now,
                                settle=False,
                                graph=graph if graph_weight > 0.0 else None,
                            )
                            ranks.append(_rank(ledger.injected_ids, query["expected"]))
                            channel_sets.add(tuple(getattr(ledger, "channels", ())))
                        if not first_channels_reported:
                            print(f"实际启用的通道：{sorted(channel_sets)}")
                            first_channels_reported = True
                        row = {
                            "task": args.task,
                            "channel": channel,
                            "graph_weight": graph_weight,
                            "min_score": min_score,
                            "decay": decay,
                            "metrics": _metrics(ranks),
                        }
                        rows.append(row)
                        print(
                            f"{channel:<18} w={graph_weight:<5} ms={min_score:<5} d={decay:<5} "
                            f"hit@1 {row['metrics']['hit@1']:.4f}  "
                            f"hit@3 {row['metrics']['hit@3']:.4f}  "
                            f"MRR {row['metrics']['mrr']:.4f}  "
                            f"n={row['metrics']['n']}"
                        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "task": args.task,
                    "n_memories": len(items),
                    "n_queries": len(task_queries),
                    "records": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n[落盘] {args.out}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
