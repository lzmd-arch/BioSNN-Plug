"""1B 规模的训练入口：memmap 语料 + 层次 softmax 树 + 低秩输入 + 断点续训。

与 `run_r0` 的分工：`run_r0` 是**小语料的实验台**（一轮读进内存、按 batch 组张量、不留检查点）；
这里只做**能跑两天不崩**的那一件事。

用法::

    uv run python -m research.sel.lm.run_1b --smoke          # 冒烟：查显存与吞吐
    uv run python -m research.sel.lm.run_1b --steps 200000   # 正式
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from research.sel.lm.data_memmap import WindowSampler
from research.sel.lm.hierarchical import HuffmanPaths, build_huffman_paths
from research.sel.lm.model import TokenEProp
from research.sel.lm.subset_tokenizer import subset_fingerprint

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/lm/corpus/skypile_tok")
OUT = REPO / "sweep_results" / "lm_1b"


#: 参考工作点：`--lr-in` 是在这个 batch 与 signal_scale 下标定的。
REF_BATCH = 16


def effective_lr_in(lr_ref: float, batch: int, signal_scale: float) -> float:
    """把"参考工作点上报的学习率"折算到实际 batch。

    **为什么必须折算**：隐层更新是 `w_in += eps_inᵀ @ post`，**对 batch 求和**，不是平均。
    直接拿在小 batch 上调好的 lr 去跑大 batch，有效步长会乘上 batch 倍。
    实测：batch=1024、scale=3 时有效步长是参考点的 192 倍，一步就把隐层打崩
    （发放率 0.0087 → 0.0006、CE 8.2 → 51.4，模型完全塌掉）。

    这个函数是那条事故的唯一入口，所以它有守卫（`test_run_1b_infra.py`），
    且守卫量的是**行为**——同一批样本换 batch 之后单样本更新必须不变。
    """
    return lr_ref * REF_BATCH / max(batch * signal_scale, 1e-9)


def model_state(model: TokenEProp) -> dict:
    """读出的 4 GB 权重挂在 `model.tree`（不是 Tensor 属性），**必须单独存**。

    第一版漏了它 —— 检查点只有 54 MB，续训出来等于换了个模型。
    """
    blob = {k: v for k, v in vars(model).items() if isinstance(v, torch.Tensor)}
    if model.tree is not None:
        blob["__tree_weight"] = model.tree.weight
        blob["__tree_bias"] = model.tree.bias
    return blob


def load_state(model: TokenEProp, blob: dict) -> None:
    """**先丢掉旧的再装新的**：直接赋值会让新旧两份权重同时在显存里（各 2 GB），
    实测把占用顶到 7.68 GB / 8.15 GB，训练随即落进换页区。"""
    # **树权重先装**：它最大（2 GB），先腾出旧的那份，峰值才不会新旧两份同时在显存里
    ordered = sorted(blob.items(), key=lambda kv: 0 if kv[0] == "__tree_weight" else 1)
    for key, value in ordered:
        if key == "__tree_weight":
            model.tree.weight = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.tree.weight = value.to(model.device)
        elif key == "__tree_bias":
            model.tree.bias = value.to(model.device)
        else:
            setattr(model, key, value.to(model.device))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_SAVE_LOCK = threading.Lock()
_SAVE_THREAD: threading.Thread | None = None


def save_async(blob: dict, path: Path) -> None:
    """把检查点**先快照到 CPU，再后台写盘**。

    2 GB 的写盘会把 GPU 空转几秒，每 500 步一次（≈15 分钟）——实测这就是
    `nvidia-smi` 上周期性掉坑的主要来源。GPU→CPU 的 2 GB 拷贝约 0.2 s（PCIe），
    比写盘短一个量级，而且可以等上一次写完再开始，不排队。
    """
    global _SAVE_THREAD
    with _SAVE_LOCK:
        if _SAVE_THREAD is not None and _SAVE_THREAD.is_alive():
            print("  （上一次检查点还在写，跳过这次）", flush=True)
            return
        snapshot = {
            "step": blob["step"],
            "tensors": {k: v.detach().to("cpu", copy=True) for k, v in blob["tensors"].items()},
            "config": blob["config"],
        }

        def _write() -> None:
            # **先轮转再写**：写盘中途崩掉时，`<tag>.prev.pt` 还是上一次的完整文件。
            # 只有一个检查点文件的跑法，一次写坏就等于把 50 小时清零。
            previous = path.with_suffix(".prev.pt")
            if path.exists():
                path.replace(previous)
            torch.save(snapshot, path)

        _SAVE_THREAD = threading.Thread(target=_write, daemon=True)
        _SAVE_THREAD.start()


def wait_saves() -> None:
    global _SAVE_THREAD
    if _SAVE_THREAD is not None:
        _SAVE_THREAD.join()
        _SAVE_THREAD = None


def save_paths(paths: HuffmanPaths, path: Path, *, vocab: int, counts_key: str) -> None:
    """把**树结构本身**存下来。它是"续训还能不能接着算"的承重件。

    为什么必须单独存：内部节点 id 是**合并顺序**发的（见 ``build_huffman_paths``），
    而合并顺序由频次决定。换一批更大的语料重算频次，节点 id 会整体重排，
    于是检查点里那 4 GB 读出权重会**接到错误的节点上**——不报错，只是悄悄变差。
    """
    torch.save(
        {
            "nodes": paths.nodes.cpu(),
            "bits": paths.bits.cpu(),
            "length": paths.length.cpu(),
            "n_nodes": int(paths.n_nodes),
            "right_share": None if paths.right_share is None else paths.right_share.cpu(),
            "vocab": int(vocab),
            "counts_key": counts_key,
        },
        path,
    )


def load_paths(path: Path) -> HuffmanPaths:
    blob = torch.load(path, map_location="cpu")
    if "nodes" not in blob:
        raise SystemExit(
            f"{path} 不是树文件（没有 nodes 字段）。要的是训练时写出的 `<tag>_tree.pt`，"
            "不是检查点——检查点里只有读出权重，拿它反推不出节点 id。"
        )
    return HuffmanPaths(
        blob["nodes"], blob["bits"], blob["length"], int(blob["n_nodes"]), blob.get("right_share")
    )


def counts_key(counts: torch.Tensor) -> str:
    """语料指纹：用来判"这批频次和检查点是不是同一批"。"""
    return hashlib.sha256(counts.numpy().tobytes()).hexdigest()[:16]


def corpus_counts(paths: list[Path], vocab: int, cache: Path) -> tuple[torch.Tensor, bool]:
    """词频：算一次存下来（2B token 上重算要几分钟，续训时不该重复付）。

    **第二个返回值是"这次是不是从缓存读的"**，它决定续训能不能安全重建树：
    从缓存读 ⇒ 与上一次同频次 ⇒ 建出来的树**逐位相同**，续训安全；
    现算 ⇒ 语料可能变了 ⇒ 节点 id 可能重排，必须要求调用方显式放行。
    """
    if cache.exists():
        return torch.from_numpy(np.load(cache)), True
    counts = np.zeros(vocab, dtype=np.int64)
    for path in paths:
        flat = np.memmap(path, dtype=np.uint16, mode="r")
        for start in range(0, flat.size, 50_000_000):
            chunk = np.asarray(flat[start : start + 50_000_000])
            counts += np.bincount(chunk, minlength=vocab)[:vocab]
    np.save(cache, counts)
    return torch.from_numpy(counts), False


@torch.no_grad()
def evaluate(model, tokens: torch.Tensor, valid: torch.Tensor) -> dict:
    """在给定的一批序列上算 top-1 / top-5 / top-10 / 期望选择成本。"""
    trace = model.new_trace(tokens.shape[0])
    hits = {1: 0, 5: 0, 10: 0}
    cost = 0.0
    counted = 0
    for t in range(tokens.shape[1] - 1):
        target = tokens[:, t + 1]
        mask = valid[:, t + 1]
        model.step(tokens[:, t], target.masked_fill(~mask, -100), trace, update=False)
        if not bool(mask.any()):
            continue
        probs = model._token_probabilities(model._features(trace))[mask]
        truth = target[mask].unsqueeze(1)
        top10 = probs.topk(10, dim=-1)
        hit = probs.argmax(dim=-1, keepdim=True) == truth
        hits[1] += int(hit.sum())
        for k in (5, 10):
            hits[k] += int((top10.indices[:, :k] == truth).any(dim=1).sum())
        rank = top10.indices == truth
        in_top = rank.any(dim=1)
        pos = rank.to(torch.float32).argmax(dim=1) + 1
        cost += float(torch.where(in_top, pos, torch.full_like(pos, 11.0)).sum())
        counted += int(mask.sum())
    n = max(counted, 1)
    return {
        "top1": round(hits[1] / n, 4),
        "top5": round(hits[5] / n, 4),
        "top10": round(hits[10] / n, 4),
        "mean_rank_cost": round(cost / n, 4),
        "tokens": counted,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="1B 训练入口（memmap + 树 + 低秩输入）")
    ap.add_argument("--corpus", nargs="+", default=[str(DATA / "corpus.u16")])
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="训练所用 32k 子集的 subset_32k.json；指纹写进检查点，服务端据此校验",
    )
    ap.add_argument("--params", type=float, default=5.0e8, help="读出参数预算（决定 M）")
    ap.add_argument("--window", type=int, default=192)
    #: 实测：B=256、M=31,250 时峰值 6.11 GB，8 GB 卡上换页 ⇒ 一步 225 ms。
    #: B=128 把特征与临时张量全部减半，是第一个该降的旋钮。
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=2000)
    #: 单个窗口只有 256×192 = 49k token，top-1≈0.1 时标准误就有 0.13pp ——
    #: 看不出 2,000 步之间的差别。默认取 4 个窗口（≈20 万 token）。
    ap.add_argument("--eval-windows", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--embed-rank", type=int, default=256)
    ap.add_argument(
        "--emb-init",
        type=Path,
        default=None,
        help="词嵌入几何迁移的产物（transfer_embedding.py 的 .pt）",
    )
    ap.add_argument("--tree-signal-scale", type=float, default=3.0)
    ap.add_argument(
        "--tree-packed",
        action="store_true",
        help="ragged 布局：只走真实配对。矩形按 depth_max=50 补位，4x 浪费",
    )
    ap.add_argument(
        "--lr-in",
        type=float,
        default=1e-3,
        help="**参考工作点（batch=16、signal_scale=1）上的学习率**；实际值按 16/(batch×scale) 自动折算",
    )
    ap.add_argument("--lr-out", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="run1b")
    ap.add_argument("--resume", default="", help="检查点路径；给了就从那里续")
    ap.add_argument(
        "--tree-from",
        type=Path,
        default=None,
        help="冻结的树文件（save 出来的 <tag>_tree.pt）；换语料续训时必须给，否则节点 id 会错位",
    )
    ap.add_argument(
        "--allow-tree-rebuild",
        action="store_true",
        help="找不到树文件时仍按当前语料重建（**只在语料没变时才对**，等于旧行为）",
    )
    ap.add_argument("--smoke", action="store_true", help="冒烟：50 步、查显存与吞吐")
    args = ap.parse_args(argv)

    if args.smoke:
        args.steps = min(args.steps, 50)
        args.eval_every = 0

    subset_hash = ""
    if args.subset is not None:
        if not args.subset.exists():
            raise SystemExit(f"--subset 指向的文件不存在：{args.subset}")
        subset_hash = subset_fingerprint(args.subset)
        size = len(json.loads(args.subset.read_text(encoding="utf-8"))["ids"])
        if size != args.vocab:
            raise SystemExit(
                f"--vocab {args.vocab} 与子集长度 {size} 不一致。"
                "模型会按 --vocab 建嵌入表，而语料里的 id 是子集下标——对不上就整块错位。"
            )
        print(f"子集指纹 {subset_hash}（{args.subset.name}，{size:,} 个 token）", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [Path(p) for p in args.corpus]
    args.out.mkdir(parents=True, exist_ok=True)

    tree_file = args.out / f"{args.tag}_tree.pt"
    paths_t: HuffmanPaths | None = None
    counts: torch.Tensor | None = None
    if args.tree_from is not None:
        paths_t = load_paths(args.tree_from)
        print(f"树从 {args.tree_from} 载入（冻结，{len(paths_t):,} token）", flush=True)
    elif args.resume and tree_file.exists():
        paths_t = load_paths(tree_file)
        print(f"树从 {tree_file} 载入（续训，冻结）", flush=True)
    if paths_t is None:
        counts, from_cache = corpus_counts(paths, args.vocab, args.out / f"counts_{args.vocab}.npy")
        paths_t = build_huffman_paths(counts.to(torch.float32))
        # **现算频次**才危险：语料换过 ⇒ 合并顺序变 ⇒ 内部节点 id 整体重排 ⇒ 检查点里那
        # 几 GB 的读出权重会接到错误的节点上，而且不报错、只会悄悄变差。
        # 从缓存读则与上一次同频次，建出来的树逐位相同，续训安全。
        if args.resume and not from_cache and not args.allow_tree_rebuild:
            raise SystemExit(
                f"续训要现算词频（{args.out / f'counts_{args.vocab}.npy'} 不存在），"
                "换过语料的话内部节点 id 会整体重排，检查点里的读出权重会接到错误的节点上，"
                "而且**不会报错、只会悄悄变差**。"
                "要么用 --tree-from 指一份树文件，要么确认语料没变再加 --allow-tree-rebuild。"
            )
        save_paths(paths_t, tree_file, vocab=args.vocab, counts_key=counts_key(counts))
        print(f"树写入 {tree_file}（之后的续训以它为准）", flush=True)
    paths_t = paths_t.to(device)
    width = int(args.params / args.vocab)
    n_rec = max(width // 2, 64)
    weighted = (
        f"{float((counts / counts.sum() * paths_t.length.float().cpu()).sum()):.2f}"
        if counts is not None
        else "（冻结树，未重算）"
    )
    print(
        f"V={args.vocab:,} M={width:,} N={n_rec:,} 树：加权深度 {weighted} / 最深 {paths_t.depth}",
        flush=True,
    )

    model = TokenEProp(
        args.vocab,
        n_rec,
        recurrence="none",
        readout_taus=(2.0, 128.0),
        tree=paths_t,
        tree_signal_scale=args.tree_signal_scale,
        tree_packed=bool(args.tree_packed),
        # 续训时树权重马上被覆盖，别先建 2 GB 随机权重
        tree_init_weight=not bool(args.resume),
        embed_rank=args.embed_rank,
        embed_init=(
            torch.load(args.emb_init, map_location="cpu")["emb"] if args.emb_init else None
        ),
        # **隐层更新是 `eps_inᵀ @ post` 对 batch 求和** ⇒ 有效步长 ∝ lr_in × batch × signal_scale。
        # P1 的 1e-3 是在 batch=16、scale=1 上标的；直接拿到 batch=1024 上会把隐层打崩
        # （实测：发放率 0.0087 → 0.0006、CE 8.2 → 51.4，模型完全塌掉）。
        lr_in=effective_lr_in(args.lr_in, args.batch, args.tree_signal_scale),
        lr_out=args.lr_out,
        generator=torch.Generator().manual_seed(0),
        device=device,
    )
    # 读出权重挂在 `model.tree`（TreeReadout）里，不是 Tensor 属性，要单独数
    # 权重构造会在缓存分配器里留下 4 GB 的碎片，先还给它
    del counts
    torch.cuda.empty_cache() if device.type == "cuda" else None
    n_params = sum(int(v.numel()) for v in vars(model).values() if isinstance(v, torch.Tensor))
    if model.tree is not None:
        n_params += int(model.tree.weight.numel()) + int(model.tree.bias.numel())
    print(f"参数 {n_params / 1e9:.3f}B（读出 {width * args.vocab / 1e9:.3f}B）", flush=True)

    start_step = 0
    if args.resume:
        # 先落到 CPU 再逐个搬到显存：`map_location=device` 会让新旧两份 2 GB 权重
        # 同时在场，实测把占用顶到 7.68 GB / 8.15 GB。
        source = Path(args.resume)
        if not source.exists():
            fallback = source.with_suffix(".prev.pt")
            print(f"{source} 不存在，回退到 {fallback}", flush=True)
            source = fallback
        blob = torch.load(source, map_location="cpu")
        load_state(model, blob["tensors"])
        start_step = int(blob["step"])
        del blob  # 2 GB 的 CPU 副本，读完立刻释放
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"从 {source} 续训，第 {start_step} 步", flush=True)

    sampler = WindowSampler(
        paths,
        batch=args.batch,
        window=args.window,
        generator=torch.Generator().manual_seed(1),
        device=device,
    )
    probes = [sampler.next() for _ in range(max(args.eval_windows, 1))]
    torch.cuda.reset_peak_memory_stats()
    running = 0.0
    started = time.time()
    tokens_seen = 0
    for step in range(start_step + 1, args.steps + 1):
        tokens, valid = sampler.next()
        trace = model.new_trace(tokens.shape[0])
        t_step = time.time()
        result = model.run(tokens, trace, valid=valid)
        if args.smoke:
            torch.cuda.synchronize()
            print(
                f"  step {step}: {time.time() - t_step:.2f} s/update"
                f"（{tokens.shape[1] * tokens.shape[0] / (time.time() - t_step) / 1e3:.1f}k tok/s）",
                flush=True,
            )
        running += result["ce"]
        tokens_seen += int(tokens.shape[1]) * int(tokens.shape[0])
        if step % 50 == 0:
            rate = tokens_seen / (time.time() - started)
            peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(
                f"  {step} 步：训练 CE {running / 50:.3f}，{rate / 1e3:.1f}k tok/s，"
                f"峰值显存 {peak:.2f} GB",
                flush=True,
            )
            running = 0.0
        if args.eval_every and step % args.eval_every == 0:
            tally: dict[str, float] = {}
            total = 0
            for probe_tokens, probe_valid in probes:
                part = evaluate(model, probe_tokens, probe_valid)
                weight = part["tokens"]
                total += weight
                for key, value in part.items():
                    if key != "tokens":
                        tally[key] = tally.get(key, 0.0) + float(value) * weight
            stats = {k: round(v / max(total, 1), 4) for k, v in tally.items()}
            stats["tokens"] = total
            print(f"  [评测 {step}] {json.dumps(stats, ensure_ascii=False)}", flush=True)
            # 评测会为全词表 (B, V) 分配大张量，分配器会把它缓存住不还。
            # 实测 `max_memory_allocated` 从 2.33 GB 涨到 4.47 GB，系统占用 6.5 GB——
            # 训练从此只剩 1.6 GB 余量，和 1B 掉进换页是同一类风险。评测完就还给驱动。
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if args.ckpt_every and step % args.ckpt_every == 0:
            blob = {
                "step": step,
                "tensors": model_state(model),
                "config": {
                    **{k: str(v) for k, v in vars(args).items()},
                    # **id 空间的指纹**：换一份子集，同一个整数就指向另一个 token，
                    # 读出权重整块错位且不报错。存下来，服务端才校验得了。
                    "subset_sha256": subset_hash,
                },
            }
            save_async(blob, args.out / f"{args.tag}_ckpt.pt")
    wait_saves()
    elapsed = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    print(f"峰值显存 {peak_gb:.2f} GB", flush=True)
    print(
        f"完成 {args.steps - start_step} 步，{elapsed / 60:.1f} 分，"
        f"{tokens_seen / elapsed / 1e3:.1f}k tok/s，"
        f"两天可训 {tokens_seen / elapsed * 172800 / 1e9:.2f}B token",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
