"""1B 规模的训练入口：memmap 语料 + 层次 softmax 树 + 低秩输入 + 断点续训。

与 `run_r0` 的分工：`run_r0` 是**小语料的实验台**（一轮读进内存、按 batch 组张量、不留检查点）；
这里只做**能跑两天不崩**的那一件事。

用法::

    uv run python -m research.sel.lm.run_1b --smoke          # 冒烟：查显存与吞吐
    uv run python -m research.sel.lm.run_1b --steps 200000   # 正式
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from research.sel.lm.data_memmap import WindowSampler
from research.sel.lm.hierarchical import build_huffman_paths
from research.sel.lm.model import TokenEProp

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/lm/corpus/skypile_tok")
OUT = REPO / "sweep_results" / "lm_1b"


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


def corpus_counts(paths: list[Path], vocab: int, cache: Path) -> torch.Tensor:
    """词频：算一次存下来（2B token 上重算要几分钟，续训时不该重复付）。"""
    if cache.exists():
        return torch.from_numpy(np.load(cache))
    counts = np.zeros(vocab, dtype=np.int64)
    for path in paths:
        flat = np.memmap(path, dtype=np.uint16, mode="r")
        for start in range(0, flat.size, 50_000_000):
            chunk = np.asarray(flat[start : start + 50_000_000])
            counts += np.bincount(chunk, minlength=vocab)[:vocab]
    np.save(cache, counts)
    return torch.from_numpy(counts)


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
    ap.add_argument("--smoke", action="store_true", help="冒烟：50 步、查显存与吞吐")
    args = ap.parse_args(argv)

    if args.smoke:
        args.steps = min(args.steps, 50)
        args.eval_every = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [Path(p) for p in args.corpus]
    args.out.mkdir(parents=True, exist_ok=True)

    counts = corpus_counts(paths, args.vocab, args.out / f"counts_{args.vocab}.npy")
    paths_t = build_huffman_paths(counts.to(torch.float32)).to(device)
    width = int(args.params / args.vocab)
    n_rec = max(width // 2, 64)
    print(
        f"V={args.vocab:,} M={width:,} N={n_rec:,} 树：加权深度 "
        f"{float((counts / counts.sum() * paths_t.length.float().cpu()).sum()):.2f}"
        f" / 最深 {paths_t.depth}",
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
        lr_in=args.lr_in * 16.0 / max(args.batch * args.tree_signal_scale, 1e-9),
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
                "config": {k: str(v) for k, v in vars(args).items()},
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
