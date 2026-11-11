"""按片下载 SkyPile-150B 的 **zh_head**（最高质量档），够用即停。

为什么按序而不是随机挑：同一时段（2020-40 → 2020-45 → 2020-50）的分布更一致，
tokenizer 与树的频次统计不会被时段差异污染。

默认落在 ``D:/AxoData/sel/lm/corpus/skypile/data``（与已有的 0000–0007 同一目录），
``build_tokenized_corpus.py`` 自动会捡到。**下载可断点续传**（hf_hub_download 自带）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO = "Skywork/SkyPile-150B"
DEFAULT_DIR = Path("D:/AxoData/sel/lm/corpus/skypile")

#: 按顺序取，前面的用完再往后
GROUPS = ["2020-40", "2020-45", "2020-50", "2021-04", "2021-10"]


def plan(groups: list[str], counts: dict[str, int], have: set[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for index in range(counts[group]):
            name = f"data/{group}_zh_head_{index:04d}.jsonl"
            if name not in have:
                out.append(name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下 SkyPile zh_head 分片，够目标字节数就停")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--target-gb", type=float, default=40.0)
    ap.add_argument("--groups", default=",".join(GROUPS))
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    data_dir = args.dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    have = {f"data/{p.name}" for p in data_dir.glob("*.jsonl")}
    have_bytes = sum(p.stat().st_size for p in data_dir.glob("*.jsonl"))
    print(f"已有 {len(have)} 片 / {have_bytes / 1e9:.1f} GB（含此前下的）", flush=True)

    api = HfApi()
    info = api.dataset_info(REPO)
    all_names = [s.rfilename for s in info.siblings if s.rfilename.endswith(".jsonl")]
    counts = {g: sum(1 for n in all_names if f"/{g}_zh_head_" in n) for g in args.groups.split(",")}
    queue = plan(args.groups.split(","), counts, have)
    print(
        f"可下 {sum(counts.values())} 片，待下队列 {len(queue)} 片；目标再下 {args.target_gb} GB",
        flush=True,
    )

    start_bytes = have_bytes
    target = args.target_gb * 1e9
    t0 = time.time()
    for name in queue:
        if have_bytes - start_bytes >= target:
            break
        t_file = time.time()
        try:
            path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(args.dir))
        except Exception as exc:  # 网络抖动：跳过，下一轮再补
            print(f"  失败 {name}: {type(exc).__name__} {exc}", flush=True)
            time.sleep(10)
            continue
        size = Path(path).stat().st_size
        have_bytes = sum(p.stat().st_size for p in data_dir.glob("*.jsonl"))
        dt = time.time() - t_file
        print(
            f"  {name}  +{size / 1e9:.2f} GB（{dt / 60:.1f} 分）"
            f" ⇒ 新增 {(have_bytes - start_bytes) / 1e9:.1f} / {args.target_gb} GB",
            flush=True,
        )
    print(
        f"完成：{len(list(data_dir.glob('*.jsonl')))} 片 / {have_bytes / 1e9:.1f} GB，"
        f"用时 {(time.time() - t0) / 60:.1f} 分",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
