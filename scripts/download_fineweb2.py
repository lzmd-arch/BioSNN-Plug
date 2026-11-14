"""下 FineWeb-2 的中文档（``cmn_Hani``），按片、可续传。

**为什么换掉 SkyPile**：SkyPile-150B 在 HuggingFace 上**没有 license 字段**，卡片要求
"Skywork Community License"，而那份协议授予的是 Skywork-13B 及其 *Model Derivatives*
（基于其权重/输出迁移的模型）。我们从零训练、不含它的权重，按定义不属于其衍生品，
协议里也没有授予"用这份数据训出来的模型"再分发的条款——**是一片空白，不是一条许可**。
详见 BioSNN-Plug 的 `research/sel/plan-semprea-versions.md` §九 T-5。

FineWeb-2 的许可是 **ODC-By 1.0**（只要求署名），HF 的 `license` 字段与标签都是
``odc-by``，`gated=False`。质量侧它走的是 FineWeb 那套过滤流水线。

用法::

    uv run --with huggingface_hub python scripts/download_fineweb2.py --shards 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO = "HuggingFaceFW/fineweb-2"
LANG = "cmn_Hani"
DEFAULT_DIR = Path("D:/AxoData/sel/lm/corpus/fineweb2_cmn")


def shard_list(lang: str) -> list[str]:
    from huggingface_hub import HfApi

    prefix = f"data/{lang}/train/"
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(prefix)]
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下 FineWeb-2 中文分片（ODC-By 1.0）")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--lang", default=LANG)
    ap.add_argument("--shards", type=int, default=12, help="下多少片（从 000_00000 起按序）")
    ap.add_argument("--retries", type=int, default=4, help="单片失败重试次数（传输层抖动）")
    args = ap.parse_args(argv)

    # **关掉 Xet**：它的 CAS 走 us.aws.cdn.hf.co，本机经代理时实测会挂在
    # "File reconstruction error"，而普通 HTTP 路径（老 SkyPile 就是这么下的）是通的。
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import hf_hub_download

    all_files = shard_list(args.lang)
    if not all_files:
        raise SystemExit(f"{REPO} 里没有 {args.lang} 的训练分片")
    queue = all_files[: args.shards]

    data_dir = args.dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in data_dir.glob("*.parquet")}
    have_bytes = sum(p.stat().st_size for p in data_dir.glob("*.parquet"))
    print(
        f"共 {len(all_files)} 片可选，本次取前 {len(queue)} 片；已有 {len(have)} 片 / "
        f"{have_bytes / 1e9:.2f} GB",
        flush=True,
    )

    started = time.time()
    total = have_bytes
    for index, name in enumerate(queue, 1):
        target = data_dir / Path(name).name
        if target.exists():
            continue
        t0 = time.time()
        got = None
        for attempt in range(1, args.retries + 1):
            try:
                got = hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(args.dir))
                break
            except Exception as exc:  # 传输层抖动（Xet CDN、代理、限流）都走这里
                if attempt == args.retries:
                    raise
                wait = min(15 * attempt, 60)
                print(
                    f"  {target.name} 第 {attempt} 次失败（{type(exc).__name__}: "
                    f"{str(exc)[:120]}），{wait} 秒后重试",
                    flush=True,
                )
                time.sleep(wait)
        if got is None:
            raise SystemExit(f"{target.name} 重试 {args.retries} 次仍失败")
        source = Path(got)
        if source != target:
            source.replace(target)
        size = target.stat().st_size
        total += size
        rate = size / max(time.time() - t0, 1e-9) / 1e6
        print(
            f"  [{index}/{len(queue)}] {target.name} {size / 1e9:.2f} GB"
            f"（{rate:.0f} MB/s，累计 {total / 1e9:.1f} GB）",
            flush=True,
        )
    print(f"完成：{total / 1e9:.2f} GB，{(time.time() - started) / 60:.1f} 分钟", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
