#!/usr/bin/env python
"""数据下载与预处理（计划书 §12.1、§12.4）。

计划书 §12.1 规定**本仓库不提供数据镜像**，只提供下载与预处理脚本；§12.4 要求
每个阶段的精确环境可复现。这个脚本就是那条规定的落点。

用法::

    uv run python scripts/download_data.py                 # 下载全部
    uv run python scripts/download_data.py mnist           # 只下 MNIST
    uv run python scripts/download_data.py --verify-only   # 只校验，不下载
    uv run python scripts/download_data.py --list

数据落在 ``data/``（已进 .gitignore），每个数据集一个子目录。

## 为什么校验的是 MD5

**这不是安全机制，是完整性校验**——防的是下载截断与镜像内容漂移，不是防篡改。

选 MD5 是因为对 MNIST 而言，MD5 是**唯一有独立第三方公布**的校验值：
``torchvision.datasets.MNIST`` 把四个文件的 MD5 硬编码在自己的源码里
（``mirrors``/``resources``），本脚本钉的就是这四个值。原始发布页
（yann.lecun.com）当前不可达，所以「照抄 torchvision 钉过的值」是可交叉核对的做法；
自己算一个 SHA-256 钉上去则只是自己给自己背书。

下载后会把校验通过的 SHA-256 一并打印出来，供复现记录留档。

## 为什么自己解析 IDX 而不直接用 torchvision

``scripts/`` 下的脚本属于仓库工具链，与骨架库一样只依赖 numpy（见 ADR-0002 的
分层取向）。torchvision 在 ``research`` 依赖组里，装它才能下数据会让"下数据"这件
事莫名其妙地依赖整个 torch 生态。IDX 格式本身很简单，二十行就能读完。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

#: 下载超时（秒）。数据集文件不大，但镜像偶尔很慢。
TIMEOUT = 120


@dataclass(frozen=True)
class DataFile:
    """一个待下载的文件及其完整性校验值。"""

    filename: str
    md5: str

    @property
    def kind(self) -> str:
        """``images`` / ``labels`` / ``other``——决定落到哪个子目录。"""
        if "images" in self.filename:
            return "images"
        if "labels" in self.filename:
            return "labels"
        return "other"


@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的下载规格。"""

    name: str
    #: 依次尝试的镜像前缀。第一个可用即可，多的只是冗余。
    mirrors: tuple[str, ...]
    files: tuple[DataFile, ...]
    #: 数据源的许可说明。§12.1：遵循各数据源许可，本仓库不做镜像。
    license_note: str
    #: 预处理产物：``输入文件名 -> 输出 .npy 文件名``。
    parse: dict[str, str]


#: MNIST。四个文件的 MD5 取自 torchvision 源码里硬编码的值（可交叉核对）。
#: 按 §12.1，数据的再分发要另看数据源自身的许可，所以这里只下载、不入库。
MNIST = DatasetSpec(
    name="mnist",
    mirrors=(
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://raw.githubusercontent.com/fgnt/mnist/master/",
    ),
    files=(
        DataFile("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        DataFile("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        DataFile("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        DataFile("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ),
    license_note="Yann LeCun 与 Corinna Cortes 发布，研究用途；本仓库不镜像、不再分发。",
    parse={
        "train-images-idx3-ubyte.gz": "train_images.npy",
        "train-labels-idx1-ubyte.gz": "train_labels.npy",
        "t10k-images-idx3-ubyte.gz": "test_images.npy",
        "t10k-labels-idx1-ubyte.gz": "test_labels.npy",
    },
)

#: 数据集注册表。**新增数据集时在这里加一项，其余代码不用动。**
#:
#: 尚未加入的：
#:
#: * **SHD**（Spiking Heidelberg Digits）——e-prop 认知层的引文对标数据集
#:   （Bellec et al. 2020 与 Pes et al. 2025 都用它）。它随第一阶段第三条线
#:   （``research/eprop/``）一起接入，届时在此登记并钉校验值。它是 HDF5 格式，
#:   解析要用 h5py（已随 spikingjelly 进入 research 依赖组）。
DATASETS: dict[str, DatasetSpec] = {
    MNIST.name: MNIST,
}


# ────────────────────────────── IDX 解析 ──────────────────────────────


def read_idx(raw: bytes) -> np.ndarray:
    """解析 IDX 格式（MNIST 用的那个）。

    头部是定长的：magic 4 字节 + 维度数 1 字节 + 每维 4 字节，全部大端。
    magic 的低字节给出元素类型码，高字节是维度数——但真正可靠的是第 4 个
    字节（维度数），所以按它来。

    Raises:
        ValueError: magic 不合法或长度对不上。
    """
    if len(raw) < 4:
        raise ValueError(f"IDX 数据太短：{len(raw)} 字节。")

    magic = raw[0:4]
    if magic[0] != 0 or magic[1] != 0:
        raise ValueError(f"IDX magic 不合法：{magic.hex()}。前两字节应为 0x0000。")

    dtype_code, ndim = magic[2], magic[3]
    dtypes = {0x08: np.uint8, 0x09: np.int8, 0x0B: np.dtype(">i2"), 0x0C: np.dtype(">i4")}
    if dtype_code not in dtypes:
        raise ValueError(f"IDX 元素类型码 0x{dtype_code:02x} 不支持。")
    if ndim < 1:
        raise ValueError(f"IDX 维度数必须 >= 1，收到 {ndim}。")

    header_end = 4 + 4 * ndim
    if len(raw) < header_end:
        raise ValueError(f"IDX 头部需要 {header_end} 字节，只有 {len(raw)} 字节。")

    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    dtype = np.dtype(dtypes[dtype_code])
    count = int(np.prod(shape))
    # 按**字节**算，不是按元素个数。MNIST 全是 uint8（1 字节）所以两者相等，
    # 但 IDX 也用于 int16/int32——按元素个数校验会把合法数据判成截断。
    expected = header_end + count * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"IDX 长度与形状不符：形状 {shape} 需要 {expected} 字节，实得 {len(raw)} 字节。")

    array = np.frombuffer(raw, dtype=dtype, count=count, offset=header_end)
    return array.reshape(shape)


# ────────────────────────────── 下载与校验 ──────────────────────────────


def md5_of(path: Path) -> str:
    """分块算 MD5，避免把大文件整个读进内存。"""
    digest = hashlib.md5()  # noqa: S324 - 完整性校验，不是安全机制；见模块 docstring
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """流式下载到 ``target``（先写 ``.part``，成功后再改名）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(target)


def fetch(spec: DatasetSpec, data_file: DataFile, root: Path, *, verify_only: bool) -> Path:
    """确保某个文件存在且校验通过，返回它的路径。"""
    target = root / spec.name / data_file.filename

    if target.exists() and md5_of(target) == data_file.md5:
        print(f"  已就绪  {data_file.filename}")
        return target
    if target.exists():
        print(f"  校验失败，重新下载  {data_file.filename}", file=sys.stderr)
    if verify_only:
        raise SystemExit(
            f"{target} 缺失或校验不通过，而指定了 --verify-only。请先不带该参数跑一次。"
        )

    last_error: Exception | None = None
    for mirror in spec.mirrors:
        url = mirror + data_file.filename
        try:
            print(f"  下载    {data_file.filename}  <- {mirror}")
            download(url, target)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"  镜像不可用（{exc}），换下一个", file=sys.stderr)
            continue

        actual = md5_of(target)
        if actual == data_file.md5:
            print(f"  校验通过  {data_file.filename}  md5={actual}")
            return target

        # 校验不过就删掉，免得半截文件在下次运行时被当成"已就绪"。
        target.unlink()
        last_error = ValueError(f"{data_file.filename} 的 md5 是 {actual}，期望 {data_file.md5}")

    raise SystemExit(f"全部镜像都失败：{last_error}")


def preprocess(spec: DatasetSpec, raw_path: Path, out_path: Path) -> None:
    """把校验通过的原始文件解析成 ``.npy``，供下游直接 ``np.load``。"""
    import gzip

    with gzip.open(raw_path, "rb") as handle:
        array = read_idx(handle.read())
    np.save(out_path, array)
    print(f"  预处理  {out_path.name}  shape={array.shape}  dtype={array.dtype}")


def prepare(name: str, data_dir: Path, *, verify_only: bool) -> None:
    """下载、校验并预处理一个数据集。"""
    spec = DATASETS[name]
    root = data_dir
    print(f"\n[{spec.name}]  {spec.license_note}")

    for data_file in spec.files:
        raw = fetch(spec, data_file, root, verify_only=verify_only)
        print(f"          sha256={sha256_of(raw)}")
        out_name = spec.parse.get(data_file.filename)
        if out_name:
            out_path = root / spec.name / out_name
            if verify_only or not out_path.exists():
                preprocess(spec, raw, out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("datasets", nargs="*", help="要处理的数据集名（默认全部）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="数据落盘目录")
    parser.add_argument("--verify-only", action="store_true", help="只校验已有文件，不下载")
    parser.add_argument("--list", action="store_true", help="列出已登记的数据集")
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in DATASETS.items():
            print(f"{name}: {len(spec.files)} 个文件")
            print(f"  {spec.license_note}")
        return 0

    names = args.datasets or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        print(f"未登记的数据集：{unknown}。已登记的有 {list(DATASETS)}。", file=sys.stderr)
        return 2

    for name in names:
        prepare(name, args.data_dir, verify_only=args.verify_only)

    print(f"\n完成。数据在 {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
