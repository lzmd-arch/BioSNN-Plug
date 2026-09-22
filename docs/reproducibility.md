# 复现清单

> 计划书 §12.4 要求"每个阶段的精确环境（commit hash、依赖版本、随机种子）"。

## 为什么需要它

复现失败最常见的原因不是代码错了，是**环境对不上**：换了 numpy 版本、换了
PyTorch 版本、随机种子没记、跑的是本地未提交的改动。

所以每次对外报告实验结论时，都要附上下面这段。**没附的结论视为未经复现。**

## 模板

复制这段填好，放进实验记录或 issue 里：

```text
实验名称：
日期：
Git commit：          # git rev-parse HEAD
Git 状态：            # git status --porcelain 的输出；非空说明有未提交改动
Python：              # python -V
操作系统 / 架构：
硬件：                # GPU 型号 + 显存
随机种子：            # 实验里用到的每一个，注明各自的用途
依赖快照：            # uv.lock 的哈希，或 `uv pip freeze` 的输出
运行命令：
耗时：
```

## 怎么拿到这些信息

本仓库用 `uv` 管理依赖，`uv.lock` 提交进版本库——这是复现的基础：
**同样的 lock 文件解析出同样的依赖版本**。

```bash
git rev-parse HEAD                     # commit hash
git status --porcelain                 # 有输出 = 工作区不干净，复现结果不可信
python -V
uv pip freeze                          # 已安装依赖的精确版本
shasum -a 256 uv.lock                  # Windows 用 Get-FileHash uv.lock
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

一条命令拿全（Windows / Git Bash）：

```bash
echo "commit: $(git rev-parse HEAD)"; \
echo "dirty: $(git status --porcelain | wc -l) files"; \
python -V; uv pip freeze | shasum -a 256; \
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

## 随机种子

**骨架库的随机性只有一处**：脉冲总线末端的稀疏随机投影
（`SparseRandomProjection`）。它的种子由 `SpikeBus(seed=...)` 决定。

约定：

- 骨架库内部**不使用**任何依赖进程状态的随机源。具体说，投影矩阵的种子由
  `zlib.crc32` 从 `(seed, 通道, 模态构成)` 派生，**不是** Python 内置的 `hash()`
  ——后者对字符串按进程加盐（`PYTHONHASHSEED`），会让同一份配置在不同进程里得到
  不同的投影矩阵，"固定种子即可复现"这句话就只在单进程内成立。

  这条有回归测试盯着：`packages/biosnn-bus/tests/test_bus.py::test_projection_seed_survives_a_process_boundary`
  用两个不同的 `PYTHONHASHSEED` 起子进程，比对输出必须一致。

- 插件自己的随机性由插件自己负责。写插件时请把 `seed` 暴露成构造参数，
  不要直接调用 `np.random.random()`。

**研究代码（`research/`）的随机性**走
[`research/common/seeding.py`](../research/common/seeding.py)：先登记用途、再施加。
那里的 `derive_seed` 用 `zlib.crc32` 而非内置 `hash()`，理由与骨架库相同（后者按
进程加盐，跨进程就不可复现），并且有同样的跨进程回归测试盯着。各验证线**不应**
直接 `np.random.seed(...)`——那样种子不会出现在复现记录里。

## 提交前自检

除 [`CONTRIBUTING.md`](../CONTRIBUTING.md) 的通用清单外，复现相关还要确认这一条：

```bash
uv sync --locked      # lock 与 pyproject 必须一致
```

上面"同样的 lock 文件解析出同样的依赖版本"这句话，前提就是这一步通过。

## 骨架库阶段的环境记录

第零阶段（骨架库与开源基建）不涉及 GPU 实验，结论都是**确定性**的——
测试套件在 CPU 上跑，结果与硬件无关。因此这一阶段只需要记录依赖版本。

| 项目 | 值 |
| :--- | :--- |
| 骨架库版本 | `biosnn-bus` 0.1.0 |
| Python | >= 3.10（CI 覆盖 3.10 / 3.11 / 3.12） |
| 运行时依赖 | 仅 `numpy>=1.24` |
| 硬件要求 | 无。CPU 即可 |

## 第一阶段（单规则验证）的环境记录

这一阶段的实验**有 GPU 参与**，因此不再是纯确定性的：同一个配置在 CPU 与 CUDA 上
可能给出不同的浮点结果。所以除了依赖版本，还必须记录硬件与显存。

| 项目 | 值 |
| :--- | :--- |
| Python | **>= 3.11**（SpikingJelly 2.0.0rc1 的下限，见 `docs/adr/ADR-0008`） |
| 关键依赖 | `torch`、`torchvision`、`torchaudio`、`spikingjelly==2.0.0rc1`、`gymnasium`（`research` 依赖组） |
| torch 来源 | 按平台换源：Windows → `download.pytorch.org/whl/cu130`（覆盖 sm_120）；其它平台 → `whl/cpu`。理由见 `pyproject.toml` 的 `[tool.uv.sources]` 注释 |
| 硬件 | NVIDIA GeForce RTX 5060，8,151 MiB，sm_120 |
| 骨架库 | 不受影响：仍只依赖 numpy，仍是 `requires-python >=3.10`（CI 的 `package` job 用 3.10 腿覆盖） |

**依赖组的隔离**：`research` 组默认**不装**。`uv sync --locked` 得到的是第零阶段那个
无 torch 的环境，`uv sync --locked --group research` 才是第一阶段的实验环境。骨架库
"不依赖 torch"（ADR-0002）这句声明因此在本地也仍然可验证。

每个模块的 docstring 顶部带 12 个月破坏性变更免责期与到期日（至 2027-09）。

### 已完成的实验记录

| 验证线 | 记录位置 | 验收数字 |
| :--- | :--- | :--- |
| W1 核化 IB-Hebbian 感知层 | [`research/ib_hebbian/README.md`](../research/ib_hebbian/README.md) | MNIST **98.01%**（监督 SGD 读出）/ **97.79%**（岭回归闭式解读出，现行默认）；阈值 70%。**两种读出都是监督的**——98.01% 是「局部学到的特征 + 监督读出」的联合结果，不是纯局部的精度，见 README 的「「局部」的准确表述」 |
| W2 e-prop 认知层 | [`research/eprop/README.md`](../research/eprop/README.md) | sMNIST 77.48%；活跃神经元比例 1.0000（阈值 > 60%） |
| W3 R-STDP + TD-LTP Critic | [`research/rstdp/README.md`](../research/rstdp/README.md) | ✅ **三项判据全部达到**：CartPole 中位数 **237.8 步**（种子 45–64，20 个**全新**、从未参与任何选择；阈值 ≥ 200）；偏移比最大 **0.0403**（全部种子 < 0.10 ✓）；最小值 **121**（自加的「稳定」条件 ≥ 100 ✓，无种子低于 100）。用于调参的 25–44 上中位数 319.8、最小 109——**两批都过线**。配置与理由见 [ADR-0009](adr/ADR-0009-w3-behaviour-policy-and-trace-centring.md) |

## 后续阶段的记录位置

从第一阶段（单规则验证）起，每篇实验记录都要带上完整模板。这一段由
[`research/common/provenance.py`](../research/common/provenance.py) 生成，用法是：

```python
from research.common.provenance import DegradationLog, collect
from research.common.seeding import SeedBook

book = SeedBook(base=0)
book.derive("权重初始化")

record = collect(
    "ib_hebbian/mnist",
    seeds=book.render(),
    elapsed_s=123.4,
    peak_mb=None,  # 真实实验里传 measure_peak_memory() 给出的 stats["peak_mb"]
    degradation=DegradationLog(scale_reduction=True, notes=["5 万 → 1 万神经元"]),
)
print(record.render_block())  # 可直接粘进 Markdown 的 text 围栏块
```

它自动读出四个**人工填不对**的字段：`git rev-parse HEAD`、`git status --porcelain`
（非空即标"工作区不干净，本结论不可信"）、`uv.lock` 的 SHA-256、以及硬件与显存。

GPU 实验另外需要记录：

- 显存峰值（计划书 §6.2 把 8GB 列为硬约束）——由 `research/common/device.py` 的
  `measure_peak_memory()` 测，取 PyTorch 分配器口径而非 `nvidia-smi`
- 是否触发了计划书 §6.2 的降级路径（规模降级 / 分块训练 / INT8 痕迹量化）——
  `DegradationLog` 登记；**没触发是常态**，所以只有显式登记过的才算数
- 训练时长与能耗相关量（§9 的能效指标以"调用频率-能耗曲线"形式报告）

## 数据

按计划书 §12.1，本仓库**不提供数据镜像**。

下载与预处理脚本是 [`scripts/download_data.py`](../scripts/download_data.py)：

```bash
uv run python scripts/download_data.py --list        # 看已登记的数据集
uv run python scripts/download_data.py mnist         # 下载 + 校验 + 转成 .npy
uv run python scripts/download_data.py --verify-only # 只校验已有文件，不联网
```

数据落到 `.gitignore` 忽略的 `data/` 下，每个文件**逐个校验哈希**后才算可用，并把
通过校验的 SHA-256 打印出来供复现记录留档。

**校验值用的是 MD5，这是刻意的**：它不是安全机制，是完整性校验（防下载截断与镜像
漂移）。选 MD5 是因为对 MNIST 而言它是**唯一有独立第三方公布**的校验值——
`torchvision.datasets.MNIST` 把四个文件的 MD5 硬编码在自己的源码里，本脚本钉的就是
那四个值，可以交叉核对。自己算一个 SHA-256 钉上去只是自己给自己背书。
