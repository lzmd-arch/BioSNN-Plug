# 复现清单

> 计划书 §12.4 要求"每个阶段的精确环境（commit hash、依赖版本、随机种子）"。
> 这份文档给出模板与骨架库阶段的实际记录。

## 为什么需要它

这个项目的核心主张之一是**"所有结论可在公开仓库中复现"**（计划书 §1.2 第 5 条）。
而复现失败最常见的原因不是代码错了，是**环境对不上**：换了 numpy 版本、换了
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

## 提交前自检

```bash
uv sync --locked                        # lock 与 pyproject 必须一致
uv run pytest -q                        # 全绿
uv run pre-commit run --all-files       # 含文档代码块、版本号、引用清单检查
uv run python examples/quickstart_register_plugin.py
```

## 骨架库阶段的环境记录

第零阶段（骨架库与开源基建）不涉及 GPU 实验，结论都是**确定性**的——
测试套件在 CPU 上跑，结果与硬件无关。因此这一阶段只需要记录依赖版本。

| 项目 | 值 |
| :--- | :--- |
| 骨架库版本 | `biosnn-bus` 0.1.0 |
| Python | >= 3.10（CI 覆盖 3.10 / 3.11 / 3.12） |
| 运行时依赖 | 仅 `numpy>=1.24` |
| 随机性 | 仅稀疏随机投影，见上 |
| 硬件要求 | 无。CPU 即可 |

## 后续阶段的记录位置

从第一阶段（单规则验证）起，每篇实验记录都要带上完整模板。GPU 实验另外需要记录：

- 显存峰值（计划书 §6.2 把 8GB 列为硬约束）
- 是否触发了计划书 §6.2 的降级路径（规模降级 / 分块训练 / INT8 痕迹量化）
- 训练时长与能耗相关量（§9 的能效指标以"调用频率-能耗曲线"形式报告）

## 数据

按计划书 §12.1，本仓库**不提供数据镜像**。下载与预处理脚本将随第一阶段提供——
目前 `scripts/` 下只有仓库自身的检查脚本，还没有任何数据脚本。

约定：数据脚本会把数据落到 `.gitignore` 忽略的 `data/` 下，并校验哈希。
