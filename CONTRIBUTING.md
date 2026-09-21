# 贡献指南

感谢愿意花时间。先说三件需要你知道的事，免得白费功夫：

1. **这是一个研究原型，不是生产框架。** 有些看起来"显然该加"的功能会被搁置，
   理由是它不在计划书的路线里。
2. **单人维护，issue 与 PR 的响应不设 SLA。** 带完整复现步骤的报告会被优先处理。
3. **写模态插件不需要往这里提 PR。** 独立成包即可，见下面的"第三方插件"一节。

## 开发环境

需要 [uv](https://docs.astral.sh/uv/)（Python 包与项目管理器）与 Python 3.10+。

```bash
git clone https://github.com/lzmd-arch/BioSNN-Plug.git
cd BioSNN-Plug
uv sync                    # 一条命令装好全部开发依赖（含骨架库 editable 安装）
uv run pre-commit install  # 装提交钩子（首次需要联网拉取钩子仓库）
```

验证环境：

```bash
uv run pytest -q                                  # 测试全绿
uv run python examples/quickstart_register_plugin.py   # 示例能跑
```

## 提交前核查清单

计划书 §12.5 要求 PR 前做"**引用点验 + 代码块检查**"。逐项确认：

```bash
uv run pre-commit run --all-files    # 下面这些检查全都跑一遍
```

| 检查 | 脚本 | 拦什么 |
| :--- | :--- | :--- |
| 单元测试 | `pytest` | 实现错误 |
| 风格 | `ruff check` / `ruff format` | 格式与常见 bug |
| 文档代码块 | `scripts/check_doc_code_blocks.py` | 文档里的代码跑不通 |
| 版本号一致性 | `scripts/check_version_consistency.py` | 文件名与内容版本号错位 |
| 引用清单 | `scripts/check_references.py` | 编号跳号、死链格式、重复条目 |
| notebook 同步 | `scripts/build_notebook.py` | Colab notebook 与示例脚本漂移 |

CI 会再跑一遍，另外加 **lychee 链接存活检查**与**依赖许可审计**。

### 文档代码块：默认执行，例外显式标注

`scripts/check_doc_code_blocks.py` **会真的执行** Markdown 里的 Python 代码块。
原因是纯语法检查（`ast.parse`）抓不到计划书 §12.3 点名的那类问题——
`self.threshold` 语法完全合法，错在于运行时这个属性不存在。

- 同一个文档里的代码块**共享命名空间、按出现顺序执行**。写教程时可以像写脚本一样
  一路写下去；
- 无法独立运行的片段（依赖第三方包、只是示意），在围栏上标注：

  ````text
  ```python no-run
  bus.register(get_plugin("audio")())   # 依赖一个本仓库没有的第三方包
  ```
  ````

**`no-run` 是给"按定义跑不了"的代码留的口子，不是绕过检查的后门。**
如果一段代码明明能在本仓库里跑却标了 `no-run`，review 时会要求改回来。

### 引用点验

新增或修改任何外部引用时：

在 [`docs/references.md`](docs/references.md) 里登记并标明**核验状态**。

核验规则（要核哪四件事、各状态什么含义、核不到怎么办）由那个文件定义——它是引用
规则的唯一归属。这个仓库真实发生过一次署名错误，起因就记在那里。

## 改动类型与兼容性

| 改动 | 要求 |
| :--- | :--- |
| 骨架库 `packages/biosnn-bus/` | 遵循 semver。破坏性变更需升版本号；新增能力优先做加法，不要改现有签名 |
| 研究代码 `research/` | 前 12 个月（至 2027-09）不做兼容性承诺，可自由重构 |
| 文档 / 工具链 | 无特殊要求 |

骨架库当前是 `0.x`。按 semver 约定，`0.x` 的次版本号允许破坏性变更——这是刻意的，
为了让它在还没有外部使用者时快速迭代。第一个被外部依赖的版本会升到 `1.0.0`。

### 改了重大取舍？

如果这个 PR 做了一个后来人会问"为什么不那样写"的决定，**新增一篇 ADR**。
格式与判断标准见 [`docs/adr/README.md`](docs/adr/README.md)。

ADR 一旦记录就不改正文——改变主意时新写一篇并互相链接。

## PR 流程

1. Fork 或开分支（分支名建议 `feat/...`、`fix/...`、`docs/...`）；
2. 改动 + 测试。**修 bug 要带一个能复现该 bug 的测试**；
3. 跑通提交前核查清单；
4. 提 PR，填好 PR 模板里的核查清单。

CI 必须全绿。目前 CI 覆盖 Python 3.10 / 3.11 / 3.12，全部在 CPU 上跑——
骨架库刻意不依赖 GPU（见 [ADR-0002](docs/adr/ADR-0002-numpy-core-torch-optional.md)），
所以如果你发现自己的改动需要 GPU 才能测，那多半意味着抽象层次放错了。

## 第三方插件

**不要往本仓库提 PR。** 在你自己的 `pyproject.toml` 里声明一个 entry point
（group 为 `biosnn_bus.plugins`）即可，本仓库零改动。

配置怎么写、用户侧怎么接入，见
[`docs/plugin_guide.md`](docs/plugin_guide.md#让第三方包提供插件)。

做出了模态插件欢迎开 issue 告诉大家。

## 发布

`biosnn-bus` 通过 [`.github/workflows/release.yml`](.github/workflows/release.yml)
发布：推一个 `v<version>` tag 即触发。流程与首次发布前必须做的 PyPI 侧配置
（trusted publishing 登记）都写在该文件顶部的注释里。

维护者摘要：

```bash
# 1. 改 packages/biosnn-bus/pyproject.toml 的 version，并同步 __init__.py 的 __version__
#    （scripts/check_version_consistency.py 会校这两处 + 计划书文件名）
# 2. 提交推送后打 tag
git tag v0.1.0 && git push origin v0.1.0
```

`verify` job 会先确认 tag 名与包版本一致——写错 tag 会在这一步被拦下，不会出现
"发了 v0.2.0 但包内容还是 0.1.0"这类事后难补救的情况。

PyPI 尚未配置好之前，可以用 `gh workflow run release.yml` 手动触发流水线：
它只跑校验与构建，不发布。

## 文档翻译

文档有中文 / English / 日本語 三版。改动任何一份受管文档时，**三份要一起改**——
漂移的译文比没有译文更误导人，读者会以为读到的是当前状态。

`scripts/check_translations.py` 拦四类漂移，本地跑一次即可自查：

```bash
uv run python scripts/check_translations.py
```

| 它会拦什么 | 怎么发生的 |
| :--- | :--- |
| 漏译 | 新增文档时忘了补 `.en.md` / `.ja.md` |
| 标题结构错位 | 译文漏掉一节，或把 `###` 写成 `##` |
| 代码块被改坏 | 翻译时手滑改了参数值、标识符或命令 |
| 内部链接指回别的语言 | 译文的链接没跟着文件名一起改 |

**文档正文翻译，代码不翻译。** 代码块里的注释与 docstring 可以译；变量名、命令、
路径、注册名等字符串字面量不行——`get_plugin("audio")` 里的 `"audio"` 译了，代码就
跑不通了（CI 会执行文档里的代码块，所以这类错误当场暴露）。

术语以 [`docs/GLOSSARY.md`](docs/GLOSSARY.md) 为准；表里没有的新术语，先补表再用。

用任何语言提 issue 或 PR 都可以。

## 报告问题

用 [issue 模板](https://github.com/lzmd-arch/BioSNN-Plug/issues/new/choose)。
Bug 报告请务必带上**最小复现脚本**与**环境信息**——模板里给了现成的命令。

## 许可证

贡献即表示你同意你的贡献以 [Apache-2.0](LICENSE) 授权（代码）
或 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)（文档）发布。
