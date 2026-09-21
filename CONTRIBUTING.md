# 贡献指南

感谢愿意花时间。先说三件需要你知道的事，免得白费功夫：

1. **这是一个研究原型，不是生产框架。** 有些看起来"显然该加"的功能会被搁置，
   理由是它不在计划书的路线里。
2. **单人维护，issue 与 PR 的响应不设 SLA。** 管理预期比假装有团队更利于信任。
   带完整复现步骤的报告会被优先处理。
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

这条规则值得单独说，因为它跟多数项目不一样。

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

1. 在 [`docs/references.md`](docs/references.md) 里登记，标明**核验状态**；
2. 核验四件事：存在性、元数据、归属（别把 A 的结论记到 B 头上）、
   **正文里归因于它的具体数字能否在原文中找到**；
3. 核验不到就如实标 `unverified` 或 `partial`，并在备注里写清哪一项没对上。

这个仓库真实发生过一次署名错误：参考文献 [8] 的作者被误署为 "Khacef et al."，
v6.1 才更正为 Hajizada et al.。**一份诚实的 `unverified` 比一份假的 `verified`
有价值得多。**

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

ADR 一旦记录就不改正文——改变主意时新写一篇并互相链接。历史比整洁重要。

## PR 流程

1. Fork 或开分支（分支名建议 `feat/...`、`fix/...`、`docs/...`）；
2. 改动 + 测试。**修 bug 要带一个能复现该 bug 的测试**；
3. 跑通提交前核查清单；
4. 提 PR，填好 PR 模板里的核查清单。

CI 必须全绿。目前 CI 覆盖 Python 3.10 / 3.11 / 3.12，全部在 CPU 上跑——
骨架库刻意不依赖 GPU（见 [ADR-0002](docs/adr/ADR-0002-numpy-core-torch-optional.md)），
所以如果你发现自己的改动需要 GPU 才能测，那多半意味着抽象层次放错了。

## 第三方插件

**不要往本仓库提 PR。** 在你的包里声明 entry point，用户侧 `discover_plugins()`
就能发现它：

```toml
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

完整教程见 [`docs/plugin_guide.md`](docs/plugin_guide.md)。

如果你做出了一个模态插件，欢迎开 issue 告诉大家，我们会把它列进 README。
这也正是计划书 §12.6 里"第三方插件生态"从愿景变成事实的路径。

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

## 报告问题

用 [issue 模板](https://github.com/lzmd-arch/BioSNN-Plug/issues/new/choose)。
Bug 报告请务必带上**最小复现脚本**与**环境信息**——模板里给了现成的命令。

没有复现步骤的报告往往只能停在"无法重现"。

## 许可证

贡献即表示你同意你的贡献以 [Apache-2.0](LICENSE) 授权（代码）
或 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)（文档）发布。
