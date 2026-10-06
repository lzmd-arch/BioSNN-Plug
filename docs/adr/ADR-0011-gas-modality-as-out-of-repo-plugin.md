# ADR-0011：气体（化学传感）模态以仓库外第三方包提供，不进 packages/

- **状态**：已采纳
- **日期**：2026-10-06
- **相关**：计划书 §2.3 模态表、§七 第二阶段、§12.3、§12.5、§12.6；
  [ADR-0001](ADR-0001-skeleton-as-separate-library.md)、
  [ADR-0003](ADR-0003-entry-point-plugin-discovery.md)、
  [ADR-0004](ADR-0004-monorepo-uv-workspace.md)；[docs/references.md](../references.md)

## Context

计划书 §2.3 的模态表在 v6.3 之前只有文本 / 图像 / 音频三行，对应 §七 第二阶段的「文本 + 图像插件」
与第三阶段的「音频插件接入」。第四模态是**气体（MOS 阵列 / 电子鼻）**，而它的实现来自本仓库之外。

本仓库有两条现成的落地路径，且它们是为**不同目的**设计的：

- `packages/*` —— uv workspace 成员（[ADR-0004](ADR-0004-monorepo-uv-workspace.md)），一等公民。
  被根 `pyproject.toml` 的 `[tool.pytest.ini_options] testpaths` 与 `[tool.coverage.run] source`、
  `ci.yml` 的 `test` 与 `package` job、`release.yml` 的发布链**写死**引用；`packages/*/README.md`
  受三语检查管辖，且 README 里的 Python 代码块会被 CI **真实执行**。
- entry points（[ADR-0003](ADR-0003-entry-point-plugin-discovery.md)）—— 第三方插件通道，
  group `biosnn_bus.plugins`。`CONTRIBUTING.md` 写着「写模态插件不需要往这里提 PR。独立成包即可」，
  并给出 `[project.entry-points]` 的写法。

## Decision

气体模态的实现作为**仓库外的独立分发包**存在，经 entry point（group `biosnn_bus.plugins`）接入，
**本仓库对它的代码零改动**。计划书侧只登记三件事：§2.3 模态表的一行、§七 第二阶段的排期、
以及版本说明的一条修订记录。

## Consequences

先写接受的代价：

- 它**不进本仓库的 CI**——§12.3 那张「CI 实际跑什么」的清单覆盖不到它，它也不随本仓库发版；
- 它依赖 PyPI 上的 `biosnn-bus` 而非 workspace 内的路径版本，因此骨架库一旦破坏性变更，
  **它不会在本仓库的 CI 里先红**；ADR-0005 给骨架库的 semver 是它唯一的保护；
- §12.3 的「最小复现脚本」一行要求每篇核心文献对应一个 `examples/paper_<name>.py`，这一条在本仓
  **没有对应物**——该脚本落在仓库外包内。这是一处**显式记录的偏离**，不是遗漏。

再写得到的好处：

- 本仓库零改动，`testpaths` / `coverage.source` / `ci.yml` / `release.yml` 里那些写死的路径
  一个都不需要动；
- 它是 `CONTRIBUTING.md` 与 [ADR-0003](ADR-0003-entry-point-plugin-discovery.md) 承诺的第三方
  通道的**首次实测**——在此之前，「第三方包零改动接入」在本仓库只有散文声明，没有任何测试跑过
  `discover_plugins()` 命中一个真实的第三方 entry point；
- 它正是 §12.6「第 24 个月：外部贡献的模态插件 ≥ 1」那一格的第一个实例，且**不改变**第三阶段
  「三模态接入」的成功标准——那是项目自研交付物，与外部插件不是同一件事。

## Alternatives

- **放进 `packages/biosnn-plug-gassensor/`**：**否决**。`CONTRIBUTING.md` 对第三方插件的指引是
  明文的反向要求（「不要往本仓库提 PR」）；而且它会连带触发一整套硬性义务——三语 README、
  README 代码块被 CI 真实执行、根 `pyproject.toml` 的 `testpaths` 与 `coverage.source`、
  `ci.yml` 的 `test` 与 `package` job、`release.yml` 的发布链，以及计划书 §12.3 表的登记。
  这些义务是给**项目自研交付物**设计的，加在一个外部模态上不成比例。
- **放进 `research/gas_sensor/`**：**否决**。`research/` 是三条**自研**验证线（W1/W2/W3）的所在，
  享有 12 个月的破坏性变更免责期（ADR-0005），且与骨架库目前完全解耦；把外部实现放进去会同时
  混淆「谁在验证什么」与「谁享有免责期」两件事。
- **把第四模态排进 §七 第三阶段（与音频并列）**：**否决**。第三阶段的任务是「音频插件接入」
  （单模态、项目自研）；第二阶段的「模态插件框架与双通道融合」才是新增模态该在的位置。
