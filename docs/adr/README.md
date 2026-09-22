# 架构决策记录（ADR）

> 计划书 §12.4 把 ADR 列为"对抗 bus factor = 1 的核心手段"。本项目单人维护，
> 代码能说明"怎么做"，说不清"**为什么这么做**"和"**试过什么别的**"——ADR 补这一段。

## 什么该写 ADR

每个**重大取舍**一篇。判断标准很简单：如果后来有人问"这里为什么不那样写"，
而且答案不在代码里，就该有一篇 ADR。

## 格式

采用 Michael Nygard 的经典四段式：

| 段落 | 内容 |
| :--- | :--- |
| **Context** | 当时的约束与背景。不写约束的 ADR 无法被重新评估。 |
| **Decision** | 决定了什么。用陈述句，写"我们做了 X"。 |
| **Consequences** | 好处、代价、以及**接受了哪些坏处**。 |
| **Alternatives** | 认真考虑过并否决的方案，以及否决的理由。 |

## 状态

`提议` / `已采纳` / `已废弃`（被哪一篇取代）/ `已取代`。

**ADR 一旦记录就不修改正文。** 改变主意时新写一篇，并在两边互相链接。

## 索引

| 编号 | 标题 | 状态 |
| :--- | :--- | :--- |
| [ADR-0001](ADR-0001-skeleton-as-separate-library.md) | 骨架库独立为 `biosnn-bus` | 已采纳 |
| [ADR-0002](ADR-0002-numpy-core-torch-optional.md) | 骨架库核心基于 numpy，torch 作为可选桥接 | 已采纳 |
| [ADR-0003](ADR-0003-entry-point-plugin-discovery.md) | 插件发现用 entry points 而非全局注册表 | 已采纳 |
| [ADR-0004](ADR-0004-monorepo-uv-workspace.md) | 采用 monorepo + uv workspace，而非双仓库 | 已采纳 |
| [ADR-0005](ADR-0005-versioning-policy.md) | 版本策略：骨架库 semver，研究代码 12 个月免责期 | 已采纳 |
| [ADR-0006](ADR-0006-plugin-interface-fidelity.md) | `ModalityPlugin` 保持计划书 §2.3 的原始签名 | 已采纳 |
| [ADR-0007](ADR-0007-td-ltp-critic-provenance.md) | Critic 采用 TD-LTP，出处为 Frémaux et al. (2013) | 已采纳 |
| [ADR-0008](ADR-0008-spikingjelly-license-and-python-floor.md) | SpikingJelly 采用 OIOSL 1.0，并据此提升开发环境 Python 下限 | 已采纳 |
