# ADR-0004：采用 monorepo + uv workspace，而非双仓库

- **状态**：已采纳
- **日期**：2026-09-21
- **相关**：计划书 §12.2 分层开源路线；[ADR-0001](ADR-0001-skeleton-as-separate-library.md)、[ADR-0005](ADR-0005-versioning-policy.md)

## Context

[ADR-0001](ADR-0001-skeleton-as-separate-library.md) 决定把骨架库抽成独立分发包。
"独立分发包"不等于"独立仓库"——这是两件事，容易混为一谈。

计划书 §12.2 的分层开源路线写的是"骨架库（biosnn-bus）先行开源"，字面上更像
两个仓库。但计划书同时明确本项目是**单人维护**（§12.5：issue 响应不设 SLA，
在 README 明示）。

单人维护是这里的决定性约束。两个仓库意味着：两套 CI、两次提交、两个 issue 队列、
以及最麻烦的——**跨仓库改动**。骨架库改一个接口、研究代码跟着改，要提两个 PR、
等两次 CI、在两边互相引用 commit hash。

## Decision

**单仓库 monorepo**，骨架库作为子包：

```text
BioSNN-Plug/
├─ packages/biosnn-bus/    独立分发包：独立 pyproject、独立版本号、独立测试
├─ research/               研究代码（第一阶段起）
└─ docs/  examples/  scripts/
```

用 **uv workspace** 管理：根 `pyproject.toml` 声明成员，`uv sync` 一条命令把
开发环境装好，骨架库以 editable 方式安装、改代码即时生效。

关键点：**它仍然是可独立发布的分发包**。`uv build --package biosnn-bus` 产出的
wheel 与从独立仓库构建的完全一样，可以单独发到 PyPI。仓库形态与分发形态解耦。

## Consequences

**好处**

- 一次提交可以同时改骨架库与研究代码，CI 一次跑完；
- 一个 issue 队列、一套治理文件；
- `uv sync` 一条命令装好全部开发依赖；
- 骨架库的"独立性"由**依赖图**保证（它不 import 认知核心，CI 里断言装完 wheel 后
  没有 torch），而不是由目录位置保证——后者是形式，前者是实质。

**代价（已接受）**

- 仓库对只关心骨架库的人显得臃肿。`packages/biosnn-bus/README.md` 与发布到 PyPI 的
  README 独立，这一点上做了缓解；
- 骨架库的 CI 与研究代码的 CI 在同一个 workflow 里。目前用 job 分离，如果将来研究
  代码的 CI 变得很重（GPU 测试、长训练），需要加路径过滤；
- 如果将来真的出现活跃的第三方维护者，可能需要把骨架库拆出去。**那时再拆**——
  拆分比重组容易得多。

**没有解决的**

- GitHub 的 contributor 统计、star 数都是按仓库算的，骨架库无法独立积累这些指标。
  考虑到计划书 §12.6 明确说"被同行论文引用"比 star 数更真实，这个损失可接受。

## Alternatives

**A. 双仓库：`biosnn-bus` 与 `BioSNN-Plug` 分开。**
否决（现阶段）。单人维护下，跨仓库改动的摩擦会实实在在地拖慢开发，而它换来的
"骨架库先行开源"这个叙事效果，用 monorepo 的子包同样能达到。
**这个决定是可以反悔的**：`packages/biosnn-bus/` 目录本身就是一个完整可发布的包，
真要拆时 `git subtree split` 即可，历史都能带走。

**B. 单仓库，但骨架库不独立打包（直接 `biosnn_bus/` 顶层目录）。**
否决。那就回到了 [ADR-0001](ADR-0001-skeleton-as-separate-library.md) 否决的方案：
无法独立发布，外部用户装不到。

**C. 用 git submodule 把骨架库挂进来。**
否决。submodule 的日常操作（更新指针、detach HEAD、忘记 `--init`）对贡献者是纯粹
的摩擦，而收益只是"看起来像两个仓库"。
