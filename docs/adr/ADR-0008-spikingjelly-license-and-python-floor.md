# ADR-0008：SpikingJelly 的许可证是 OIOSL 1.0，并据此提升开发环境的 Python 下限

- **状态**：已采纳
- **日期**：2026-09-22
- **相关**：计划书 §1.4、§七 第一阶段、§八、§12.1、§12.3；[docs/references.md](../references.md) 的 v6.3 待更正表

## Context

计划书 §1.4 与 §八 把 **SpikingJelly** 列为本项目的框架选型。§12.1 的许可证策略表里，源代码一行写着：

> 源代码 | **Apache-2.0** | 与 SpikingJelly（同一许可证）生态兼容；专利授权条款对工业用户友好

第一阶段的第一个任务是「环境搭建」，也就是真正把 SpikingJelly 装进来。装之前核了它的许可证，结论与上面那句话不符。

### 事实一：SpikingJelly 用的不是 Apache-2.0，是启智开源许可证 1.0

**启智开源许可证 1.0（Open-Intelligence Open Source License，OIOSL）**，可核的证据：

- 仓库根有 `LICENSE`（该许可证的中文正文）与 `LICENSES/`（译本与使用分发指南）；README 的 License 一节写的是 "SpikingJelly is distributed under the Open-Intelligence Open Source License Version 1.0"；
- GitHub API 的 `license` 字段是 `NOASSERTION`——licensee 未能把它识别为标准许可证；
- PyPI 上稳定版 `0.0.0.0.14` 的 classifier 是 `License :: Other/Proprietary License`；`2.0.0rc1` 的 License 字段为空。

据许可证正文，OIOSL 1.0 的实际条件包括：

- 源码与可执行码形式的**使用与再发布都允许**，须保留许可条件、许可证声明与免责声明；
- **商业目的的使用或再发布，事前须在 AITISA**（新一代人工智能产业技术创新战略联盟，aitisa.org.cn）**官网声明披露**，至少含使用者/再发布者名称、联系方式、地址、电话、电子邮箱、使用目的；
- 专利许可写在「自愿性专利声明」里：若许可者拥有覆盖本软件的专利，对不拥有相关专利的使用者，在 FRAND-RF / 加入专利池 / FRAND 三者中选一项；对拥有相关专利的使用者，许可者「有权利但无义务」按对等原则许可。

### 事实二：本仓库现有的许可审计会静默放行它

`scripts/check_licenses.py` 按许可证名**前缀**比对 `FORBIDDEN_PREFIXES = ("AGPL", "GPL", "SSPL", "BUSL", "CPAL", "OSL")`。`Open-Intelligence Open Source License` 与这六个前缀都不匹配；而 SpikingJelly 的 wheel 元数据里 License 字段是空的，于是它落进 `unknown` 桶——而 CI 调这个脚本时没传 `--strict`，**unknown 只打印、不失败**。

计划书 §12.3 说许可审计的作用是拦截这类问题。它拦不住这一个。

### 事实三：SpikingJelly 2.0.0rc1 要求 Python >= 3.11

`2.0.0rc1` 的 `requires_python` 是 `>=3.11`，依赖 `torch>=2.6`。而它是**唯一**支持现代 torch 的版本线：上一个稳定版 `0.0.0.0.14` 发布于 2023-03，早于 numpy 2 与 torch 2.14。本仓库 root 工程当时是 `requires-python = ">=3.10"`，本机 Python 是 3.10.11。

### 顺带核实的一项

计划书 §八 里的另一个组件 **eprop-PyTorch 是 Apache-2.0**，许可证本身合规。但最后一次提交是 2022-02-18。

## Decision

**1. 保留 SpikingJelly，接受 OIOSL 1.0 的条件。** 替换它等于放弃计划书 §1.4/§八 指定的技术栈，而它在本项目里只承担两件事：提供 LIF/ALIF 神经元原语，以及作为代理梯度 BPTT 的对照基线。本项目是研究原型、非商业用途，商业使用披露义务在当前形态下不触发；但它是一条**下游使用者会继承的义务**，所以必须写在文档里，而不是留在检测不到的角落。

**2. root 工程 Python 下限升到 >= 3.11**，SpikingJelly 锁 `==2.0.0rc1`。

代价是 CI 的 pytest 矩阵从 3.10/3.11/3.12 收为 3.11/3.12。**`biosnn-bus` 自己的 `requires-python >=3.10` 不变**——它确实只依赖 numpy，3.10 能跑，这个对外声明不该被 root 开发环境的下限绑架。为了不让这句声明变成空话，CI 的 `package` job 扩成 3.10 与 3.12 两腿：在干净 venv 里装 wheel 再跑骨架库测试套件。**改的是开发环境的下限，不是分发包的下限。**

**3. eprop-PyTorch 只作阅读参考与交叉验证，不进依赖。** 许可证合规，但 2022 年后停更，锁的 torch API 与 2.14 对不上。e-prop 自实现；最困难处（ALIF 资格痕迹）拿它的源码当对照物逐项核对，**冲突时以 Bellec et al. 2020 原文为准**。

**4. 许可审计收紧：把「检测不到」变成「必须显式声明」。** `scripts/check_licenses.py` 新增白名单，登记**已知并接受**的非标准许可证（每条附理由与本 ADR 的链接），并新增 `--strict-unknown`：未被白名单登记的 unknown 一律失败。CI 改用 `--strict-unknown`，且审计前 `uv sync` 带上 `--group research`——否则 torch / SpikingJelly / gymnasium 根本不进审计范围。

白名单是漏洞，所以每条都要写理由。这是 §12.5 给 `EXEMPT` 表立的同一条规矩。

**5. 计划书 §12.1 需要 v6.3 更正**，把「与 SpikingJelly（同一许可证）生态兼容」改成据实的表述，并补上 OIOSL 的商业使用披露义务。在更正之前，凡引用 §12.1 的场合以本 ADR 为准。

## Consequences

**好处**

- 环境搭建不再建立在一句没核过的许可证断言上；
- 许可审计从「只拦强 copyleft」变成「未知即失败、接受须登记」。OIOSL 这类**既不是 copyleft、又不是标准开源许可证**的情况，终于有了归宿；
- 骨架库对外的 Python 支持声明有了真实覆盖（CI 3.10 腿装 wheel 跑测试套件），而不是随 root 一起被抬走；
- eprop-PyTorch 的定位写清楚了，将来有人问「为什么不直接用它」有答案。

**代价（已接受）**

- **OIOSL 的商业使用披露义务随分发传递。** 本仓库自身的研究用途不触发它；但若有人把本项目（连带 SpikingJelly 依赖）用于商业目的，须自行承担对 AITISA 的披露义务。这一点要写进 README 的依赖说明，不能只躺在 ADR 里。
- **依赖一个预发布版。** `spikingjelly==2.0.0rc1` 精确锁定；它升到正式版时本 ADR 需重新评估——**尤其是许可证条款是否变化**。
- CI 的 pytest 少了 3.10 一腿；3.10 上的骨架库测试改由 `package` job 承担，覆盖方式从「源码树直跑」变成「装 wheel 再跑」，两者不完全等价。
- `uv.lock` 同时记录 CPU 与 cu130 两套 torch 轮子，文件更大，且 torch 升级时 lock 的 diff 会明显变吵。

**明确不声称的**

1. **不声称 OIOSL 1.0「不是开源许可证」。** 本 ADR 记录的是**可核的事实**：它不是 Apache-2.0，GitHub 的 licensee 未能识别（`NOASSERTION`），PyPI 稳定版标 `Other/Proprietary`，并带有商业使用披露义务。**本次未逐一核对 OSI 官方认可全表**（opensource.org 的许可证列表页为脚本不可读），故不对其 OSI 状态下断言。
2. **不声称已做法律审查。** 以上是许可证正文的摘读，不是法律意见。若本项目形态变化（例如接受商业赞助、或对外提供托管服务），应重新评估。
3. **不声称自实现神经元原语是等价替代。** e-prop 的正确性依赖 ALIF 动力学的精确实现，自实现同样有风险；本 ADR 选择保留 SpikingJelly，正是为了不把这块风险引进来。

## Alternatives

**A. 自实现 LIF/ALIF 神经元原语，彻底不引入 SpikingJelly（约 200–300 行）。**

否决。好处明确：零许可牵连、Python 下限不动、CI 三腿全保留。代价是 ALIF 资格痕迹的正确性全压在自己身上——而它恰恰是第一阶段最需要外部参照的部分。计划书 §1.4/§八 指定 SpikingJelly 是同向的判断。若将来 OIOSL 的条款收紧到不可接受，这是成本最低的退出路径。

**B. 锁旧稳定版 `spikingjelly==0.0.0.0.14`，Python 下限不动。**

否决。该版本发布于 2023-03，早于 numpy 2 与 torch 2.14；装上去要么 import 失败，要么被迫把 torch 一起降到旧版——而旧版 torch 不支持 RTX 5060 的 sm_120。这不是「保守」，是把环境钉死在一个跑不了本项目硬件的组合上。

**C. 换一个 Apache/MIT 许可的 SNN 框架。**

暂缓，不是否决。它能同时解决许可与 Python 下限两个问题，但会偏离计划书的技术选型，且 §七 三条线的神经元原语与代理梯度基线都要重新对一遍。**本 ADR 不评估这条路线的可行性**——若 OIOSL 的商业使用披露义务将来成为实际障碍，应新开一篇 ADR 认真比较。

**D. 保持审计脚本原样，仅在文档里注明 SpikingJelly 的许可证。**

否决。这正是「检测不到的角落」——§12.3 把许可审计列为 CI 的一项，图的就是不依赖人记得。文档会漂移，`--strict-unknown` 不会。
