# 术语表 / Glossary / 用語集

> **所有译文的术语以本表为准。** 表中没有的新术语，先补进本表再使用——否则同一个词会
> 在不同文档里出现不同译法，而译文集的可用性取决于一致性，不取决于单篇的遣词。
>
> 本表本身即三语，不设 `.en` / `.ja` 变体（`scripts/check_translations.py` 对它豁免）。

词频一列是第零阶段结束时在各文档里的实测出现次数，用来判断哪些词最不能译错。

## 一、项目核心概念

| 中文 | English | 日本語 | 频次 | 备注 |
| :--- | :--- | :--- | ---: | :--- |
| 计划书 | project plan | プロジェクト計画書 | 104 | 指 `BioSNN-Plug_项目计划书_v6.2.md`。**不译作 "design doc" 或 "spec"** |
| 骨架库 | skeleton library | スケルトンライブラリ | 65 | 指 `biosnn-bus`。**不译作 framework / backbone / scaffold** |
| 认知核心 | cognitive core | 認知コア | 23 | |
| 模态插件 | modality plugin | モダリティプラグイン | 15 | |
| 脉冲总线 | spike bus | スパイクバス | 8 | |
| 融合通道 | fusion channel | 融合チャネル | 3 | `temporal` / `semantic` 两条 |
| 双通道融合 | dual-channel fusion | 二重チャネル融合 | 4 | 指 `temporal` + `semantic` 两条融合通道并存。**本项目的架构设计，不归因于 TAAF**（见 `docs/references.md`） |
| 脉冲序列 | spike train | スパイク列 | 3 | 对应类名 `SpikeTrain`（类名本身不译） |
| 局部学习规则 | local learning rule | 局所学習則 | 1 | |
| 学习规则 | learning rule | 学習則 | — | 泛指；「局部学习规则」见表内 |
| 代理梯度 | surrogate gradient | 代理勾配 | 4 | |
| 资格痕迹 | eligibility trace | 資格痕跡 | 2 | |
| 神经发生 | neurogenesis | 神経新生 | 2 | |
| 仲裁器 | arbiter | アービタ | 2 | ES 元学习仲裁器 |
| 元学习 | meta-learning | メタ学習 | 3 | |
| 工作记忆 | working memory | ワーキングメモリ | 1 | |
| 情景记忆 | episodic memory | エピソード記憶 | 1 | |
| 元认知门控 | metacognitive gating | メタ認知ゲーティング | 2 | |
| 模式分离 | pattern separation | パターン分離 | 1 | |
| 阶段 | phase | フェーズ | 38 | 第零阶段 = phase 0、第一阶段 = phase 1（README 英文版已用该写法）。**不译作 "stage"**——同一份译文里 phase 与 stage 混用，读者会以为是两套东西 |
| 单规则验证 | single-rule validation | 単一規則検証 | 1 | 第一阶段名称（计划书 §七） |
| 开源基建 | open-source infrastructure | オープンソース基盤 | 1 | 第零阶段名称；**不译作 infrastructure building** |
| 复现清单 | reproducibility checklist | 再現チェックリスト | 2 | `docs/reproducibility.md` 的标题；README 英文版已用该写法 |
| 研究原型 | research prototype | 研究プロトタイプ | — | 与"生产框架"相对 |
| 生产框架 | production framework | プロダクションフレームワーク | — | 本仓库刻意不是 |
| 第三方插件 | third-party plugin | サードパーティプラグイン | — | 独立成包、走 entry point，不往本仓库提 PR |
| 模态 | modality | モダリティ | — | 全模态 = fully multimodal；**不译作 "modal"** |
| 融合 | fusion | 融合 | — | 融合策略、融合通道；与「双通道融合」区分 |
| 融合策略 | fusion strategy | 融合戦略 | — | 脉冲总线当前默认的是「平凡的拼接」；`FusionStrategy` 是替换点。**日文版不写作「ストラテジー」** |
| 双通道路由 | dual-channel routing | 二重チャネルルーティング | — | 路由机制。**与「双通道融合」不是一回事**——后者是架构设计 |
| 认知主体 | cognitive subject | 認知主体 | — | SNN 是认知主体，LLM 只负责生成 |
| 全模态 | fully multimodal | 全モダリティ | 3 | 项目定位用语；**不译作 omni-modal** |
| 生物合理性 | biological plausibility | 生物学的妥当性 | 4 | 高生物合理性 = high biological plausibility |
| 感知层 | perception layer | 知覚層 | 11 | 模态插件的接入层（架构图 L1） |
| 认知层 | cognitive layer | 認知層 | 5 | 认知核心所在层（架构图 L3） |
| 执行层 | execution layer | 実行層 | 8 | 动作生成与模态解码（架构图 L4） |
| 调度与协同 | orchestration | オーケストレーション | 2 | 架构图 L5；LLM 调度与协同层 = LLM orchestration layer。**不译作 collaboration**（那是「协同」在「双规则协同」里的意思，见工程节） |
| 推理引擎 | inference engine | 推論エンジン | 8 | LLM 作为可替换的推理引擎。**不译作 reasoning engine** |
| 双通道 | dual-channel | 二重チャネル | 8 | 脉冲总线 `temporal` / `semantic` 两条；与「双通道融合」同源 |
| 依赖组 | dependency group | 依存グループ | — | `pyproject.toml` 的 `[dependency-groups]`；`dev` 与 `research` 两组 |
| 复现记录 | reproducibility record | 再現記録 | — | 一次实验按 §12.4 模板填出的那段文本；由 `research/common/provenance.py` 生成。「复现清单」是本表的另一条，指 `docs/reproducibility.md` 那份文档 |
| 论文复现脚本 | paper reproduction script | 論文再現スクリプト | — | 计划书 §12.3 要求的 `examples/paper_<name>.py`：每篇核心文献一个最小可跑示例，用来拦"论文说能跑但仓库跑不通"。**与「最小复现脚本」区分**——后者指报 bug 时附的那个 |
| 启智开源许可证 1.0 | Open-Intelligence Open Source License 1.0 (OIOSL) | 啓智オープンソースライセンス 1.0 | — | SpikingJelly 使用的许可证。**不是 Apache-2.0**，计划书 §12.1 曾误记为与本项目同一许可证。见 `docs/adr/ADR-0008`。缩写 OIOSL 三语均保留原文 |
| 商业使用披露义务 | commercial-use disclosure obligation | 商業利用の開示義務 | — | OIOSL 1.0 的条件之一：商业目的的使用或再发布前，须在 AITISA 官网声明相关信息 |
| 预发布版 | pre-release | プレリリース | — | `spikingjelly==2.0.0rc1` 即预发布版；uv 默认不选预发布版，因此必须精确锁定 |

## 二、工程与协作

| 中文 | English | 日本語 | 频次 | 备注 |
| :--- | :--- | :--- | ---: | :--- |
| 插件 | plugin | プラグイン | — | 模态插件、第三方插件 |
| 仓库 | repository | リポジトリ | — | 本仓库 = this repository |
| 发布 | release | リリース | — | 「尚未发布到 PyPI」= PyPI にまだリリースしていない |
| 参考实现 | reference implementation | リファレンス実装 | — | ✅ 参考实现 = ✅ リファレンス実装 |
| 示例 | example | サンプル | — | `examples/` 下的可跑脚本；脚本是唯一真相源 |
| 数据集 | dataset | データセット | — | 遵循各数据源许可；本仓库不提供数据镜像 |
| 版本号 | version number | バージョン番号 | 18 | |
| 破坏性变更 | breaking change | 破壊的変更 | 9 | |
| 免责期 | grace period | 免責期間 | 10 | 研究代码头 12 个月的破坏性变更免责期 |
| 复现 | reproduce / reproduction | 再現 | 16 | 可复现 = reproducible = 再現可能 |
| 核验状态 | verification status | 検証ステータス | 6 | 引用清单里的 `verified` / `partial` 等 |
| 核验 | verify / verification | 検証 | 18 | 动词形式；名词见上一条「核验状态」。**不译作 check**——本仓库的 check 特指 CI 脚本（`check_*.py`） |
| 引用清单 | reference list | 引用リスト | 4 | 指 `docs/references.md` |
| 架构决策记录（ADR） | architecture decision record (ADR) | アーキテクチャ決定記録（ADR） | 2 | `docs/adr/`；简称 ADR 不译 |
| ADR 状态 | ADR status | ADR ステータス | 4 | 见 `docs/adr/README.md`。ADR 头部的 **状态** 字段在英文版写作 **Status**、日文版写作 **ステータス** |
| 提议 / 已采纳 / 已废弃 / 已取代 | Proposed / Accepted / Deprecated / Superseded | 提案 / 承認済み / 非推奨 / 置換済み | — | Michael Nygard 的 ADR 四状态 |
| 取舍 | trade-off | トレードオフ | — | 「重大取舍」= significant trade-off；ADR 的判断对象 |
| 签名 | signature | シグネチャ | — | 函数 / 方法签名；`ModalityPlugin` 的原始签名 |
| bus factor | bus factor | bus factor | — | 计划书 §12.4 的原文表述；三语一律保留英文原词，不译作「バス係数」 |
| 契约 | contract | コントラクト | 8 | 插件接口契约 |
| 调用方 | caller | 呼び出し側 | — | 替换融合策略不改动调用方代码 |
| 注册表 | registry | レジストリ | 8 | |
| 装饰器 | decorator | デコレータ | 4 | |
| 结构化协议 | structural protocol | 構造的プロトコル | 4 | Python `Protocol` |
| 抽象基类 | abstract base class | 抽象基底クラス | 2 | ABC |
| 抽象方法 | abstract method | 抽象メソッド | 3 | `@abstractmethod`；`ModalityPlugin` 的三个方法（ADR-0006） |
| 抽象属性 | abstract property | 抽象プロパティ | 2 | `modality_name` / `spike_dim` |
| 实例属性 | instance attribute | インスタンス属性 | 1 | 与「类属性」相对；类定义期还没有值可查，所以不能在 `__init_subclass__` 里校验（ADR-0006） |
| 成员 | member | メンバー | 5 | 接口的成员 = メソッド与属性；「五个成员」= 5 つのメンバー |
| 接口 | interface | インターフェース | — | 插件接口契约 = プラグインインターフェースコントラクト。**日文一律用 インターフェース**，不写 インタフェース（否则同一份译文里两种写法混用） |
| 接口规范 | interface specification | インターフェース仕様 | 3 | 计划书 §2.3 的那份文本；与上一条「接口」区分——「接口」指代码里的东西，「接口规范」指写它的文本 |
| 张量 | tensor | テンソル | — | torch 张量 = torch テンソル |
| 数组 | array | 配列 | — | numpy 数组 = numpy 配列；稠密数组 = 密な配列 |
| 后端 | backend | バックエンド | — | 与后端无关 = バックエンドに依存しない |
| 仿真器 | simulator | シミュレータ | — | 骨架库**不是**轻量 SNN 仿真器（ADR-0002） |
| 自检 | self-check | セルフチェック | 3 | `validate()` |
| 幂等 | idempotent | 冪等 | 2 | |
| 夹具 | fixture | フィクスチャ | 2 | pytest fixture；测试夹具 = テスト用フィクスチャ |
| 进程内注册 | in-process registration | プロセス内登録 | 2 | `@register_plugin` 写入模块级注册表；与 entry points 发现并列 |
| 导入副作用 | import side effect | インポートの副作用 | 1 | 不 import 插件模块就不会注册 |
| 降级路径 | fallback path | 縮退パス | 3 | 计划书 §十一 |
| 维护者 | maintainer | メンテナ | — | |
| 贡献者 | contributor | コントリビュータ | — | |
| 早期采用者 | early adopter | 早期採用者 | — | 0.x 阶段就依赖骨架库的人 |
| 响应预期管理 | expectation management for responses | 応答に関する期待値の管理 | 1 | 计划书 §12.5 的表述：单人维护，issue 响应不设 SLA。**不译作「期待値マネジメント」** |
| 提交前核查清单 | pre-submit checklist | 提出前チェックリスト | — | |
| 钩子 | hook | フック | 2 | |
| 命名空间 | namespace | 名前空間 | 1 | |
| 类型标注 | type hint | 型ヒント | 2 | |
| 静态检查 | static check | 静的チェック | 1 | |
| 入口点 | entry point | エントリポイント | — | packaging 概念 |
| 插件发现 | plugin discovery | プラグイン発見 | — | `discover_plugins`；ADR-0001 日文版作「登録と発見のメカニズム」，`docs/adr/README.md` 日文版作「プラグイン発見」 |
| 分发包 | distribution package | 配布パッケージ | 10 | packaging 概念：`pip install` 装的那个东西，不是"模块"或"仓库" |
| 私服分发 | private-index distribution | プライベートインデックスからの配布 | 1 | Python 打包生态的好处之一；走私服 = プライベートインデックス経由 |
| 基线 | baseline | ベースライン | 4 | 对照基线 |
| 消融 | ablation | アブレーション | 1 | |
| 路线图 | roadmap | ロードマップ | — | 指计划书里的阶段路线 |
| 开发环境 | development environment | 開発環境 | — | |
| 单元测试 | unit test | ユニットテスト | — | |
| 兼容性 | compatibility | 互換性 | — | 兼容性承诺 = 互換性の約束 |
| 次版本号 | minor version | マイナーバージョン | — | semver 的 minor |
| 主版本号 | major version | メジャーバージョン | — | semver 的 major |
| 分支 | branch | ブランチ | — | |
| 提交钩子 | commit hook | コミットフック | — | pre-commit |
| 工具链 | toolchain | ツールチェーン | — | |
| 流水线 | pipeline | パイプライン | — | GitHub Actions 的 workflow |
| 围栏 | fence | フェンス | — | Markdown 的代码围栏；围栏信息串（如 `python no-run`）不译 |
| 最小复现脚本 | minimal reproduction script | 最小再現スクリプト | — | 报告 bug 时的要求 |
| 随机种子 | random seed | 乱数シード | 4 | 复数用 seeds |
| 回归测试 | regression test | 回帰テスト | 1 | |
| 数据镜像 | data mirror | データミラー | 2 | 本仓库不提供 |
| 预处理脚本 | preprocessing script | 前処理スクリプト | 2 | 随第一阶段提供 |
| 能效 / 能耗 | energy efficiency / energy consumption | エネルギー効率 / エネルギー消費 | 1 / 2 | 计划书 §9 |
| 研究代码 | research code | 研究コード | — | 指 `research/` 下的代码；12 个月破坏性变更免责期 |
| 验收项 | acceptance item | 受入項目 | — | 计划书 §9 的验收项；验收标准 = 受入基準 |
| 引用点验 | citation spot-check | 引用スポット検証 | — | 计划书 §12.5 的 PR 前动作之一 |
| 署名错误 | attribution error | クレジット表記の誤り | — | 本仓库真实发生过一次 |
| 死链 | dead link | デッドリンク | — | |
| 链接存活检查 | link liveness check | リンク生存確認 | — | CI 里由 lychee 跑 |
| 依赖许可审计 | dependency license audit | 依存ライセンス監査 | — | CI 的一项 |
| 待办 | to-do | TODO | — | `docs/references.md` 的「待办」一节 |
| 元数据 | metadata | メタデータ | — | 引用核验的四项之一 |
| 归属 | attribution | 帰属 | — | 误归属 = 把 A 的结论记到 B 头上 |
| 书目 | bibliography | 書誌 | — | 著者 / 年份 / 卷期页 / 文章号 |
| 文章号 | article number | 論文番号 | — | Nature 等出版方的 Article number |
| 开放获取 | open access | オープンアクセス | — | |
| 对抗性复核 | adversarial re-verification | 対抗的再検証 | — | 归属类结论的必做步骤 |
| 出处考证 | provenance verification | 出典考証 | — | TD-LTP 的 P0 待办 |
| 全局 | global | グローバル | — | 全局标量、全局反向传播 |
| 分层开源路线 | layered open-source route | 段階的オープンソース化路線 | — | 计划书 §12.2；骨架先行、研究随后、成果公开可验 |
| 单人维护 | single-maintainer | 単独メンテナンス | — | 计划书 §12.5；本项目的事实约束，译文不得弱化 |
| 双仓库 | two repositories | 二リポジトリ | — | 与 monorepo 相对；ADR-0004 否决的方案。**`monorepo` 这一写法在日文版中也保留原样**，与原文一致 |
| 子包 | subpackage | サブパッケージ | — | monorepo 内承载独立分发包的那个目录 |
| 跨仓库改动 | cross-repository change | リポジトリ横断の変更 | — | 双仓库方案最麻烦的那部分摩擦 |
| 依赖图 | dependency graph | 依存グラフ | — | 骨架库独立性的实质保证（不是目录位置） |
| editable 安装 | editable install | editable インストール | — | `uv sync` 之后改代码即时生效 |
| 确定性 | deterministic | 決定論的 | 1 | 与「随机」相对；结论可复现的依据 |
| 依赖快照 | dependency snapshot | 依存関係スナップショット | 1 | 复现模板字段：`uv.lock` 的哈希 |
| 硬约束 | hard constraint | ハード制約 | 1 | 计划书 §6.2 把 8GB 显存列为硬约束 |
| 规模降级 | scale reduction | 規模縮退 | 1 | §6.2 降级路径之一（另两条：分块训练、INT8 痕迹量化） |
| 形状契约 | shape contract | 形状コントラクト | 1 | 插件输入/输出的形状约定 |
| 测试模板 | test template | テストテンプレート | 1 | 插件开发指南给出的测试骨架 |
| 元信息 | plugin metadata | メタ情報 | 1 | `modality_name` / `spike_dim` 这类附带声明；与「元数据 / メタデータ」（引用核验）区分 |
| 示例插件 | example plugin | サンプルプラグイン | 2 | 本仓库自带的 `DiffImagePlugin` |
| 治理 | governance | ガバナンス | — | 开源治理 = オープンソースガバナンス；一套治理文件 = ガバナンス文書が 1 セット |
| 出处 | provenance | 出典 | 20 | 与「出处考证」同源；"出处待补"= provenance to be added。**不译作 source**（"来源"太泛，读者会当成网址） |
| 期刊版 | journal version | ジャーナル版 | 2 | 与会议版 / arXiv 版相对；出处考证时会核对。见 `docs/references.md` [2]、[13] |
| 付费墙 | paywall | ペイウォール | 1 | 期刊版在付费墙后 = ジャーナル版はペイウォールの後ろにある |
| 无监督偏差 | unsupervised bias | 教師なしバイアス | 3 | R-STDP 的无监督偏差，必须引入刺激特异性奖励预测来纠正 |
| Critic | Critic | Critic | 8 | 执行层的奖励预测 Critic。**三语一律保留原词**，不译作「批評者」等 |
| 非局部 | non-local | 非局所 | 3 | 与「局所学習則」相对；非局部量 = 非局所量。TD-LTP 的非局部量只有一个标量 |
| 全局标量 | global scalar | グローバルスカラー | 3 | 见「全局」条；承载 TD 误差 δ(t) |
| 符合窗 | coincidence window | コインシデンス窓 | 1 | pre-before-post 配对的计数窗口；TD-LTP 的因子 1、2 |
| 多巴胺式广播 | dopaminergic broadcast | ドーパミン様ブロードキャスト | 1 | 全局标量 δ(t) 的承载方式 |
| 移动平均 | moving average | 移動平均 | 1 | 移动平均基线 Critic = 移動平均ベースライン Critic（§十一 降级路径之一） |
| 同构 | isomorphic | 同型 | 2 | 「与 §3.2 的设计同构」；独立证据的判据 |
| 规则冲突 | rule conflict | 規則衝突 | 1 | 第二阶段评估的对象；见 `docs/adr/ADR-0007` |
| 符号体系 | notation | 記法体系 | 1 | 论文的方程符号约定；ADR-0007 未逐字核验期刊版的记法 |
| 插件化 | pluginization | プラグイン化 | — | 把某层能力交给插件实现的设计取向；「插件化的边界」= the pluginization boundary |
| 适配层 | adapter layer | アダプタ層 | — | 为桥接不兼容接口而写的中间层；与「结构化协议」相对——后者不需要显式继承 |
| 活跃神经元比例 | active neuron fraction | 活動ニューロン比率 | — | 计划书 §9 的网络健康指标；口径见 `research/common/metrics.py`——观察窗内**至少发放过一次**的神经元占比。§3.1 规定低于 60% 触发阈值调整 |
| 脉冲稀疏度 | spike sparsity | スパイク疎度 | — | 计划书 §9 指标；零元素占比，即 `SpikeTrain.density` 的补数。与「稀疏随机投影」区分——后者是骨架库的一个组件 |
| 死亡神经元 | dead neuron | 死んだニューロン | — | 从不发放的神经元；§3.1「死亡神经元防护」把放电阈值提升为可训练参数来应对 |
| 迹传播 | Trace Propagation (TP) | トレース伝播 | — | Pes et al. 2025 的方法；把资格痕迹存储从按突触的 O(N²) 降到 O(N)。**不译作「痕迹传播」**，与「资格痕迹」区分 |
| 成功偏移 | success-signal offset | 成功信号のオフセット | — | R-STDP 的失败模式：成功信号的平均值偏离零。§七 第一阶段要求 < 10%σR，测量见 `research/rstdp/measure_bias.py` |
| 显存峰值 | peak GPU memory | GPU メモリのピーク | — | §12.4 要求 GPU 实验记录；§6.2 把 8GB 列为硬约束，峰值是判断有无踩线的依据 |

## 三、神经科学与模型

| 中文 | English | 日本語 | 频次 | 备注 |
| :--- | :--- | :--- | ---: | :--- |
| 时间注意力引导融合 | temporal-attention-guided fusion (TAAF) | 時間注意誘導融合（TAAF） | — | 计划书 §2.2；第二阶段任务，当前默认策略**不是**它 |
| 平凡拼接 | plain concatenation | 単純連結 | — | 脉冲总线当前的默认融合策略 |
| 群体脉冲 | population of spikes | 集団スパイク | — | 与「群体水平编码 / 集団レベル符号化」区分 |
| 不确定性监控 | uncertainty monitoring | 不確実性モニタリング | — | 元认知门控的输入 |
| 动作生成 | action generation | 行動生成 | — | 执行层 |
| 模态解码器 | modality decoder | モダリティデコーダ | — | 执行层 |
| 脉冲神经网络 | spiking neural network | スパイキングニューラルネットワーク | — | 首次出现可附缩写 SNN |
| 神经元 | neuron | ニューロン | 12 | |
| 突触 | synapse | シナプス | 5 | |
| 脉冲 | spike | スパイク | 28 | |
| 发放率 | firing rate | 発火率 | 1 | |
| 膜电位 | membrane potential | 膜電位 | 1 | |
| 时间常数 | time constant | 時定数 | 1 | |
| 奖励预测 | reward prediction | 報酬予測 | 4 | 奖励预测 Critic |
| 稀疏随机投影 | sparse random projection | スパースランダム射影 | 4 | |
| 差分编码 | difference encoding | 差分符号化 | 1 | 图像插件 |
| 编码 | encode / encoding | 符号化 | — | `ModalityPlugin.encode`；「差分编码」「群体水平编码」见表内 |
| 事件相机 | event camera | イベントカメラ | 1 | DVS |
| 耳蜗模型 | cochlear model | 蝸牛モデル | 1 | |
| 时间网格 | time grid | 時間グリッド | 1 | 模态对齐 |
| 时间对齐 | time alignment | 時間整列 | — | 脉冲总线流程中的一步；README 日文版架构图作「時間グリッド整列」 |
| 显存 | GPU memory | GPU メモリ | 2 | |
| 量化 | quantization | 量子化 | 3 | INT8 痕迹量化 |
| 量化（数値化の意） | quantify | 定量化 | — | 「量化差距报告」= ギャップの定量報告。**与同表「量化 / 量子化」（INT8 痕迹量化）语义不同** |
| 分块训练 | chunked training | 分割学習 | 1 | |
| 持续学习 | continual learning | 継続学習 | 1 | |
| 检索 | retrieval | 検索 | 6 | 跨模态检索 |
| 检索（文献） | search | 文献検索 | 6 | 指文献检索（"三个独立检索角度"）= search，**不要用 retrieval**——retrieval 留给上一条的跨模态检索 |
| 痕迹 | trace | 痕跡 | 4 | 资格痕迹 = eligibility trace；INT8 痕迹量化 = INT8 trace quantization |
| 时间注意力 | temporal attention | 時間アテンション | 3 | TAAF 时间注意力引导融合 = TAAF temporal-attention-guided fusion（计划书 §2.2，第二阶段研究任务） |
| 群体水平编码 | population-level encoding | 集団レベル符号化 | 1 | 插件开发指南的 `LevelEncoder` 示例 |
| 核化 | kernelized | カーネル化 | 16 | 核化 IB-Hebbian 感知层 |
| 除法归一化 | divisive normalization | 除法正規化 | 11 | 感知层；神经科学概念，**与 batch normalization 无关**，不要省称 normalization |
| 自适应阈值 | adaptive threshold | 適応閾値 | 4 | ALIF 的阈值自适应 |
| 三因子 | three-factor | 三因子 | — | 三因子 Hebbian 规则（pre×post → κ 滤波 → 乘标量 δ） |
| 表征瓶颈 | representation bottleneck | 表現ボトルネック | — | 深度网络的表征瓶颈；「避免了」≠「不遭受」，见 `docs/references.md` [4] |
| 信用分配 | credit assignment | 信用割当 | — | TD 误差沿时间的信用分配 |
| 时间尺度 | time scale | 時間スケール | 3 | `temporal_scale`；与「时间常数 / 時定数」区分 |
| 抑制性输入 | inhibitory input | 抑制性入力 | 1 | 稀疏随机投影带 ±1 权重，负值表示的就是它 |
| 反向传播 | back-propagation | 逆伝播 | 3 | 误差向量的反向传播；见「全局」条 |
| 刺激特异性奖励预测 | stimulus-specific reward prediction | 刺激特異的報酬予測 | 3 | 纠正 R-STDP 无监督偏差所需的东西 |
| 正交类中心 | orthogonal class center | 直交クラス中心 | 1 | PS-SNN 的归因数字之一，见 `docs/references.md` [7] |
| 增量准确率 | incremental accuracy | 増分精度 | 1 | PS-SNN 的 76.42%，类增量学习场景 |
| 神经元动力学 | neuron dynamics | ニューロンダイナミクス | — | 骨架库不做神经元动力学（ADR-0002）；认知核心与 SpikingJelly 的事 |
| 核化信息瓶颈 | kernelized information bottleneck | カーネル化情報ボトルネック | — | Pogodin & Latham 2020 的方法；缩写 KB 不用。见「核化」条 |
| 局部目标 | local objective | 局所目的 | — | 每层各自最小化的目标函数（这里是 pHSIC 瓶颈目标），与全局损失相对。见 `research/ib_hebbian/layers.py` |
| 教学信号 | teaching signal | 教学信号 | — | 三因子规则里来自标签的那一项；类别均衡时是二值的（同类 1、异类 −1/(n−1)）。**不译作 teacher signal** |
| 分组 | grouping | グループ化 | — | 把一层神经元分成 c_k 组，各组算自己的方差。与「分组数」区分——后者是组的个数 |
| 分组数 | number of groups | グループ数 | — | c_k；论文 Table 3 的 MNIST 行给 16 或 32 |
| 平滑偏移 | smoothing offset | 平滑オフセット | — | 分组方差里的 δ，防止方差为零时除零；论文 D.5 取 1 |
| 瓶颈平衡参数 | bottleneck balance parameter | ボトルネック平衡パラメータ | — | 目标里的 γ，平衡 pHSIC(Z,Z) 与 pHSIC(Y,Z)；论文取 2 |
| 读出 | readout | 読み出し | — | 接在隐藏层之后的线性分类器。本项目里它用交叉熵训练，与隐藏层的局部规则不同 |
| 弱消融 | weak ablation | 弱いアブレーション | — | 只说明某一项在起作用、不足以构成对原论文断言的独立验证的那种消融。README 里明确标注 |

## 四、不翻译的内容

以下是**刻意保留原样**的，翻译它们反而是错的：

| 类别 | 例子 | 理由 |
| :--- | :--- | :--- |
| 代码标识符 | `ModalityPlugin`、`SpikeBus`、`SpikeTrain`、`discover_plugins` | 翻译后代码跑不通、读者搜不到 |
| 文件名与路径 | `packages/biosnn-bus/`、`docs/plugin-guide.md` | 实际路径 |
| 命令与配置 | `uv sync --locked`、`pip install`、`[project.entry-points."biosnn_bus.plugins"]` | 照抄才能执行 |
| 书目条目 | 作者名、论文标题、期刊名、DOI | 学术引用的规范：保持原文 |
| 规格与许可证名 | Apache-2.0、CC-BY 4.0、Contributor Covenant、semver、PEP 621 | 专有名词 |
| 枚举值与状态字面量 | `temporal`、`semantic`、`verified`、`partial`、`metadata-error` | 代码里的字面值 |
| 环境与平台名 | PyPI、Colab、GitHub Actions、RTX 5060、Loihi 2 | 专有名词 |
| 规则与算法名 | TD-LTP、TD-STDP、R-STDP、e-prop、actor-critic | 原文命名；翻译后读者无法检索文献。**actor-critic 在日文版中也保留英文原样**，与 `docs/references.ja.md` 一致 |

## 五、译文语气约定

- **英文**：技术写作的常规语气，第二人称直呼读者（"you"），不用 "we" 指代读者。
  避免 marketing 措辞（"powerful"、"seamless"、"cutting-edge"）。
- **日文**：です・ます 体，技术文档的常规文体。避免过度敬语；
  术语用片假名（上表所列），不用汉字音读生造词。
- **共同**：中文原文里的**诚实边界表述**（如"尚未发布到 PyPI"、"这些检查是空转"）
  在译文里必须**同样明确**，不得弱化或美化。这是项目原则，不是措辞偏好。
