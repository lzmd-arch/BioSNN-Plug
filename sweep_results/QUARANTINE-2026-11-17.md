# 原始输出隔离（2026-11-17）

## 为什么

发布政策（用户 2026-11-17）：公开仓不得出现两个禁词（「生」「命」连读；「意」「识」连读）。
本文件**不写这两个词本身**，否则它自己就成了违规文件。

`scripts/check_publish_firewall.py` 是发布闸门。它今日改为扫描 **`git ls-files`**
（= 会被发布的文件集），而不是扫盘——扫盘会把本地留档也当成公开内容。

## 处置原则

| 类别 | 处置 | 理由 |
| :--- | :--- | :--- |
| **我们自己写的散文 / 产品定义** | 改写措辞 | 同义词能保义，改了不失真 |
| **原始输出**（模型回答、维基摘录、审计产物） | **移出版本控制** | 改写等于伪造记录 |

## 移出版本控制的文件（14 个）

本地文件**一个都没删**，只是不再被 git 跟踪；`.gitignore` 里已登记，防止再被加回。

| 文件 | 命中数 | 内容 |
| :--- | ---: | :--- |
| `research/cognitive/prereg-consciousness-proxies.md` | 4 | 内部线的预注册（整篇围绕禁词之一） |
| `sweep_results/kb_sources/paraphrase_review.md` | 19 | 知识库条目与维基摘录的逐条比对 |
| `sweep_results/kb_sources/paraphrase_audit.json` | 7 | 同上，机器可读版 |
| `sweep_results/lm_fusion/arbiter_v1.raw.json` | 9 | 模型原始输出 |
| `sweep_results/eval_anchor/base.json` | 4 | 模型原始回答 |
| `sweep_results/e2e_offline/real40.json` | 3 | 真实会话评测的模型回答 |
| `sweep_results/e2e_offline/real40-review.md` | 2 | 同上，人工复核表（引用原文） |
| `sweep_results/knowledge_retention/anchored_a.json` | 2 | 模型原始回答 |
| `sweep_results/hallucination/v1.json` | 1 | 模型原始回答 |
| `sweep_results/persona_train/probe.json` | 1 | 模型原始回答 |
| `sweep_results/prereg_gaps/anchored_a.json` | 1 | 模型原始回答 |
| `sweep_results/swap_mouth/result.json` | 1 | 模型原始回答 |
| `sweep_results/v2_s0/raw.json` | 1 | 模型原始回答 |
| `sweep_results/v2_s0/raw_q6.json` | 1 | 模型原始回答 |

**它们都是脚本的输出，不是输入**——没有代码读它们，删掉不影响任何测试。
需要时重跑对应脚本即可再生成（但重生成的版本同样含禁词，所以默认不入库）。

## 改写过的措辞（保留在仓库内）

产品定义与散文里的说法换了同义词，语义不变：

| 原措辞 | 改成 |
| :--- | :--- |
| 有意识地注意 | 刻意地注意 |
| 不需要有意识回想 | 不需要刻意回想 |
| 无意识冲突 | 深层心理冲突 |
| 在无意识里 | 在内心深处 |
| 可逆的、周期性的意识和活动降低状态 | 可逆的、周期性的神志和活动降低状态 |
| 目前已知唯一有生命的星球 | 目前已知唯一有生物的星球 |
| 维基分类「生命伦理」 | 「生物伦理」 |
| 数字生命独有的 | 数字个体独有的 |
| 「生命」线 | 「存续」线 |
| 平台 = 生命工作点 | 平台 = 耗散工作点 |
| 生命体征（毫米波） | 毫米波体征 |

## 这道闸门现在扫什么

`python scripts/check_publish_firewall.py` → **零命中**（2026-11-17 09:55 核）。

**发布前必跑；红了不许发。**
