# Semprea 运行层：单端口 OpenAI 兼容服务（B1–B4 规格）

**状态：已实现并实测（2026-11-14）。** 代码在 `D:\Projects\Semprea-4.5Buntime\`（`semprea/` 运行层 + 26 条守卫测试），
对应 [`plan-semprea-v1-optimization.md`](plan-semprea-v1-optimization.md) 的 B1/B2/B4/B6。实现与规格的偏差见文末 §七。

## 为什么要这一层

现在的包要求使用者自己拼两个端点（慢半 `:8081` + 快半 `:18734`），并且假设调用方是 Axo。
**通用部署必须对外只有一个地址、一套 OpenAI 语义**——Axo 只是众多消费者之一。

## 一、对外接口（唯一的 port，默认 18740）

| 路径 | 语义 | 实现 |
| :--- | :--- | :--- |
| `POST /v1/chat/completions` | OpenAI 聊天补全 | 慢半 + 人格 system 消息；α>0 时叠加快半 |
| `POST /v1/completions` | OpenAI 文本补全 | 同上，但不加人格 |
| `POST /v1/embeddings` | OpenAI 嵌入 | 转发到快半 `/encode` |
| `GET  /health` | 能力与档位 | 见 §四 |

**三条契约**（与既有验收一致）：

1. `alpha = 0` 时，响应必须与**慢半单独**逐位相同（T-3）；
2. 任何降级都要在 `/health` 里可见，**不许"看起来能用"**；
3. 请求/响应字段与 OpenAI 官方一致，不发明私有字段（私有信息放 `x-semprea-*` 头）。

## 二、慢半后端适配链（B2）

按顺序探测，第一个可用的胜出：

| 顺序 | 后端 | 探测方式 |
| ---: | :--- | :--- |
| 1 | 包内 `bin/llama-server` | 文件存在且 `--version` 可跑 |
| 2 | 系统 `llama-server` | `shutil.which` |
| 3 | Ollama | `GET 127.0.0.1:11434/api/tags` 有模型 |
| 4 | 任意 OpenAI 兼容端点 | 配置里给了 `slow.base_url` |
| — | 全不可用 | **退化为"只有快半"**，`chat: false` |

**配置**（`semprea.toml` 或环境变量，环境变量优先）：

```toml
[slow]
backend = "auto"          # auto | llama-server | ollama | openai
base_url = ""             # backend=openai 时必填
model = "Qwen3-4B-Q4_K_M.gguf"

[fast]
ckpt = "fast/semprea_v1_ckpt.pt"
device = "auto"           # auto | cuda | cpu

[fusion]
alpha = 0.03

[server]
port = 18740
```

## 三、降级档位（B5/B6）

| 档 | 条件 | 能力 |
| :--- | :--- | :--- |
| **full** | 慢半可用 + 快半在 GPU | chat + 融合 + embeddings |
| **full-cpu** | 慢半可用 + 快半只能 CPU | 同上，`/complete` 慢 10× |
| **chat-only** | 只有慢半 | chat，`alpha` 强制 0 |
| **fast-only** | 只有快半 | 仅 `/v1/embeddings` 与候选补全，`chat: false` |
| **offline** | 都没有 | `503` + 指引；**Host 应自行退到检索直出** |

**启动顺序**：先监听端口并报 `fast-only`/`offline`，模型在后台加载完成后**热切**到更高档——
**用户不该对着 20 秒的白屏等待**。

## 四、`/health` 的形状

```json
{
  "semprea": "1.0",
  "tier": "full",
  "chat": true,
  "embeddings": true,
  "fusion": {"alpha": 0.03, "active": true},
  "slow": {"backend": "llama-server", "model": "Qwen3-4B-Q4_K_M", "ready": true},
  "fast": {"device": "cuda", "step": 100000, "ready": true},
  "persona": {"name": "千岁（阿螈）", "version": 5}
}
```

## 五、实现顺序（建议）

1. `semprea/serve.py`：FastAPI 或 `http.server`（与现有一致，零新依赖）+ 四个路由；
2. 后端适配器 `semprea/backends.py`：四个探测函数 + 统一的 `complete(prompt) -> logprobs`；
3. 融合接线：`alpha>0` 时取慢半 top-20 + 快半 top-10，做前缀归并（复用 `fusion/semprea_fuse.py`）；
4. `/health` 与档位状态机；
5. 守卫：`alpha=0` 时响应逐位等于慢半（T-3 的运行时版本）。

## 六、不做什么

- **不引入 web 框架依赖**：`http.server` 够用，包越小越好；
- **不做鉴权**：只监听 `127.0.0.1`；
- **不做多租户/并发调度**：单用户本机场景，够用即可。

## 七、实现记录（2026-11-14）

**落地位置**：`D:/Projects/Semprea-4.5B/runtime/`

| 文件 | 内容 |
| :--- | :--- |
| `semprea/serve.py` | 四个路由 + `/v1/models` + 档位状态机 + 逐 token 融合解码 |
| `semprea/backends.py` | 四条探测 + 统一 `SlowClient` + 由本进程拉起 llama-server |
| `semprea/config.py` | `semprea.toml` + 环境变量（环境变量优先） |
| `tests/test_runtime.py` | **26 条守卫**，含 α=0 逐字节与五档降级 |

### 与规格的三处偏差（都是实测逼出来的）

1. **`α=0` 用「原样转发字节」，不是「转发 JSON」。** 规格只写"逐位相同"；解析再序列化
   会改键序与转义（`你好` → `\u4f60\u597d`）。已用**变异测试**证明这条守卫会变红：
   把转发路径换成 `json.dumps(json.loads(raw))` 后，`TestAlphaZeroIsByteIdentical` 立刻失败。
2. **`backend` 显式指定时只试那一条。** 规格的顺序表是 auto 模式的；若配了 `openai` 却
   静默退到 Ollama，"配错了"会看起来像"能用"。守卫：`TestBackendChain` 两条。
3. **人设只能取 `persona.md` 的 ```text 围栏**。整篇塞进去是 702 字 ≈ 391 token，
   在 `-c 512 --parallel 4` 的 llama-server 上直接 400（`exceeds context size`）。
   这是实现期真实撞到的，不是预防性设计。

### 新发现：两半同卡时逐 token 融合会慢 30 倍

融合每一步都要在**两个 CUDA 进程之间来回切**。显存吃紧时（实测 7.6 / 8.1 GB），
切一次约 **1.5 秒**，两个服务一样慢：

| 调用方式 | 慢半 | 快半 |
| :--- | ---: | ---: |
| 单个服务连打 10 次（不交替） | — | **0.02–0.03 s** |
| 慢半、快半交替，**两半都在 GPU** | **1.5 s** | **1.5 s** |
| 慢半 GPU + **快半 CPU** 交替 | **0.014 s** | **0.02–0.035 s** |

⇒ **快半放 CPU 是默认部署形态。** 同一台机器上一个来回从 1.5 秒降到 0.04 秒。
`/health` 此时报 `full-cpu` 而不是 `full`——**这是如实报告，不是降级失败**。

### 实测延迟（RTX 5060 8 GB，Qwen3-4B-Q4_K_M，快半 step100000）

| 项 | 数 |
| :--- | :--- |
| `α=0` 转发 | ~60 tok/s（llama-server 原生速度） |
| **融合解码 `α=0.03`** | **0.45 s/token（≈2.2 tok/s）** |
| 慢半 `next_logprobs` 单步（前缀缓存命中） | 0.022 s |
| 快半 `/complete`，4 字上下文 | 0.022–0.03 s |
| 快半 `/complete`，220 字上下文 | 0.33–0.44 s |

**融合的全部代价落在快半**，且与上下文长度成正比（约 1.5 ms/字）。
慢半那边因为 llama-server 的前缀缓存，每步只要 0.022 s。
⇒ **A3（按键级增量状态）是融合进默认路径的唯一悬念**，不是慢半。

### 追加（同日）：包变成自足的

快半服务原来要求旁边有一个 BioSNN-Plug 检出。现在**原样 vendor 进运行层**
（`runtime/semprea/fast/`：`model.py` / `hierarchical.py` / `tree_kernel.py` /
`subset_tokenizer.py` / `serve_snn.py` 五件，唯一改动是 import 路径），
给了 `[fast] ckpt` 就由运行层把快半当子进程拉起来。

**两条硬要求，都来自实际踩到的故障：**

1. **torch 不进主层**——快半在子进程里，主层只用 HTTP；
2. **父进程死了子进程必须跟着死**。`taskkill /F` 不会触发清理，实测留下过一个占着
   端口和 2 GB 内存的孤儿。做法：stdin 给成管道，子进程看门狗阻塞在 `read()`，
   父进程一死管道关闭 ⇒ EOF ⇒ 立刻退出。**已实测**：杀掉父进程 4 秒后两个端口都空了。

**保真核对**（发布前必跑）：`runtime/tools/verify_fast_parity.py` 用同一检查点各起一次，
比 `/complete`（候选逐元素）、`/score`（差 < 1e-6）、`/encode`（最大差 < 1e-6）。
当前结果：**4 段上下文候选逐元素相同、12 组 score 一致、encode 最大差 0.00e+00**。

**启动顺序的一个坑（已修）**：`refresh()` 原先先拉快半再探慢半，而快半在 CPU 上
加载 2 GB 要 ~20 秒 ⇒ 这 20 秒里 `/health` 报 `offline`，**慢半其实 1.5 秒就好了**。
改成先探慢半（对话是主能力），启动 3 秒即报 `chat-only, chat=true`，25 秒切 `full-cpu`。

运行层现在有 **29 条守卫**（原 26 条 + 设备策略 auto=CPU、没给检查点不自拉、
检查点缺失只报错不崩）。
