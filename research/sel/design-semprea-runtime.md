# Semprea 运行层：单端口 OpenAI 兼容服务（B1–B4 规格）

**状态：规格落盘，实现待做。** 对应 [`plan-semprea-v1-optimization.md`](plan-semprea-v1-optimization.md) 的 B1–B4。

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
