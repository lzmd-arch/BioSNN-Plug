# ADR-0006：`ModalityPlugin` 保持计划书 §2.3 的原始签名

- **状态**：已采纳
- **日期**：2026-09-21
- **相关**：计划书 §2.3 模态插件接口规范；[ADR-0002](ADR-0002-numpy-core-torch-optional.md)

## Context

计划书 §2.3 给出了 `ModalityPlugin` 的完整接口定义，包括三方法两属性、以及两个
带默认值的属性。这份规范是整个插件化设计的地基，也是计划书里少数写得足够具体、
可以直接照抄的部分。

但 [ADR-0002](ADR-0002-numpy-core-torch-optional.md) 决定骨架库核心不依赖 torch，
而计划书的规范里有这么一行：

```python no-run
@abstractmethod
def get_membrane(self) -> nn.Module: ...
```

`nn.Module` 是 torch 的类型。照抄它，骨架库就必须 `import torch`。

于是有了一个真实的取舍：**是改接口，还是改类型标注？**

## Decision

**签名一字不改，只把返回类型换成结构化协议。**

五个成员全部按计划书原文实现：

| 成员 | 计划书原文 | 骨架库实现 |
| :--- | :--- | :--- |
| `encode(raw_input)` | 抽象方法 | 一致 |
| `get_membrane()` | 抽象方法，返回 `nn.Module` | 返回 `Membrane` 协议（**唯一偏离**） |
| `decode(spike_output)` | 抽象方法 | 一致 |
| `modality_name` | 抽象属性 | 一致 |
| `spike_dim` | 抽象属性 | 一致 |
| `temporal_scale` | 默认 `10.0` | 一致 |
| `fusion_channel` | 默认 `'temporal'` | 一致 |

`Membrane` 是一个 `runtime_checkable` 的 `Protocol`，只要求存在 `forward` 方法：

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Membrane(Protocol):
    def forward(self, x): ...
```

认知核心返回的 `nn.Module` 子类**结构上自动满足**这个协议——`nn.Module` 本来就
要求子类实现 `forward`。所以对认知核心和插件作者而言，写法完全没变。

另外新增两个计划书里没有的成员，都是纯粹的加法、不影响兼容：

- `channel` 属性：`fusion_channel` 的枚举形式，供总线内部使用。它同时承担校验职责，
  `fusion_channel` 返回非法字符串时在这里报错；
- `validate()` 方法：自检接口契约。`SpikeBus.register()` 会调用它。

## Consequences

**好处**

- 计划书的接口规范仍然是对外契约，读计划书和读代码看到的是同一套东西；
- 插件作者按计划书写代码就能跑，不需要理解 Protocol 与 `nn.Module` 的关系；
- 骨架库保持零 torch 依赖。

**代价（已接受）**

- **类型检查变弱**：`Membrane` 是结构化协议，`isinstance(x, Membrane)` 只检查有没有
  `forward` 属性，抓不到签名不匹配（比如 `forward(self)` 少一个参数）。这是 Python
  Protocol 的固有限制，不是本决定引入的；
- 严格来说这是**对计划书的一处偏离**，虽然很小。读代码的人看到 `Membrane`
  而不是 `nn.Module` 时，应能在这里找到原因，而不是以为是抄漏了。

**为什么校验放在 `validate()` 而不是 `__init_subclass__`**

计划书相关的讨论里曾设想在 `__init_subclass__` 里校验 `spike_dim` 等约束。这做不到：
它们是**实例属性**，类定义期还没有值可查。真正的时机有两个——实例化时、或注册时。
选了注册时（`SpikeBus.register` 调用 `validate()`），因为：

- ABC 不该强制子类实现 `__init__`（很多插件不需要）；
- 注册是插件进入系统的唯一入口，在这里拦截覆盖面最广、错误信息也最有上下文
  （能带上"是哪个总线拒绝的"）。

## Alternatives

**A. 严格照抄计划书，`get_membrane() -> nn.Module`。**
否决。直接违反 [ADR-0002](ADR-0002-numpy-core-torch-optional.md)，进而违反第零阶段
"Colab 无 GPU 可跑"的硬性成功标准。

**B. 删掉 `get_membrane`，让认知核心自己管膜电位。**
否决。这是对计划书接口的**实质性**修改，不是类型层面的调整。`get_membrane` 是插件
向认知核心暴露"我这个模态的神经元长什么样"的唯一途径；删掉它，认知核心就得反过来
知道每个模态的内部结构，插件化的边界就破了。

**C. 把返回类型标注为 `Any`。**
否决。看起来最省事，实际最差：类型检查器完全失去作用，读者也不知道这个返回值该
满足什么。Protocol 至少把契约写下来了，而且是可执行的文档。

**D. 定义自己的 `Membrane` 抽象基类（ABC），要求插件继承。**
否决。那样 `nn.Module` 子类就**不能**直接返回了，认知核心必须写适配层。
结构化协议的好处正在于不需要显式继承。
