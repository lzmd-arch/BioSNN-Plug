# ADR-0002：骨架库核心基于 numpy，torch 作为可选桥接

- **状态**：已采纳
- **日期**：2026-09-21
- **相关**：计划书 §七 第零阶段成功标准、§1.4 技术约束；[ADR-0001](ADR-0001-skeleton-as-separate-library.md)

## Context

计划书 §七 第零阶段给骨架库定的硬性成功标准里有这一条：

> 演示在 Colab **无 GPU** 环境可跑。

而计划书 §1.4 与 §八 技术选型表把整个项目的技术栈定为 **SpikingJelly + PyTorch +
eprop-PyTorch**。认知核心毫无疑问要用 torch。

问题在于骨架库要不要跟着用。

`torch` 的 wheel 体积约 2GB（CUDA 版）。如果 `pip install biosnn-bus` 会把它拖进来，
那么：

- 一个只想写个模态插件的人要下 2GB；
- Colab 里 `pip install` 要等好几分钟，"5 分钟跑通演示"直接破产；
- 骨架库与"独立轻量库"的定位自相矛盾。

还有一个更隐蔽的问题：计划书 §2.3 的接口规范里写着

```python no-run
@abstractmethod
def get_membrane(self) -> nn.Module: ...
```

`nn.Module` 是 torch 的类型。只要签名里写它，骨架库就必须 `import torch`，
依赖就绕不过去了。

## Decision

**骨架库核心只依赖 numpy。**

1. `SpikeTrain` 内部存 numpy 数组，与后端无关；torch 张量通过
   `SpikeTrain.to_torch()` / `SpikeTrain.from_torch()` 桥接，实现在可选 extra
   `biosnn-bus[torch]` 下，延迟 import。
2. `get_membrane` 的返回类型从 `nn.Module` 改为结构化协议
   [`Membrane`](../../packages/biosnn-bus/src/biosnn_bus/types.py)——只要求存在
   `forward` 方法。认知核心返回的 `nn.Module` 子类**结构上自动满足**该协议，
   插件作者的写法不受影响。详见 [ADR-0006](ADR-0006-plugin-interface-fidelity.md)。
3. CI 里显式断言：在干净 venv 里装完 wheel 后，`torch` 必须**不存在**。

## Consequences

**好处**

- `pip install biosnn-bus` 只拉 numpy，秒级完成；
- Colab 演示无需 GPU、无需等待，第零阶段的硬性标准直接满足；
- 骨架库可以在任何 Python 环境里测试，CI 矩阵跑 3.10/3.11/3.12 毫无压力；
- 反过来逼出一个好结果：接口**必须**与具体张量库解耦。将来若要换 JAX 或直接对接
  Loihi/Lava，接口不用改。

**代价（已接受）**

- 跨库传数据时要显式转换（`.to_torch()`），比直接传张量多一步；
- `Membrane` 协议是结构化类型，`isinstance` 检查只看有没有 `forward`，抓不到
  签名不匹配。这是 Python Protocol 的固有限制；
- 总线里的数值运算在 numpy 上跑，比 torch 慢。骨架阶段的数据量小，可以接受；
  真成为瓶颈时，认知核心侧可以直接绕过 `SpikeBus` 拿原始 `SpikeTrain`。

**明确的非目标**

- 骨架库**不是**一个轻量 SNN 仿真器。它不做神经元动力学、不做学习规则。那些是
  认知核心与 SpikingJelly 的事。

## Alternatives

**A. 核心依赖 torch。**
否决。直接违反第零阶段的硬性成功标准，且 2GB 的安装体积与"独立轻量库"的定位冲突。

**B. 核心不依赖任何数组库，用纯 Python list。**
否决。脉冲数据是 `(T, N)` 的稠密数组，纯 Python 会慢到无法做演示，而且 numpy 本身
只有十几 MB，不是负担。

**C. 用 `array-api-compat` 做后端无关抽象。**
否决（现阶段）。它解决的是"同时支持 numpy/torch/jax"的问题，而我们只需要
"核心用 numpy，边界上桥接 torch"。多一层抽象换不来实际收益，等真需要多后端时
再引入不迟。
