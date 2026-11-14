# persona-core

**通用**的人格与记忆核。不 import 任何宿主程序（不依赖 axo），只用标准库。

融合模型的能力分两半：

| 半边 | 在哪 | 能不能随包分发 |
| :--- | :--- | :--- |
| 通用能力（常识 / 伦理 / 心理学） | **权重里** | 能 |
| 这个用户的记忆与人格 | **权重外**（本地文件） | **不能** |

所以包必须带一条"把记忆那半边建起来"的命令，否则用户拿到的只是一半。

## 用法

```bash
python -m persona_core init-memory --root ~/persona-core
python -m persona_core manifest   --root ~/persona-core
```

`init-memory` 建出：

```
<root>/memory/mem.db        记忆库（schema 与 memcore 同源）
<root>/identity/instance.json  实例 id、诞生日期、起点年龄
```

并在库里写一条**钉住的内置人格锚**（`kind=persona_anchor`、`source_ref=persona://builtin`）。
**幂等**：重复跑不会重复写。

`manifest` 把"她由哪些文件组成"写成 `<root>/MANIFEST.json`（逐件 sha256），供打包与核对。

## 通用性约束

- 人格正文来自包内 `persona.md`，宿主程序可覆盖（`--persona`）；
- 数据根由 `--root` 给定，**没有任何写死的产品路径**；
- 记忆库表结构与 `memcore` 同源，宿主装了 `memcore` 就能直接读写同一个库。
