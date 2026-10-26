# 真实记忆 50k 触发器

目标是：当 `D:\AxoData\memory\mem.db` 的真实记忆达到 **50,000** 条时，跑一次
真实库召回基准，结果落到：

```
D:\Projects\BioSNN-Plug-neuron\sweep_results\axo_real_50k\real_memory.json
```

## 组成

- `axo_real_memory_50k.py`：只读复制真实库（SQLite 在线备份），不够阈值就直接
  退出；够了才建索引、抽样 100 条做 30% 字符丢字查询，跑词法与语义两条路径，
  并抽 10 条做候选 vs 全量对照。只写聚合数字，不写正文/查询/id。
- `run_real_50k_trigger.ps1`：给计划任务用的包装器；用 Axo 的 venv 跑（语义路径
  需要 httpx）。日志在 `D:\Projects\_scratch_axo\real_50k_trigger.log`。
- `--require-semantic`：语义不可用时退出 2 且**不写最终结果**，次日重试；避免
  因为 Ollama 没起来留下一个只有词法的“完成”结果。

## 手动检查

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Projects\BioSNN-Plug-neuron\scripts\run_real_50k_trigger.ps1"
```

库不足 50k 时会打印 `not_ready: N < 50000` 并退出 0。当前真库 193 条。

## 注册每日任务（需要你自己执行）

```powershell
schtasks /Create /SC DAILY /ST 09:37 /TN "AxoReal50kBenchmark" ^
  /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"D:\Projects\BioSNN-Plug-neuron\scripts\run_real_50k_trigger.ps1\"" /F
```

或直接在「任务计划程序」里新建一个每日 09:37 的任务，操作同上。任务只会在用户
登录时运行；运行前请确保本地 Ollama 在跑（`nomic-embed-text` 可用）。

查询 / 删除：

```powershell
schtasks /Query /TN "AxoReal50kBenchmark"
schtasks /Delete /TN "AxoReal50kBenchmark" /F
```

## 重复运行

- 结果文件已存在且 `count >= threshold` 时，脚本直接跳过；这是“达到 50k 跑一次”
  的语义。
- 要手动重跑：加 `--force`，并指定一个临时 `--out`，避免覆盖首次结果。
