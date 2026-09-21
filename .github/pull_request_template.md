## 这个 PR 做了什么

<!-- 一两句话说清。如果是修 bug，说清修的是什么症状。 -->

## 关联 issue

<!-- 例如 Closes #12 -->

## 提交前核查清单

计划书 §12.5 要求 PR 前做"引用点验 + 代码块检查"。请逐项确认：

- [ ] `uv run pre-commit run --all-files` 通过
- [ ] `uv run pytest -q` 通过
- [ ] 新增/修改的文档代码块**能跑通**（`scripts/check_doc_code_blocks.py` 会真的执行它们）
      ——无法独立运行的片段，请在围栏上标注 `python no-run`
- [ ] 如果改了 `examples/*.py`，已重跑 `python scripts/build_notebook.py` 更新 notebook
      （pre-commit 钩子会自动做，确认它改动的文件已一并提交）
- [ ] 如果新增了外部引用，已加入 `docs/references.md` 并标明核验状态
- [ ] 如果这是一个架构层面的取舍，已新增一篇 `docs/adr/`

## 改动类型

- [ ] Bug 修复（不破坏现有 API）
- [ ] 新功能（不破坏现有 API）
- [ ] 破坏性变更（**仅在 `research/` 下的研究代码，或骨架库大版本号内**）
- [ ] 文档 / 工具链 / CI

> 注意：`biosnn-bus` 遵循 semver，破坏性变更必须升大版本号。
> `research/` 下的研究代码在头 12 个月内明确标注 API 不稳定（计划书 §12.5）。

## 是否涉及 GPU / 大规模实验

<!-- 如果涉及，请说明显存占用与运行时长。目标硬件是 RTX 5060 8GB。 -->

## 备注

<!-- 有什么需要 reviewer 特别注意的地方？ -->
