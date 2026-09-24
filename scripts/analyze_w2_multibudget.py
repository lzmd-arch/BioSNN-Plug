"""把 W2 的两批（30 epoch / 100 epoch）放在一起算，供「两个预算并列写」用。

**为什么要有这个脚本，而不是看着曲线报数。**

2026-09-24，我凭眼睛看单种子探针曲线，报了「已经收敛、渐近线约 0.85」。三处问题：

1. 那是**外推**不是实测——每 20 轮的窗口均值增量是 +0.0455 → +0.0217 → +0.0114，
   在对折但**没到零**；
2. 逐 10 轮的端点增量**不单调**（`80→90` 是 +0.0018，`90→100` 是 +0.0050），
   一条曲线分不清这是趋势还是种子噪声；
3. 我把**验证**准确率（0.8475 @100）和**测试**准确率（0.8534）混成了一个「约 0.85」。

所以这个脚本**一律算，不目测**，并且：

* 用**窗口均值**而不是端点——端点会落在局部峰谷上（探针曲线第 79 轮 0.8367、第 80 轮
  0.8407、第 81 轮 0.8365，拿第 80 轮当端点，最后 20 轮的增量就凭空好看了）；
* 每个尾部增量**同时给逐种子的极差**，让「还在爬 vs 已收敛」由数据回答，不由我回答；
* 两批用的是**同一批种子（0–4）**，所以逐种子配对：同一份代码、同一个种子、只换预算。
  截断的代价因此不是单种子的一个数，而是 **5 个配对差的均值与极差**。

**口径**：本脚本报的是**配对**差（同种子），不是「5 种子均值之差」。两者按代数恒等式必然
相等（`mean(b−a) = mean(b) − mean(a)`），但**单看某一种子时不等**——seed 0 的配对差是
+7.86，而均值差是 +7.55。引用时必须说清是哪一个。

用法::

    uv run python scripts/analyze_w2_multibudget.py

产物只打印，不写文件——数字进文档之前要有人看过。
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 30 epoch 那批（已入库）。与 100 epoch 批**同一批种子**，这是配对比较的前提。
DIR_30 = REPO / "sweep_results" / "w2_seeds"
LOG_30 = REPO / "sweep_results" / "w2_seeds.log"
#: 100 epoch 那批。跑的时候落在 gitignore 的 data/ 下，归档后搬进 sweep_results/。
#: 两个候选都查，优先归档后的那份——这样归档之后脚本不用改。
CANDIDATES_100 = [
    REPO / "sweep_results" / "w2_seeds_100ep",
    REPO / "data" / "w2_seeds_100ep",
]
LOG_100 = [
    REPO / "sweep_results" / "w2_seeds_100ep.log",
    REPO / "data" / "w2_seeds_100ep.log",
]

#: 尾部趋势的统计窗口（轮）。
WINDOW = 20


def load(dirpath: Path) -> dict[int, dict]:
    """读一个批次的所有种子记录，返回 ``{seed: record}``。"""
    out: dict[int, dict] = {}
    if not dirpath.is_dir():
        return out
    for path in sorted(dirpath.glob("seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[int(record["seed"])] = record
    return out


def pick_100() -> tuple[Path | None, dict[int, dict]]:
    """找 100 epoch 批：优先归档目录，退回 data/ 暂存区。"""
    for path in CANDIDATES_100:
        records = load(path)
        if records:
            return path, records
    return None, {}


def curve_of(record: dict) -> dict[int, float]:
    return {int(p["epoch"]): float(p["val_accuracy"]) for p in record.get("curve", [])}


def sparsity_by_seed(log_path: Path) -> dict[int, float]:
    """从日志里抓每个种子的脉冲稀疏度。

    **它不在 JSON schema 里**——`sweep_seeds.py` 落盘的字段只有
    `test_accuracy` / `active_fraction` / `n_classes` / `elapsed_s` / `curve` / `config`。
    稀疏度只被 `train_sequential.py` 打印，所以只能从日志里取。
    """
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marks = [
        (m.start(), int(m.group(1))) for m in re.finditer(r"^种子\s+(\d+)\s*$", text, flags=re.M)
    ]
    out: dict[int, float] = {}
    for index, (pos, seed) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        found = re.findall(r"脉冲稀疏度：([0-9.]+)", text[pos:end])
        if found:
            out[seed] = float(found[-1])
    return out


def window_means(curve: dict[int, float], window: int = WINDOW) -> list[tuple[int, int, float]]:
    """把曲线切成 ``window`` 轮一段，返回 ``[(起点, 终点, 均值)]``。"""
    if not curve:
        return []
    out = []
    for start in range(1, max(curve) + 1, window):
        segment = [curve[e] for e in range(start, start + window) if e in curve]
        if segment:
            out.append((start, start + len(segment) - 1, statistics.fmean(segment)))
    return out


def _fmt(value: float | None, digits: int = 4) -> str:
    return "  (缺)" if value is None else f"{value:.{digits}f}"


def report_scale(records: dict[int, dict], sparsity: dict[int, float], label: str) -> None:
    print(f"\n{'=' * 78}\n{label}：逐种子\n{'=' * 78}")
    print(
        f"  {'种子':>4}  {'测试准确率':>10}  {'活跃比例':>9}  {'稀疏度':>8}  {'耗时 s':>8}  {'评测点':>6}"
    )
    for seed in sorted(records):
        r = records[seed]
        n_points = len(r.get("curve") or [])
        print(
            f"  {seed:>4}  {r['test_accuracy']:>10.4f}  {r['active_fraction']:>9.4f}  "
            f"{_fmt(sparsity.get(seed), 4):>8}  {r['elapsed_s']:>8.0f}  {n_points:>6}"
        )
    accs = [records[s]["test_accuracy"] for s in sorted(records)]
    acts = [records[s]["active_fraction"] for s in sorted(records)]
    print()
    for values, name in ((accs, "测试准确率"), (acts, "活跃神经元比例")):
        mean = statistics.fmean(values)
        print(
            f"  {name:<12} 均值 {mean:.4f}  最小 {min(values):.4f}  最大 {max(values):.4f}  "
            f"极差 {(max(values) - min(values)) * 100:.2f} 个百分点"
        )


def main() -> int:
    path_100, rec100 = pick_100()
    rec30 = load(DIR_30)

    if not rec100:
        print("还没找到 100 epoch 批的产物。候选目录：")
        for candidate in CANDIDATES_100:
            print(f"  - {candidate}  {'在' if candidate.is_dir() else '不在'}")
        return 1
    if not rec30:
        print(f"找不到 30 epoch 批（{DIR_30}）——配对比较做不了。")
        return 1

    log_100 = next((p for p in LOG_100 if p.exists()), LOG_100[0])
    sp30 = sparsity_by_seed(LOG_30)
    sp100 = sparsity_by_seed(log_100)

    print(f"30 epoch 批：{DIR_30.relative_to(REPO)}（{len(rec30)} 个种子）")
    print(f"100 epoch 批：{path_100.relative_to(REPO)}（{len(rec100)} 个种子）")
    print(f"日志：{LOG_30.relative_to(REPO)} / {log_100.relative_to(REPO)}")

    report_scale(rec30, sp30, "30 epoch")
    report_scale(rec100, sp100, "100 epoch")

    # ── 配对比较：同一批种子，只换预算 ────────────────────────────────────────
    shared = sorted(set(rec30) & set(rec100))
    print(f"\n{'=' * 78}\n配对比较（同一份代码、同一个种子，只换 epoch 预算）\n{'=' * 78}")
    if not shared:
        print("  两批没有共同种子，配对比较做不了。")
    else:
        print(f"  {'种子':>4}  {'30 epoch':>10}  {'100 epoch':>10}  {'差（百分点）':>12}")
        deltas = []
        for seed in shared:
            a, b = rec30[seed]["test_accuracy"], rec100[seed]["test_accuracy"]
            deltas.append((b - a) * 100)
            print(f"  {seed:>4}  {a:>10.4f}  {b:>10.4f}  {(b - a) * 100:>+12.2f}")
        print()
        print(
            f"  截断的代价：均值 {statistics.fmean(deltas):+.2f} 个百分点  "
            f"最小 {min(deltas):+.2f}  最大 {max(deltas):+.2f}  "
            f"极差 {max(deltas) - min(deltas):.2f}（{len(deltas)} 对配对）"
        )
        mean_30 = statistics.fmean([rec30[s]["test_accuracy"] for s in shared])
        mean_100 = statistics.fmean([rec100[s]["test_accuracy"] for s in shared])
        print(
            f"  同口径的 5 种子均值：{mean_30:.4f} → {mean_100:.4f}"
            f"（差 {mean_100 - mean_30:+.4f} = 配对均值差的另一种算法，必然相等）"
        )
        print(
            "  ⚠️ 但**逐种子**看两者是不同的量：seed 0 的配对差与上面那个均值不等。"
            "引用时要说清是「配对差」还是「均值差」。"
        )

        # 稀疏度：方向一致比幅度重要。
        pair_sp = [(s, (sp100[s] - sp30[s]) * 100) for s in shared if s in sp30 and s in sp100]
        if pair_sp:
            print("\n  脉冲稀疏度（同一批配对）：")
            for seed, delta in pair_sp:
                print(f"    种子 {seed:>3}：{sp30[seed]:.4f} → {sp100[seed]:.4f}  {delta:+.2f} pp")
            print(
                f"  5 种子均值 {statistics.fmean(sp30.values()):.4f} → "
                f"{statistics.fmean(sp100.values()):.4f}；"
                f"全部 {sum(1 for _, d in pair_sp if d < 0)}/{len(pair_sp)} 个种子下降"
                "（越训越不稀疏，离 §九 第二阶段 > 0.90 更远）"
            )

    # ── 尾部趋势：窗口均值 + 逐种子极差 ───────────────────────────────────────
    curves100 = {seed: curve_of(rec100[seed]) for seed in sorted(rec100)}
    length = max((len(c) for c in curves100.values()), default=0)
    print(f"\n{'=' * 78}\n尾部趋势（{WINDOW} 轮窗口的均值；跨种子均值曲线）\n{'=' * 78}")
    full = {seed: c for seed, c in curves100.items() if len(c) == length}
    if len(full) < len(curves100):
        print(f"  ⚠️ 只有 {len(full)}/{len(curves100)} 个种子的曲线等长，按等长的那些算。")
    if not full:
        print("  没有可用的曲线。")
        return 0

    mean_curve = {e: statistics.fmean([full[s][e] for s in full]) for e in range(1, length + 1)}
    windows = window_means(mean_curve)
    print(f"  {'窗口':>10}  {'均值':>8}  {'比上一窗口':>12}")
    for index, (start, end, value) in enumerate(windows):
        gain = "" if index == 0 else f"{value - windows[index - 1][2]:+.4f}"
        print(f"  {start:>4}-{end:<5}  {value:>8.4f}  {gain:>12}")

    # 每个种子的尾部窗口增量——「还在爬 vs 已收敛」的判据在极差里，不在一根曲线上。
    print(f"\n  逐种子的尾部窗口增量（最后两个 {WINDOW} 轮窗口）：")
    per_seed_tail = []
    for seed in sorted(full):
        ws = window_means(full[seed])
        if len(ws) >= 2:
            prev = ws[-2][2] - ws[-3][2] if len(ws) >= 3 else None
            per_seed_tail.append(ws[-1][2] - ws[-2][2])
            prev_text = "" if prev is None else f"   再上一窗口 {prev:+.4f}"
            print(f"    种子 {seed:>3}：最后窗口 {ws[-1][2] - ws[-2][2]:+.4f}{prev_text}")
    if per_seed_tail:
        print(
            f"\n  最后窗口增量的跨种子极差：最小 {min(per_seed_tail):+.4f}  "
            f"最大 {max(per_seed_tail):+.4f}  均值 {statistics.fmean(per_seed_tail):+.4f}"
        )
        if min(per_seed_tail) <= 0 <= max(per_seed_tail):
            print("  ⚠️ 增量跨零——已收敛与否，这个预算下**不能从这些数据判定**。")
        elif min(per_seed_tail) > 0:
            print("  ⚠️ 增量在所有种子上都仍为正——**仍在上升**，不能说已收敛。")

    # ── 与 30 epoch 的逐轮核对（确定性检查）──────────────────────────────────
    print(
        f"\n{'=' * 78}\n确定性核对：100 epoch 批在第 N 轮的验证准确率 vs 30 epoch 批的终值\n{'=' * 78}"
    )
    for seed in shared:
        c30 = curve_of(rec30[seed])
        c100 = curves100.get(seed, {})
        if not c30:
            continue
        last30_epoch = max(c30)
        at = c100.get(last30_epoch)
        if at is None:
            print(f"  种子 {seed:>3}：100 epoch 批没有第 {last30_epoch} 轮的评测点，跳过")
            continue
        flag = "一致" if abs(at - c30[last30_epoch]) < 1e-9 else "⚠️ 不一致"
        print(
            f"  种子 {seed:>3}：30 epoch 批终值（第 {last30_epoch} 轮）{c30[last30_epoch]:.6f}  "
            f"100 epoch 批同轮 {at:.6f}  → {flag}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
