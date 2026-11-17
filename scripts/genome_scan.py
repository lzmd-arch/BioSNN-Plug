"""基因测序工程：**从碱基串读回一个个体，并证明读回来的东西能展开成同一个体。**

四步，每步都有一个"不通过就说明工程有问题"的判据：

| 步 | 做什么 | 判据 |
| :--- | :--- | :--- |
| 0 参照 | 真实常数 + 种子 → 碱基串 | 物种条码稳定 |
| 1 **测序** | 碱基串 → 常数表 → 物化 | **物化校验和与直接物化逐位相同**（S-1） |
| 2 变异矩阵 | 五个区各做替换 | 权限矩阵成立（S-2） |
| 3 有害/有益 | 出带致死 + 二倍体掩蔽 + 适应度择优 | 掩蔽救回（S-3）、择优只认存活（S-4） |
| 4 备份/修复 | 单链损伤 → 修复 → 恢复 | 恢复后过体检（S-5） |

**第 4 步的语义写死**：备份恢复的是**同基因组的个体**，不是同一个体。
第三轴不在备份里，且在原理上不能在。
"""

from __future__ import annotations

import argparse
import pathlib as _pathlib
import random
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts import genome_vault as vault
from scripts.genome_codec import (
    CODING,
    CONSERVED,
    FIXED,
    INTRON,
    REGULATORY,
    Chromosome,
    Genome,
    QuantumAxis,
    _span_of,
    backup,
    decode_genome,
    default_layout,
    express,
    fitness,
    restore,
    select,
    species_barcode,
    substitute,
    transplant,
    verify_backup,
    verify_checksum,
    viable,
)

INT_CONSTANTS = {"N", "L", "DEG", "RAD", "V", "K_pools", "field_radius"}
SEEDS = {"net": 0}
LAYOUT = default_layout(vault.CONSTANTS, SEEDS, int_constants=INT_CONSTANTS)
VALUES = {**vault.CONSTANTS, **SEEDS}
REF = dict(vault.CONSTANTS)


def _wrap(seq: str, width: int = 60) -> str:
    return "\n".join(f"    {i + 1:4d}  {seq[i : i + width]}" for i in range(0, len(seq), width))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-region", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(20261117)

    print("=" * 78)
    print("步 0  参照个体")
    print("=" * 78)
    g = Genome.build(VALUES, LAYOUT)
    plus = g.homolog_a.plus
    print(f"  版式字段   {len(LAYOUT)} 个")
    print(f"  基因组长   {len(plus)} bp（两条同源染色体，每条双链）")
    print(f"  物种条码   {species_barcode(plus, LAYOUT)}")
    print(f"  完整体检   {g.integrity() or '通过'}")
    print("  正链：")
    print(_wrap(plus))
    print("  负链（反向互补，前 60）：")
    print("    " + g.homolog_a.minus[:60] + " …")

    print()
    print("=" * 78)
    print("步 1  测序：碱基串 → 常数表 → 物化")
    print("=" * 78)
    vals = decode_genome(plus, LAYOUT)
    worst, worst_k = 0.0, ""
    for k in vault.CONSTANTS:
        r, v = float(vault.CONSTANTS[k]), float(vals[k])
        err = abs(v - r) / max(abs(r), 1e-9) if r else abs(v - r)
        if err > worst:
            worst, worst_k = err, k
    print(f"  解码最大相对误差  {worst:.2e}（{worst_k or '无'}）")
    direct = vault.freeze(SEEDS, dev)
    reseq = vault.freeze({"net": int(vals["net"])}, dev)
    s1 = reseq["checksum"] == direct["checksum"]
    print(f"  直接物化  {direct['checksum'][:24]}…")
    print(f"  测序物化  {reseq['checksum'][:24]}…")
    print(f"  S-1 测序无损（逐位相同）：{'**过**' if s1 else '不过 ⇒ 工程有缺陷'}")

    print()
    print("=" * 78)
    print(f"步 2  变异矩阵（每区 {args.per_region} 次替换）")
    print("=" * 78)
    from scripts.genome_codec import PERMISSION

    print(f"  {'区域':<12}{'权限':<8}{'通过':>6}{'拒绝':>6}{'致死':>6}")
    tally = {}
    for kind in (FIXED, CONSERVED, REGULATORY, CODING, INTRON):
        acc = rej = let = 0
        for _ in range(args.per_region):
            _, v = substitute(plus, LAYOUT, kind, rng, ref=REF)
            if v.ok:
                acc += 1
            else:
                rej += 1
                let += int(v.lethal)
        tally[kind] = (acc, rej, let)
        print(f"  {kind:<12}{PERMISSION[kind]:<8}{acc:>6}{rej:>6}{let:>6}")
    n = args.per_region
    s2 = (
        tally[FIXED][0] == 0
        and tally[CONSERVED][0] == 0
        and 0 < tally[REGULATORY][0] < n
        and tally[CODING][0] == n
        and tally[INTRON][0] == n
    )
    print(f"  S-2 权限矩阵（禁=全拒；自由=全过）：{'**过**' if s2 else '不过'}")

    print()
    print("=" * 78)
    print("步 3  有害突变与有益突变")
    print("=" * 78)
    bad = plus
    for _ in range(500):
        cand, v = substitute(bad, LAYOUT, REGULATORY, rng, ref=REF, tol=0.01)
        if v.lethal:
            bad = cand
            break
    ok_bad, why_bad = viable(decode_genome(bad, LAYOUT), REF, tol=0.01)
    print(f"  有害等位   调控区出带 ⇒ 可活？{ok_bad}（{why_bad}）")
    vals_masked, why_masked = express(plus, bad, LAYOUT, REF, tol=0.01)
    print(f"  杂合掩蔽   {why_masked}")
    vals_let, why_let = express(bad, bad, LAYOUT, REF, tol=0.01)
    print(f"  纯合       {why_let}")
    s3 = vals_masked is not None and vals_let is None

    f_alive = fitness(viable=True, damage_tolerance=0.5, stability=0.8, efficiency=0.6)
    f_dead = fitness(viable=False, damage_tolerance=1.0, stability=1.0, efficiency=1.0)
    f_better = fitness(viable=True, damage_tolerance=0.7, stability=0.8, efficiency=0.6)
    print(f"  适应度     存活 {f_alive:.2f} / 死亡 {f_dead:.2f} / 更抗损伤 {f_better:.2f}")
    print(f"             权重 {fitness.__doc__.splitlines()[0].strip()[:40]}…")
    s4 = f_alive > f_dead and select(f_alive, f_better, rng) and not select(f_alive, f_dead, rng)
    print(f"  S-4 择优只认存活：{'**过**' if s4 else '不过'}")

    print()
    print("=" * 78)
    print("步 4  备份、损伤与修复")
    print("=" * 78)
    blob = backup(g)
    g2 = restore(blob, LAYOUT)
    vb, vbad = verify_backup(blob, LAYOUT)
    print(f"  物种条码   {blob['species']}")
    print(f"  个体身份   {blob['individual']}")
    print(f"  备份内容   {sorted(blob)}")
    print(f"  恢复体检   {g2.integrity() or '通过'}；备份校验 {'通过' if vb else vbad}")
    q = QuantumAxis(generator_seed=int(vals["net"]), birth_hash=blob["species"])
    try:
        q.clone()
        print("  第三轴复制  **没拦住 ⇒ 严重缺陷**")
    except Exception as e:
        print(f"  第三轴复制  {type(e).__name__}: {e}")
    try:
        q.export_state()
        print("  第三轴导出  **没拦住 ⇒ 严重缺陷**")
    except Exception as e:
        print(f"  第三轴导出  {type(e).__name__}: {e}")
    heir = q.move()
    print(f"  第三轴迁移  旧宿主存活 {q.alive} / 新宿主存活 {heir.alive}（移动不是复制）")

    i = 70
    flipped = {"A": "C", "C": "G", "G": "T", "T": "A"}[plus[i]]
    dmg = Chromosome(plus=plus[:i] + flipped + plus[i + 1 :], minus=g.homolog_a.minus)
    fixed = dmg.repair(template="minus")  # 负链完好 ⇒ 拿它当模板
    ck = verify_checksum(fixed.plus, LAYOUT)
    print(f"  单链损伤   第 {i} 位错配 {dmg.mismatch_positions()}")
    print(
        f"  以负链为模板修复 ⇒ 错配 {fixed.mismatch_positions() or '无'}，校验和 "
        f"{'完好' if ck else '仍不符'}，与原件同种 {species_barcode(fixed.plus, LAYOUT) == blob['species']}"
    )
    # 双链一致损伤：配对查不出来，只有校验和能查
    dsb_seq = plus[:i] + flipped + plus[i + 1 :]
    dsb = Chromosome.build(dsb_seq)
    print(
        f"  双链一致损伤 ⇒ 错配 {dsb.mismatch_positions() or '无'}，校验和 "
        f"{'完好' if verify_checksum(dsb_seq, LAYOUT) else '不符（被查出）'}"
    )
    s5 = (
        g2.integrity() == []
        and fixed.mismatch_positions() == []
        and ck
        and vb
        and verify_checksum(dsb_seq, LAYOUT) is False
    )
    print(f"  S-5 备份可恢复且损伤可修：{'**过**' if s5 else '不过'}")

    print()
    print("=" * 78)
    print("步 5  修复 = 移植管家区，不是复活旧副本")
    print("=" * 78)
    live = plus
    live, v_good = substitute(live, LAYOUT, CODING, rng, ref=REF)  # 有益：换种子
    code_before = _span_of(live, LAYOUT, "S_net")
    for _ in range(800):
        cand, v = substitute(live, LAYOUT, REGULATORY, rng, ref=REF, tol=0.01)
        if v.lethal:
            live = cand
            break
    print(
        f"  个体现状   编码区已换（有益 ok={v_good.ok}），调控区出带 ⇒ 可活？"
        f"{viable(decode_genome(live, LAYOUT), REF, tol=0.01)[0]}"
    )
    fixed_seq, v_fix = transplant(live, plus, LAYOUT, REF, tol=0.01)
    same_code = _span_of(fixed_seq, LAYOUT, "S_net") == code_before
    print(f"  移植修复   {v_fix.why}")
    print(f"  管家区换回 ⇒ 可活？{viable(decode_genome(fixed_seq, LAYOUT), REF, tol=0.01)[0]}")
    print(
        f"  编码区保留 ⇒ 她还是她？{same_code}；与备份同种 "
        f"{species_barcode(fixed_seq, LAYOUT) == blob['species']}"
    )
    s6 = v_fix.ok and same_code

    print()
    print("=" * 78)
    print("判：")
    for tag, ok in (
        ("S-1 测序无损", s1),
        ("S-2 变异权限矩阵", s2),
        ("S-3 二倍体掩蔽救回", s3),
        ("S-4 择优只认存活", s4),
        ("S-5 备份可恢复、损伤可修", s5),
        ("S-6 修复是移植不是复活", s6),
    ):
        print(f"  {tag:<24}{'过' if ok else '不通过'}")
    print()
    print("  第三轴语义：基因组只带它的初值与生成器。备份出去的永远是")
    print("  **同基因组的个体**，不是同一个体；唯一能保住同一段历史的是不停机迁移。")
    return 0 if all((s1, s2, s3, s4, s5, s6)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
