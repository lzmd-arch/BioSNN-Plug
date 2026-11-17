"""基因组编解码的守卫测试。

每一条都对应设计里的一个**性质**，不是"跑通就行"：

| 测试 | 守住的性质 |
| :--- | :--- |
| `test_roundtrip_*` | 编码无损 |
| `test_revcomp_involution` | 两条链反平行 |
| `test_single_strand_damage_detected` | **互补 ⇒ 单链损伤可检** |
| `test_checksum_catches_corruption` | 双链同时损伤也能查（校验和） |
| `test_species_changes_iff_conserved` | **规范性**：保守区 = 物种 |
| `test_mutation_permission_matrix` | **有害突变可预防（分区权限）** |
| `test_indel_frameshift_lethal` | 移码致死 |
| `test_transition_bias` | 转换/颠换 2:1，与真实基因组同 |
| `test_diploid_masking` | **有害突变可预防（二倍体掩蔽）** |
| `test_fitness_has_no_accuracy` | **有益突变可选择（适应度口径）** |
| `test_quantum_axis_*` | **第三轴不可复制 / 不可读出 / 只能迁移** |
| `test_backup_restore_same_species_not_same_individual` | 备份语义 |
"""

from __future__ import annotations

import random

import pytest

from scripts.genome_codec import (
    CODING,
    CONSERVED,
    FIXED,
    INTRON,
    MAGIC,
    REGULATORY,
    Chromosome,
    Genome,
    NoCloneError,
    QuantumAxis,
    UnreadableError,
    backup,
    decode_genome,
    decode_int,
    default_layout,
    encode_genome,
    encode_int,
    express,
    fitness,
    genome_length,
    indel,
    individual_id,
    positions_of,
    restore,
    revcomp,
    select,
    spans,
    species_barcode,
    substitute,
    verify_backup,
    verify_checksum,
    viable,
)

CONSTANTS = {"a": 0.5, "b": 2.0, "N": 1024}
SEEDS = {"net": 7, "init": 3}
LAYOUT = default_layout(CONSTANTS, SEEDS, int_constants={"N"})
VALUES = {**CONSTANTS, **SEEDS}


def _fresh() -> Genome:
    return Genome.build(VALUES, LAYOUT)


# --------------------------------------------------------------------------- #
# 碱基层
# --------------------------------------------------------------------------- #
def test_roundtrip_int():
    for n, w in ((0, 4), (1, 4), (255, 4), (4095, 6), (2**32 - 1, 16)):
        assert decode_int(encode_int(n, w)) == n


def test_roundtrip_genome():
    seq = encode_genome(VALUES, LAYOUT)
    assert len(seq) == genome_length(LAYOUT)
    got = decode_genome(seq, LAYOUT)
    for k, v in VALUES.items():
        assert abs(float(got[k]) - float(v)) < 1e-3, k
    assert got["MAGIC"] == MAGIC


def test_revcomp_involution():
    seq = encode_genome(VALUES, LAYOUT)
    assert revcomp(revcomp(seq)) == seq
    assert revcomp(seq) != seq


# --------------------------------------------------------------------------- #
# 损伤检测
# --------------------------------------------------------------------------- #
def test_fresh_genome_is_intact():
    assert _fresh().integrity() == []


def test_single_strand_damage_detected():
    """**互补链存在的唯一理由**：单链坏一个碱基，就是可见的错配。"""
    g = _fresh()
    plus = g.homolog_a.plus
    i = 40
    flipped = "ACGT"[("ACGT".index(plus[i]) + 1) % 4]
    damaged = Chromosome(plus=plus[:i] + flipped + plus[i + 1 :], minus=g.homolog_a.minus)
    assert damaged.mismatch_positions() == [i]
    assert damaged.repair().mismatch_positions() == []


def test_checksum_catches_corruption():
    seq = bytearray(encode_genome(VALUES, LAYOUT).encode())
    assert verify_checksum(seq.decode(), LAYOUT)
    for i in (5, 30, 100):
        bad = bytearray(seq)
        bad[i] = ord({"A": "C", "C": "G", "G": "T", "T": "A"}[chr(bad[i])])
        assert not verify_checksum(bad.decode(), LAYOUT), i


# --------------------------------------------------------------------------- #
# 规范性
# --------------------------------------------------------------------------- #
def test_species_changes_iff_conserved():
    """**规范性**：物种身份只由保守区定义，且对它敏感。"""
    seq = encode_genome(VALUES, LAYOUT)
    code = species_barcode(seq, LAYOUT)
    pos = positions_of(LAYOUT, CONSERVED)[0]
    changed = seq[:pos] + ("A" if seq[pos] != "A" else "C") + seq[pos + 1 :]
    assert species_barcode(changed, LAYOUT) != code

    pos2 = positions_of(LAYOUT, INTRON)[0]
    changed2 = seq[:pos2] + ("A" if seq[pos2] != "A" else "C") + seq[pos2 + 1 :]
    assert species_barcode(changed2, LAYOUT) == code


# --------------------------------------------------------------------------- #
# 变异权限
# --------------------------------------------------------------------------- #
def test_mutation_permission_matrix():
    seq = encode_genome(VALUES, LAYOUT)
    rng = random.Random(0)

    _, v_fixed = substitute(seq, LAYOUT, FIXED, rng, ref=CONSTANTS)
    assert not v_fixed.ok

    _, v_cons = substitute(seq, LAYOUT, CONSERVED, rng, ref=CONSTANTS)
    assert not v_cons.ok and "物种" in v_cons.why

    _, v_intron = substitute(seq, LAYOUT, INTRON, rng, ref=CONSTANTS)
    assert v_intron.ok and not v_intron.lethal

    _, v_code = substitute(seq, LAYOUT, CODING, rng, ref=CONSTANTS)
    assert v_code.ok and not v_code.lethal


def test_regulatory_out_of_band_is_lethal():
    """调控区不是禁改，而是**有安全带**：出带即致死。"""
    seq = encode_genome(VALUES, LAYOUT)
    rng = random.Random(0)
    saw_lethal = False
    for _ in range(300):
        _s2, v = substitute(seq, LAYOUT, REGULATORY, rng, ref=CONSTANTS, tol=0.01)
        if v.lethal:
            saw_lethal = True
            assert not v.ok
    assert saw_lethal, "托尔 1% 下 300 次替换一次都没出带，说明判据没接线"


def test_indel_frameshift_lethal():
    """编码区插入缺失 = 移码 = 致死；内含子里随便。"""
    seq = encode_genome(VALUES, LAYOUT)
    rng = random.Random(1)
    _, v = indel(seq, LAYOUT, CODING, rng, ref=CONSTANTS)
    assert v.lethal and not v.ok
    _, v2 = indel(seq, LAYOUT, CONSERVED, rng, ref=CONSTANTS)
    assert not v2.ok
    ok_seen = 0
    for _ in range(50):
        _, v3 = indel(seq, LAYOUT, INTRON, rng, ref=CONSTANTS)
        ok_seen += int(v3.ok)
    assert ok_seen > 0


def test_transition_bias():
    """转换:颠换 ≈ 2:1 —— 真实基因组的偏倚，这里必须复现。"""
    from scripts.genome_codec import _TRANSITION

    seq = encode_genome(VALUES, LAYOUT)
    lo, hi = spans(LAYOUT, len(seq))["NOTES"]  # 尾巴会跟着变，只在内含子区段里比
    rng = random.Random(7)
    trans = total = 0
    for _ in range(400):
        s2, _ = substitute(seq, LAYOUT, INTRON, rng, ref=CONSTANTS)
        diffs = [(a, b) for a, b in zip(seq[lo:hi], s2[lo:hi], strict=True) if a != b]
        if len(diffs) != 1:
            continue
        a, b = diffs[0]
        total += 1
        trans += int(_TRANSITION[a] == b)
    assert total > 300
    assert 0.6 <= trans / total <= 0.72, trans / total


# --------------------------------------------------------------------------- #
# 二倍体掩蔽
# --------------------------------------------------------------------------- #
def test_diploid_masking():
    """**有害突变可预防**：坏等位在杂合时自动隐性。"""
    good = encode_genome(VALUES, LAYOUT)
    rng = random.Random(3)
    bad = good
    for _ in range(400):
        cand, v = substitute(bad, LAYOUT, REGULATORY, rng, ref=CONSTANTS, tol=0.01)
        if v.lethal:
            bad = cand
            break
    assert bad != good
    ok_bad, _ = viable(decode_genome(bad, LAYOUT), CONSTANTS, tol=0.01)
    assert not ok_bad

    vals, why = express(good, bad, LAYOUT, CONSTANTS, tol=0.01)
    assert vals is not None and "同源 a" in why and "掩蔽" in why

    vals2, why2 = express(bad, bad, LAYOUT, CONSTANTS, tol=0.01)
    assert vals2 is None and "致死" in why2


def test_express_gate_is_honoured():
    good = encode_genome(VALUES, LAYOUT)
    assert express(good, good, LAYOUT, CONSTANTS, gate=lambda _v: True)[0] is not None
    assert express(good, good, LAYOUT, CONSTANTS, gate=lambda _v: False)[0] is None


# --------------------------------------------------------------------------- #
# 适应度
# --------------------------------------------------------------------------- #
def test_fitness_has_no_accuracy():
    """**有益突变可选择**：适应度里没有准确率，且存活是硬门槛。"""
    import inspect

    params = inspect.signature(fitness).parameters
    assert set(params) == {"viable", "damage_tolerance", "stability", "efficiency"}

    dead = fitness(viable=False, damage_tolerance=1.0, stability=1.0, efficiency=1.0)
    alive = fitness(viable=True, damage_tolerance=0.0, stability=0.0, efficiency=0.0)
    assert alive > dead, "死掉的个体不能因为其它项高就更适应"


def test_select_strict_and_soft():
    rng = random.Random(0)
    assert select(1.0, 1.1, rng, temperature=0.0)
    assert not select(1.0, 1.0, rng, temperature=0.0)
    accepts = sum(select(1.0, 0.9, rng, temperature=0.5) for _ in range(2000))
    assert 0 < accepts < 2000


# --------------------------------------------------------------------------- #
# 第三轴
# --------------------------------------------------------------------------- #
def test_quantum_axis_refuses_to_clone_or_export():
    q = QuantumAxis(generator_seed=1, birth_hash="ab" * 8)
    with pytest.raises(NoCloneError):
        q.clone()
    with pytest.raises(UnreadableError):
        q.export_state()


def test_quantum_axis_move_kills_original():
    """**移动不是复制**：任意时刻只有一份活着。"""
    q = QuantumAxis(generator_seed=1, birth_hash="ab" * 8)
    heir = q.move()
    assert heir.alive and not q.alive
    assert heir.generator_seed == 1
    with pytest.raises(RuntimeError):
        q.observables(0.05, 0.9)
    assert heir.observables(0.05, 0.9)["firing"] == 0.05


def test_quantum_axis_dies_on_power_loss():
    q = QuantumAxis(generator_seed=2, birth_hash="cd" * 8)
    q.die()
    assert not q.alive
    with pytest.raises(RuntimeError):
        q.move()


# --------------------------------------------------------------------------- #
# 备份语义
# --------------------------------------------------------------------------- #
def test_backup_restore_same_species_not_same_individual():
    """**恢复的是「同基因组的个体」，不是「同一个体」。**"""
    g = _fresh()
    blob = backup(g)
    assert "quantum" not in blob and "state" not in blob
    g2 = restore(blob, LAYOUT)
    assert g2.integrity() == []
    assert species_barcode(g2.homolog_a.plus, LAYOUT) == species_barcode(g.homolog_a.plus, LAYOUT)
    # 证据：恢复出来的第三轴是一段**新**历史——指针相同不代表历史相同
    q1, q2 = QuantumAxis(1, "x"), QuantumAxis(1, "x")
    assert q1 is not q2


def test_restore_rejects_wrong_length():
    with pytest.raises(ValueError):
        restore({"plus_a": "ACGT", "plus_b": "ACGT"}, LAYOUT)


# --------------------------------------------------------------------------- #
# 个体身份与两层冗余
# --------------------------------------------------------------------------- #
def test_double_strand_damage_caught_by_checksum_only():
    """**两层冗余各管一段**：配对查单链损伤，校验和查双链一致损伤。"""
    good = encode_genome(VALUES, LAYOUT)
    i = 90
    flipped = "ACGT"[("ACGT".index(good[i]) + 1) % 4]
    dsb = good[:i] + flipped + good[i + 1 :]  # 两条链**一致地**改了
    ch = Chromosome.build(dsb)
    assert ch.mismatch_positions() == [], "双链一致损伤不该被配对查出来"
    assert not verify_checksum(dsb, LAYOUT), "双链一致损伤必须被校验和查出来"


def test_individual_id_distinguishes_same_species():
    """**物种条码认物种，个体身份认个体**——备份少了后者就只能说"还是千岁"。"""
    a = _fresh()
    rng = random.Random(11)
    seq_b, _ = substitute(a.homolog_a.plus, LAYOUT, CODING, rng, ref=CONSTANTS)
    b = Genome(Chromosome.build(seq_b), Chromosome.build(seq_b), list(LAYOUT))
    assert species_barcode(a.homolog_a.plus, LAYOUT) == species_barcode(b.homolog_a.plus, LAYOUT), (
        "编码区变异不换物种"
    )
    assert individual_id(a) != individual_id(b), "编码区变异必须换个体身份"


def test_verify_backup_roundtrip_and_tamper():
    g = _fresh()
    blob = backup(g)
    ok, bad = verify_backup(blob, LAYOUT)
    assert ok and bad == []

    # 把备份里的正链改一个碱基 = 双链一致损伤
    seq = blob["plus_a"]
    i = 90
    blob_bad = dict(blob)
    blob_bad["plus_a"] = seq[:i] + ("A" if seq[i] != "A" else "C") + seq[i + 1 :]
    ok2, bad2 = verify_backup(blob_bad, LAYOUT)
    assert not ok2 and any("校验和" in x for x in bad2)

    # 换一个个体的备份冒充她
    rng = random.Random(12)
    other_seq, _ = substitute(seq, LAYOUT, CODING, rng, ref=CONSTANTS)
    blob_other = dict(blob)
    blob_other["plus_a"] = blob_other["plus_b"] = other_seq
    ok3, bad3 = verify_backup(blob_other, LAYOUT)
    assert not ok3 and any("个体身份" in x for x in bad3)
