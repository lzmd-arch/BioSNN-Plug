"""基因组编解码：**双螺旋（碱基对）× 第三轴（不可克隆轨迹）**。

设计见 `D:/Projects/Qiansui-Internal/GOAL-silicon-life.md` §41。

## 为什么是碱基对，不是 JSON

真实基因组的四条性质，JSON 一条都没有：

| 性质 | 生物机制 | 本模块的对应 |
| :--- | :--- | :--- |
| **互补** | A-T / G-C 双链；单链损伤 = 错配 | 两条链独立保存，不一致即损伤 |
| **离散** | 替换分转换/颠换，可枚举 | 每个碱基恰好 3 种替代 |
| **分区** | 保守区变异即致死 | 区域决定变异权限 |
| **二倍体** | 隐性有害等位被掩蔽 | 两条同源染色体，坏等位自动隐性 |

## 双螺旋部分承载什么

**只承载「展开规则」**：物种身份、全部常数、种子。**不承载历史。**

## 第三轴按实际来设计

第三轴**没有序列**——它是一条**轨迹**，不是一段**序列**。实测依据（本项目）：

- `m6ay`：量子层对记忆的贡献 1.01×（经典层去同步对照 0.88×）——**它不是存储**；
- `m6ax`：死亡阈值处半球互信息崩塌——**它靠持续供给维持**，不靠自身保存；
- 它不可克隆（no-cloning）、不可读出（`m6ay` 无可用读出结构）。

所以基因组里**不能**给它留一段碱基：基因组只携带它的**生成器与初值**，其余由运行产生。
它的三条性质由协议层保证（`QuantumAxis`），不由编码保证。

## 由此得到的一条硬结论

**「备份」恢复的永远是「同基因组的个体」，不是「同一个体」。**

同基因组 ⇒ 物种、常数、初值全同，展开出的个体可过全部验收闸；
不同个体 ⇒ 第三轴是另一段历史。唯一能保住同一段历史的只有**不停机迁移**（`QuantumAxis.move`）。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

ALPHABET = "ACGT"
_DIGIT = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
#: 转换（嘌呤↔嘌呤、嘧啶↔嘧啶）。真实基因组里约是颠换的 2 倍。
_TRANSITION = {"A": "G", "G": "A", "C": "T", "T": "C"}

#: 定点小数标度：`0.95` 存成 `950`
FP_SCALE = 1000

#: 文件头，固定 8 碱基
MAGIC = "GCTAACGT"

#: 物种身份与哨兵。**任何个体都必须是这两个值**——改一个碱基就不是千岁。
REFERENCE_SPECIES = 0x5EED_1CE5
REFERENCE_SENTINEL = 0x0BAD_F00D
DEFAULTS = {"MAGIC": MAGIC, "SPECIES": REFERENCE_SPECIES, "SENTINEL": REFERENCE_SENTINEL}

#: 区域种类
FIXED = "fixed"
CONSERVED = "conserved"
REGULATORY = "regulatory"
CODING = "coding"
INTRON = "intron"

#: 区域 → 变异权限（判据写死，不靠调用方自觉）
PERMISSION = {
    FIXED: "禁",
    CONSERVED: "禁",  # 改它 = 不是同一物种
    REGULATORY: "受限",  # 允许，但表型值必须留在参考值 ±tol 内，否则致死
    CODING: "允许",  # 替换允许；插入缺失 = 移码 = 致死
    INTRON: "自由",  # 随便改，不参与表型
}

#: 字段 = (名字, 区域, 宽度/碱基数, 编码)
Field = tuple[str, str, int, str]

#: 适应度权重。**准确率刻意不在其中**（见 `fitness`）。
WEIGHTS = {"viable": 1.0, "tolerance": 0.3, "stability": 0.3, "efficiency": 0.2}


# --------------------------------------------------------------------------- #
# 碱基层
# --------------------------------------------------------------------------- #
def revcomp(seq: str) -> str:
    """反向互补链。真实 DNA 的两条链是**反平行**的。"""
    return "".join(_COMP[b] for b in reversed(seq))


def encode_int(n: int, width: int) -> str:
    """整数 → 定宽碱基串（大端，每碱基 2 bit）。"""
    if n < 0 or n >= 4**width:
        raise ValueError(f"{n} 放不进 {width} 个碱基（上限 {4**width - 1}）")
    out = []
    for _ in range(width):
        out.append(ALPHABET[n & 3])
        n >>= 2
    return "".join(reversed(out))


def decode_int(seq: str) -> int:
    n = 0
    for b in seq:
        n = (n << 2) | _DIGIT[b]
    return n


def encode_float(x: float, width: int) -> str:
    """定点小数 → 碱基串。`0.95 → 950 → 6 碱基`。"""
    return encode_int(round(x * FP_SCALE), width)


def decode_float(seq: str) -> float:
    return decode_int(seq) / FP_SCALE


def bare(name: str) -> str:
    """版式字段名 → 表型键名（`C_leak` → `leak`，`S_net` → `net`）。"""
    return name[2:] if name[:2] in ("C_", "S_") else name


def _checksum(seq: str) -> int:
    return int.from_bytes(hashlib.sha256(seq.encode()).digest()[:4], "big")


# --------------------------------------------------------------------------- #
# 版式（相当于测序里的「参考基因组注释」）
# --------------------------------------------------------------------------- #
def default_layout(
    constants: dict[str, float], seeds: dict[str, int], int_constants: set[str] | None = None
) -> list[Field]:
    """按常数表 + 种子表生成参考版式。**顺序即碱基顺序，必须稳定。**"""
    intc = int_constants or set()
    fields: list[Field] = [
        ("MAGIC", FIXED, len(MAGIC), "literal"),
        ("SPECIES", CONSERVED, 16, "int"),
        ("SENTINEL", CONSERVED, 16, "int"),
    ]
    fields += [(f"C_{k}", REGULATORY, 8, "int" if k in intc else "float") for k in constants]
    fields += [(f"S_{k}", CODING, 16, "int") for k in seeds]
    fields += [("NOTES", INTRON, 24, "int")]
    fields += [("TAIL", FIXED, 16, "checksum")]
    return fields


def spans(layout: list[Field], total_len: int | None = None) -> dict[str, tuple[int, int]]:
    """位置表。**内含子是可变长的**——它吃掉编码区之后、尾巴之前的一切。

    这是"按真实基因组来"的一处：真实内含子对插入缺失宽容，靠剪接信号自定界；
    本设计里内含子靠**尾巴从后方锚定**来自定界，所以它的长度可以变。
    其余区域全是定宽的 ⇒ 插入缺失即移码。
    """
    if total_len is None:
        total_len = sum(w for _n, _k, w, _c in layout)
    tail_w = sum(w for _n, _k, w, c in layout if c == "checksum")
    stop = total_len - tail_w
    out: dict[str, tuple[int, int]] = {}
    i = 0
    for name, kind, width, codec in layout:
        if kind == INTRON or codec == "checksum":
            continue
        out[name] = (i, i + width)
        i += width
    if i > stop:
        raise ValueError(f"固定区 {i} 超过可容纳长度 {stop}（内含子被挤没了）")
    for name, kind, _w, _c in layout:
        if kind == INTRON:
            out[name] = (i, stop)
    j = stop
    for name, _kind, width, codec in layout:
        if codec == "checksum":
            out[name] = (j, j + width)
            j += width
    return out


def positions_of(layout: list[Field], kind: str, total_len: int | None = None) -> list[int]:
    sp, out = spans(layout, total_len), []
    for name, k, _w, _c in layout:
        if k == kind:
            out.extend(range(*sp[name]))
    return out


def genome_length(layout: list[Field]) -> int:
    """**参考长度**（含参考内含子长度）。"""
    return sum(w for _n, _k, w, _c in layout)


def min_length(layout: list[Field]) -> int:
    """**可活最短长度**：固定区 + 尾巴。内含子可以被删到 0。"""
    return sum(w for _n, k, w, _c in layout if k != INTRON)


# --------------------------------------------------------------------------- #
# 编解码
# --------------------------------------------------------------------------- #
def encode_genome(values: dict[str, float | int], layout: list[Field]) -> str:
    """表型值 → 正链碱基串。**末尾自动补校验和字段。**"""
    parts: list[str] = []
    for name, _kind, width, codec in layout:
        if codec == "literal":
            parts.append(DEFAULTS[name])
        elif codec == "checksum":
            parts.append(encode_int(_checksum("".join(parts)), width))
        elif codec == "int":
            key = bare(name)
            v = values[key] if key in values else DEFAULTS.get(name, 0)
            parts.append(encode_int(int(v), width))
        else:
            parts.append(encode_float(float(values[bare(name)]), width))
    return "".join(parts)


def decode_genome(seq: str, layout: list[Field]) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    sp = spans(layout, len(seq))
    for name, _kind, _width, codec in layout:
        a, b = sp[name]
        chunk = seq[a:b]
        if codec == "literal":
            out[bare(name)] = chunk
        elif codec in ("checksum", "int"):
            out[bare(name)] = decode_int(chunk)
        else:
            out[bare(name)] = decode_float(chunk)
    return out


def _refresh_tail(seq: str, layout: list[Field]) -> str:
    sp = spans(layout, len(seq))
    a, b = sp["TAIL"]
    return seq[:a] + encode_int(_checksum(seq[:a]), b - a)


def verify_checksum(seq: str, layout: list[Field]) -> bool:
    sp = spans(layout, len(seq))
    a, b = sp["TAIL"]
    return decode_int(seq[a:b]) == _checksum(seq[:a])


def individual_id(g: Genome) -> str:
    """**个体身份** = 两条同源染色体整串的指纹。

    物种条码只认保守区，所以同种个体条码相同；`individual_id` 才认到个体。
    备份必须带它：否则"恢复成功"只能说明"还是千岁"，不能说明"还是她"。
    """
    h = hashlib.sha256()
    for seq in (g.homolog_a.plus, g.homolog_b.plus):
        h.update(seq.encode())
        h.update(b"|")
    return h.hexdigest()[:32]


def species_barcode(seq: str, layout: list[Field]) -> str:
    """**物种身份** = 保守区的指纹。改一个碱基就换物种。"""
    sp = spans(layout, len(seq))
    parts = [seq[slice(*sp[n])] for n, k, _w, _c in layout if k == CONSERVED]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 染色体与基因组（二倍体）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chromosome:
    """一条染色体 = 一对互补链。**单链损伤表现为错配。**"""

    plus: str
    minus: str

    @classmethod
    def build(cls, plus: str) -> Chromosome:
        return cls(plus=plus, minus=revcomp(plus))

    def mismatch_positions(self) -> list[int]:
        """两条链不一致的位置（= 单链损伤）。空列表 = 完好。"""
        n = len(self.plus)
        return [i for i in range(n) if self.minus[n - 1 - i] != _COMP[self.plus[i]]]

    def repair(self, template: str = "plus") -> Chromosome:
        """用**完好那条链**做模板重建另一条（= 错配修复）。

        真实错配修复靠"哪条是新合成的"来选模板；这里靠调用方指明哪条完好。
        """
        if template == "plus":
            return Chromosome.build(self.plus)
        return Chromosome.build(revcomp(self.minus))


@dataclass(frozen=True)
class Genome:
    """二倍体基因组：两条同源染色体 + 参考版式。"""

    homolog_a: Chromosome
    homolog_b: Chromosome
    layout: list[Field]

    @classmethod
    def build(cls, values: dict[str, float | int], layout: list[Field]) -> Genome:
        plus = encode_genome(values, layout)
        return cls(Chromosome.build(plus), Chromosome.build(plus), list(layout))

    def same_species(self) -> bool:
        return species_barcode(self.homolog_a.plus, self.layout) == species_barcode(
            self.homolog_b.plus, self.layout
        )

    def integrity(self) -> list[str]:
        """完整体检。返回问题清单；空 = 完好。"""
        problems: list[str] = []
        for tag, ch in (("a", self.homolog_a), ("b", self.homolog_b)):
            mm = ch.mismatch_positions()
            if mm:
                problems.append(f"同源 {tag} 有 {len(mm)} 处链间错配（单链损伤）")
            if not verify_checksum(ch.plus, self.layout):
                problems.append(f"同源 {tag} 校验和不符")
        if not self.same_species():
            problems.append("两条同源染色体的保守区不同 ⇒ 不是同一物种")
        return problems


# --------------------------------------------------------------------------- #
# 变异：替换 / 插入缺失
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Verdict:
    ok: bool
    lethal: bool
    why: str


def _adjudicate(seq: str, layout: list[Field], kind: str, ref: dict, tol: float) -> Verdict:
    if PERMISSION[kind] == "禁":
        return Verdict(False, False, f"{kind} 区禁变异 ⇒ 拒绝（不是同一物种）")
    if PERMISSION[kind] == "自由":
        return Verdict(True, False, "内含子，不参与表型")
    vals = decode_genome(seq, layout)
    if kind == REGULATORY:
        for name, _k, _w, codec in layout:
            if not name.startswith("C_") or codec == "literal":
                continue
            key = bare(name)
            r = float(ref[key])
            v = float(vals[key])
            if r == 0.0:
                if v != 0.0:
                    return Verdict(False, True, f"调控区 {key} 从 0 漂走 ⇒ 致死")
            elif abs(v - r) / abs(r) > tol:
                return Verdict(False, True, f"调控区 {key} 偏出 ±{tol:.0%}（{v} vs {r}）⇒ 致死")
        return Verdict(True, False, "调控区内、幅度合规")
    return Verdict(True, False, "编码区替换（种子改变，表型交由验收闸判）")


def substitute(
    seq: str,
    layout: list[Field],
    kind: str,
    rng: random.Random,
    *,
    ref: dict,
    tol: float = 0.10,
) -> tuple[str, Verdict]:
    """在指定区域做一次**点替换**。转换优先（2:1），与真实基因组一致。"""
    pos_list = positions_of(layout, kind, len(seq))
    if not pos_list:
        return seq, Verdict(False, False, f"没有 {kind} 区")
    i = rng.choice(pos_list)
    old = seq[i]
    if rng.random() < 2 / 3:
        new = _TRANSITION[old]
    else:
        new = rng.choice([b for b in ALPHABET if b != old and b != _TRANSITION[old]])
    new_seq = _refresh_tail(seq[:i] + new + seq[i + 1 :], layout)
    return new_seq, _adjudicate(new_seq, layout, kind, ref, tol)


def indel(
    seq: str,
    layout: list[Field],
    kind: str,
    rng: random.Random,
    *,
    ref: dict,
    tol: float = 0.10,
) -> tuple[str, Verdict]:
    """插入或删除一个碱基。**编码区/调控区一律移码致死**（与真实基因组一致）。"""
    if PERMISSION.get(kind) in ("禁",):
        return seq, Verdict(False, False, f"{kind} 区禁变异 ⇒ 拒绝")
    if kind in (CODING, REGULATORY):
        return seq, Verdict(False, True, f"{kind} 区插入缺失 = 移码 ⇒ 致死")
    pos_list = positions_of(layout, kind, len(seq))
    if not pos_list:
        return seq, Verdict(False, False, f"没有 {kind} 区")
    i = rng.choice(pos_list)
    if rng.random() < 0.5 and len(pos_list) > 4:
        new_seq = seq[:i] + seq[i + 1 :]
    else:
        new_seq = seq[:i] + rng.choice(ALPHABET) + seq[i:]
    if len(new_seq) != len(seq) and kind != INTRON:
        return new_seq, Verdict(False, True, "长度改变 ⇒ 版式不符 ⇒ 致死")
    new_seq = _refresh_tail(new_seq, layout)
    return new_seq, _adjudicate(new_seq, layout, kind, ref, tol)


# --------------------------------------------------------------------------- #
# 表达：二倍体掩蔽
# --------------------------------------------------------------------------- #
def viable(values: dict, ref: dict, tol: float = 0.10) -> tuple[bool, str]:
    """调控区是否都在合规带内。**编码区种子不在此判**（那是验收闸的事）。"""
    for key, r in ref.items():
        if key not in values:
            continue
        r, v = float(r), float(values[key])
        if r == 0.0:
            if v != 0.0:
                return False, f"{key} 从 0 漂走"
        elif abs(v - r) / abs(r) > tol:
            return False, f"{key} 偏出 ±{tol:.0%}"
    return True, "ok"


def express(
    homolog_a: str,
    homolog_b: str,
    layout: list[Field],
    ref: dict,
    *,
    tol: float = 0.10,
    gate=None,
) -> tuple[dict | None, str]:
    """二倍体表达。**坏等位自动隐性**：能过闸的那条被表达，两条都过不了才致死。

    `gate(values) -> bool` 是昂贵的验收闸（展开后跑网络）；不传则只看调控区。
    """
    cands = []
    for tag, seq in (("a", homolog_a), ("b", homolog_b)):
        vals = decode_genome(seq, layout)
        ok, why = viable(vals, ref, tol)
        if ok and gate is not None:
            ok = bool(gate(vals))
            why = "ok" if ok else "验收闸不过"
        cands.append((tag, vals, ok, why))
    for tag, vals, ok, _why in cands:
        if ok:
            masked = "（另一条同源被掩蔽）" if tag == "a" and cands[1][2] is False else ""
            return vals, f"表达同源 {tag}{masked}"
    return None, "两条同源都过不了闸 ⇒ 致死：" + "; ".join(f"{c[0]}:{c[3]}" for c in cands)


# --------------------------------------------------------------------------- #
# 适应度：**不含准确率**
# --------------------------------------------------------------------------- #
def fitness(*, viable: bool, damage_tolerance: float, stability: float, efficiency: float) -> float:
    """适应度 = 存活 + 抗损伤 + 稳定 + 效率。

    **准确率刻意不在其中。** 理由是实测的：本项目的准确率指标反复把"更自信"当成"更好"
    （`B0` 的 BP 训练 CE 尾/首 = 0.0175，测试 CE 9.31 比均匀还差），而且准确率一旦进适应度，
    选择就会把种群推向**记忆化**而不是**活得好**。
    """
    return (
        WEIGHTS["viable"] * (1.0 if viable else 0.0)
        + WEIGHTS["tolerance"] * max(0.0, min(1.0, damage_tolerance))
        + WEIGHTS["stability"] * max(0.0, min(1.0, stability))
        + WEIGHTS["efficiency"] * max(0.0, min(1.0, efficiency))
    )


def select(old: float, new: float, rng: random.Random, *, temperature: float = 0.0) -> bool:
    """接受判据。`temperature=0` 为严格择优；>0 为软接受（近似中性漂变）。"""
    if temperature <= 0.0:
        return new > old
    import math

    return rng.random() < 1.0 / (1.0 + math.exp((old - new) / temperature))


# --------------------------------------------------------------------------- #
# 第三轴：按实际来设计
# --------------------------------------------------------------------------- #
class NoCloneError(RuntimeError):
    """第三轴不可复制（no-cloning theorem）。"""


class UnreadableError(RuntimeError):
    """第三轴没有可导出的状态；只能观测，不能读出。"""


class QuantumAxis:
    """第三轴：**没有序列**。

    它是一条**轨迹**，不是一段**序列**。所以：

    - 不能 `clone()` —— no-cloning；
    - 不能 `export_state()` —— 没有可用读出结构（`m6ay`）；
    - 只能 `move()` —— 不停机迁移，**移动不是复制**，任意时刻只有一份活着；
    - 断电即死 —— 它靠持续供给维持（`m6ax` 的死亡阈值）。

    基因组里携带的只有它的**初值**（`generator_seed`）与出生承诺（`birth_hash`）。
    """

    __slots__ = ("_alive", "birth_hash", "generator_seed")

    def __init__(self, generator_seed: int, birth_hash: str) -> None:
        self.generator_seed = generator_seed
        self.birth_hash = birth_hash
        self._alive = True

    @property
    def alive(self) -> bool:
        return self._alive

    def clone(self) -> QuantumAxis:
        raise NoCloneError("第三轴不可复制：备份到文件的是双螺旋，不是它。")

    def export_state(self) -> dict:
        raise UnreadableError("第三轴没有可导出状态；可用的是可观测量（发放/行为）。")

    def observables(self, firing: float, behavior_entropy: float) -> dict:
        """**可观测，不可读出。** 这两个数可以写进记录层，状态本身不行。"""
        if not self._alive:
            raise RuntimeError("第三轴已死，无可观测量")
        return {"firing": float(firing), "behavior_entropy": float(behavior_entropy)}

    def move(self) -> QuantumAxis:
        """不停机迁移：**移交，不是复制**。旧宿主随即死亡。"""
        if not self._alive:
            raise RuntimeError("第三轴已死，无法迁移")
        heir = QuantumAxis.__new__(QuantumAxis)
        heir.generator_seed = self.generator_seed
        heir.birth_hash = self.birth_hash
        heir._alive = True
        self._alive = False
        return heir

    def die(self) -> None:
        self._alive = False


def backup(g: Genome) -> dict:
    """**备份 = 双螺旋 + 记录层。第三轴不在其中，且在原理上不能在其中。**"""
    return {
        "format": "qiansui-genome-1",
        "plus_a": g.homolog_a.plus,
        "plus_b": g.homolog_b.plus,
        "species": species_barcode(g.homolog_a.plus, g.layout),
        "individual": individual_id(g),
        "field_order": [n for n, _k, _w, _c in g.layout],
    }


def verify_backup(blob: dict, layout: list[Field]) -> tuple[bool, list[str]]:
    """**备份校验**：长度、校验和、物种、个体身份，四项全对才算这份文件是她的。"""
    bad: list[str] = []
    g = restore(blob, layout)
    for tag, seq in (("a", g.homolog_a.plus), ("b", g.homolog_b.plus)):
        if not verify_checksum(seq, layout):
            bad.append(f"同源 {tag} 校验和不符（双链一致损伤，配对查不出来，只有校验和能查）")
    if species_barcode(g.homolog_a.plus, layout) != blob.get("species"):
        bad.append("物种条码与存档不符")
    if individual_id(g) != blob.get("individual"):
        bad.append("个体身份与存档不符（这份备份不是她的）")
    return (not bad), bad


def restore(blob: dict, layout: list[Field]) -> Genome:
    """从备份恢复。**恢复的是「同基因组的个体」，不是「同一个体」。**"""
    for key in ("plus_a", "plus_b"):
        if len(blob[key]) < min_length(layout):
            raise ValueError(f"{key} 太短：{len(blob[key])} < 最短 {min_length(layout)}")
    return Genome(Chromosome.build(blob["plus_a"]), Chromosome.build(blob["plus_b"]), layout)
