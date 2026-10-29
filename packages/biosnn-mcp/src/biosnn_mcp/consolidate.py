"""阶段五 M3d：睡眠期噪声回放的巩固协议（纯 stdlib，协议层）。

计划书 §七 第五阶段任务 4 要求「LLM 输出的脉冲编码回注和记忆巩固（**含睡眠期噪声巩固**）」。
本模块只实现**巩固协议**：

1. 睡眠期对每个条目的嵌入做高斯噪声回放（``rng.gauss``，种子固定 → 可复现）；
2. 回放后两两算余弦相似度，``≥ merge_threshold`` 的条目**合并**（保留先出现的 id 与文本，
   嵌入取噪声回放后的均值）；
3. 重复 ``rounds`` 轮，直到没有合并或轮数用尽；
4. 返回合并后的条目与报告（合并对、轮数、平均余弦漂移）。

**边界**：它不连接真实记忆库、不做语义改写、不生成新知识；噪声是各向同性的，
合并判据是余弦阈值。这些都要写进结果 README，不能把它写成「睡眠巩固能力」。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["ConsolidationReport", "MemoryItem", "SleepConsolidator"]


@dataclass(frozen=True)
class MemoryItem:
    """一条待巩固的记忆：id、文本、嵌入。"""

    id: str
    text: str
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id 不能为空。")
        if not self.embedding:
            raise ValueError("embedding 不能为空。")
        if not all(math.isfinite(value) for value in self.embedding):
            raise ValueError("embedding 含非有限值。")


@dataclass
class ConsolidationReport:
    """巩固报告。

    Attributes:
        merged_pairs: 每轮合并掉的 (保留 id, 被并入 id)。
        rounds_used: 实际执行的轮数。
        mean_cosine_drift: 噪声回放前后嵌入的余弦相似度均值（衡量回放扰动）。
        original_count / final_count: 输入与输出的条目数。
    """

    merged_pairs: list[tuple[str, str]] = field(default_factory=list)
    rounds_used: int = 0
    mean_cosine_drift: float = 0.0
    original_count: int = 0
    final_count: int = 0


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0.0:
        return 0.0
    return dot / norm


class SleepConsolidator:
    """睡眠期噪声回放 + 余弦合并的协议实现。

    Args:
        noise_sigma: 每个维度的噪声标准差（相对嵌入的 L2 范数归一化后施加）。
        merge_threshold: 合并阈值，余弦 ≥ 该值合并（应在 [-1, 1]）。
        rounds: 最多巩固轮数（≥1）。
        seed: 随机种子；同种子 → 同结果。
    """

    def __init__(
        self,
        *,
        noise_sigma: float = 0.05,
        merge_threshold: float = 0.95,
        merge_margin: float = 0.0,
        rounds: int = 3,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= noise_sigma <= 1.0:
            raise ValueError(f"noise_sigma 应在 [0, 1]，收到 {noise_sigma!r}。")
        if not -1.0 <= merge_threshold <= 1.0:
            raise ValueError(f"merge_threshold 应在 [-1, 1]，收到 {merge_threshold!r}。")
        if not 0.0 <= merge_margin <= 2.0:
            raise ValueError(f"merge_margin 应在 [0, 2]，收到 {merge_margin!r}。")
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
            raise ValueError(f"rounds 必须是 ≥1 的整数，收到 {rounds!r}。")
        self.noise_sigma = float(noise_sigma)
        self.merge_threshold = float(merge_threshold)
        # M3f：并入前还要求「最近邻与次近邻的余弦间隔 ≥ margin」，0 = 历史行为逐位不变。
        self.merge_margin = float(merge_margin)
        self.rounds = rounds
        self.seed = int(seed)

    def _replay(self, item: MemoryItem, rng: random.Random) -> tuple[float, ...]:
        """对嵌入加各向同性噪声并重新归一化。"""
        norm = math.sqrt(sum(value * value for value in item.embedding))
        if norm == 0.0:
            return item.embedding
        # 每维标准差 = σ·‖x‖/√dim，使**总**噪声范数 ≈ σ·‖x‖（逐维用 σ·‖x‖ 会让
        # 总噪声随 √dim 放大：768 维时达到信号的 1.4 倍，回放余弦被压到 0.58）。
        scale = self.noise_sigma * norm / math.sqrt(len(item.embedding))
        noisy = [value + rng.gauss(0.0, scale) for value in item.embedding]
        noisy_norm = math.sqrt(sum(value * value for value in noisy))
        if noisy_norm == 0.0:
            return item.embedding
        return tuple(value / noisy_norm * norm for value in noisy)

    def consolidate(
        self, items: Sequence[MemoryItem]
    ) -> tuple[list[MemoryItem], ConsolidationReport]:
        """执行睡眠巩固，返回（合并后的条目，报告）。"""
        for item in items:
            if not isinstance(item, MemoryItem):
                raise TypeError(f"条目必须是 MemoryItem，收到 {type(item)!r}。")
        ids = [item.id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("条目 id 必须唯一。")

        rng = random.Random(self.seed)
        report = ConsolidationReport(original_count=len(items))
        current = list(items)
        drifts: list[float] = []

        for round_index in range(1, self.rounds + 1):
            replayed = [self._replay(item, rng) for item in current]
            drifts.extend(
                _cosine(item.embedding, noisy)
                for item, noisy in zip(current, replayed, strict=True)
            )
            survivors: list[MemoryItem] = []
            merged_this_round = 0
            for item, noisy in zip(current, replayed, strict=True):
                merged = False
                if self.merge_margin > 0 and survivors:
                    # 先选最近邻，再检查它与次近邻的间隔；间隔不足就不合并。
                    scored = sorted(
                        (
                            (_cosine(noisy, survivor.embedding), index)
                            for index, survivor in enumerate(survivors)
                        ),
                        reverse=True,
                    )
                    best_score, best_index = scored[0]
                    margin_ok = (
                        len(scored) == 1 or best_score - scored[1][0] >= self.merge_margin
                    )
                    if best_score >= self.merge_threshold and margin_ok:
                        merged = True
                        merged_this_round += 1
                        survivor = survivors[best_index]
                        report.merged_pairs.append((survivor.id, item.id))
                        survivors[best_index] = MemoryItem(
                            id=survivor.id,
                            text=survivor.text,
                            embedding=tuple(
                                (a + b) / 2.0
                                for a, b in zip(survivor.embedding, noisy, strict=True)
                            ),
                        )
                    if not merged:
                        survivors.append(MemoryItem(id=item.id, text=item.text, embedding=noisy))
                    continue
                for index, survivor in enumerate(survivors):
                    if _cosine(noisy, survivor.embedding) >= self.merge_threshold:
                        merged = True
                        merged_this_round += 1
                        report.merged_pairs.append((survivor.id, item.id))
                        survivors[index] = MemoryItem(
                            id=survivor.id,
                            text=survivor.text,
                            embedding=tuple(
                                (a + b) / 2.0
                                for a, b in zip(survivor.embedding, noisy, strict=True)
                            ),
                        )
                        break
                if not merged:
                    survivors.append(MemoryItem(id=item.id, text=item.text, embedding=noisy))
            current = survivors
            report.rounds_used = round_index
            if merged_this_round == 0:
                break

        report.final_count = len(current)
        report.mean_cosine_drift = sum(drifts) / len(drifts) if drifts else 1.0
        return current, report
