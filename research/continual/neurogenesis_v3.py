"""神经发生 v3：回放（v3-A）、竞争护栏（v3-B）与两者并用的 NCM 读出。

预注册见 `sweep_results/neurogenesis_v3_prereg.md`。相对 v2 只动两件事，且都可单独关闭：

* **回放**：生长时旧类的原型在新维度上**补零是不对的**——旧样本过新的随机投影会激活新维度。
  所以每个见过的类留 `buffer_per_class` 个原始样本，生长后只对**新增维度**用回放样本重估
  （旧维度的精确和保持不变）。这是"用少量显式样本修一个结构性错误"，不是全量重训；
* **竞争护栏**：预测时把**本任务新出现的类**的距离乘以 `1 + δ`，让它们必须明显更好才能抢走
  样本；δ 只用**旧任务验证集 ∪ 当前任务训练集切出的验证片**选（**不看当前任务的测试集**）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.continual.neurogenesis import GrowthController, GrowthDecision

__all__ = ["MARGIN_GRID", "ReplayNCMReadout"]

#: 竞争护栏 δ 的网格（预注册写死）。
MARGIN_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)


@dataclass
class ArmStats:
    """逐任务的记录（落盘用）。"""

    decisions: list[GrowthDecision] = field(default_factory=list)
    chosen_margins: list[float] = field(default_factory=list)


class ReplayNCMReadout:
    """NCM + 可选回放 + 可选竞争护栏。

    Args:
        n_in: 输入维数。
        n_classes: 总类数。
        n0: 初始特征数（默认 64，与 v1/v2 同口径）。
        growth: 每次增长的维度（默认 16）。
        max_ratio: 特征上限倍数（默认 2.0）。
        min_accuracy: 触发增长的训练准确率（默认 0.90）。
        buffer_per_class: 每类回放缓冲大小（预注册写死 20）。
        use_replay: 是否在生长后修复旧类原型的新维度。
        use_margin: 是否启用竞争护栏（用 `choose_margin` 选 δ）。
        seed: 随机种子。
    """

    def __init__(
        self,
        n_in: int,
        n_classes: int,
        *,
        n0: int = 64,
        growth: int = 16,
        max_ratio: float = 2.0,
        min_accuracy: float = 0.90,
        buffer_per_class: int = 20,
        use_replay: bool = False,
        use_margin: bool = False,
        seed: int = 0,
    ) -> None:
        if buffer_per_class <= 0:
            raise ValueError(f"buffer_per_class 必须为正，收到 {buffer_per_class!r}。")
        self.n_in = int(n_in)
        self.n_classes = int(n_classes)
        self.rng = np.random.default_rng(seed)
        self.n_features = int(n0)
        self.projection = self._projection(n0)
        self.sums = np.zeros((self.n_classes, self.n_features))
        self.counts = np.zeros(self.n_classes, dtype=np.int64)
        #: 每个类第一次出现在哪个任务（护栏只保护「本任务新出现的类」）。
        self.introduced = np.full(self.n_classes, -1, dtype=np.int64)
        self.controller = GrowthController(
            n0=n0, growth=growth, max_ratio=max_ratio, min_accuracy=min_accuracy
        )
        self.buffer_per_class = int(buffer_per_class)
        self.buffer: dict[int, np.ndarray] = {}
        self.use_replay = bool(use_replay)
        self.use_margin = bool(use_margin)
        self.margin = 0.0
        self.stats = ArmStats()

    # ------------------------------------------------------------------ 内部
    def _projection(self, n: int) -> np.ndarray:
        return self.rng.normal(size=(self.n_in, n)) / np.sqrt(self.n_in)

    def _features(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(x, dtype=np.float64) @ self.projection, 0.0)

    def _distances(self, features: np.ndarray) -> np.ndarray:
        means = self.sums / np.maximum(self.counts[:, None], 1)
        distances = ((features[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
        distances[:, self.counts == 0] = np.inf
        return distances

    def _repair_grown_dims(self, added: int) -> None:
        """生长后：把旧类**新增维度**的原型用回放样本重估（旧维度不动）。"""
        if not self.use_replay or added <= 0:
            return
        sub_projection = self.projection[:, -added:]
        for label, samples in self.buffer.items():
            if self.counts[label] == 0:
                continue
            features = np.maximum(np.asarray(samples) @ sub_projection, 0.0)
            estimate = features.mean(axis=0) * float(self.counts[label])
            self.sums[label, -added:] = estimate

    # ------------------------------------------------------------------ 对外
    def grow(self) -> None:
        added = self.controller.growth
        self.projection = np.hstack([self.projection, self._projection(added)])
        self.sums = np.hstack([self.sums, np.zeros((self.n_classes, added))])
        self.n_features += added
        self._repair_grown_dims(added)

    def fit_task(
        self, x: np.ndarray, y: np.ndarray, *, classes: np.ndarray, task: int = 0
    ) -> float:
        """写本任务的类原型，并按需更新回放缓冲。旧类的旧维度不动。"""
        f = self._features(x)
        labels = np.asarray(y)
        for label in classes:
            rows = labels == label
            self.sums[int(label)] = f[rows].sum(axis=0)
            self.counts[int(label)] = int(rows.sum())
            if self.introduced[int(label)] < 0:
                self.introduced[int(label)] = int(task)
            if self.use_replay:
                stored = np.asarray(x)[rows]
                if len(stored) > self.buffer_per_class:
                    index = self.rng.choice(len(stored), self.buffer_per_class, replace=False)
                    stored = stored[index]
                self.buffer[int(label)] = stored
        predicted = self.predict(x, task=task)
        return float(np.mean(predicted == labels))

    def predict(self, x: np.ndarray, *, task: int | None = None) -> np.ndarray:
        """预测；启用护栏时把「本任务新出现的类」的距离乘以 `1 + δ`。"""
        distances = self._distances(self._features(x))
        if self.use_margin and self.margin > 0.0 and task is not None:
            fresh = self.introduced == int(task)
            distances[:, fresh] *= 1.0 + self.margin
            distances[:, self.counts == 0] = np.inf
        return np.argmin(distances, axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray, *, task: int | None = None) -> float:
        return float(np.mean(self.predict(x, task=task) == np.asarray(y)))

    def choose_margin(
        self,
        validation: list[tuple[np.ndarray, np.ndarray]],
        *,
        task: int,
    ) -> float:
        """在给定验证片上挑 δ（旧任务验证集 ∪ 当前任务的训练验证片）。"""
        if not self.use_margin:
            self.margin = 0.0
            return 0.0
        best_margin, best_score = 0.0, -1.0
        for margin in MARGIN_GRID:
            self.margin = margin
            score = sum(
                float(np.mean(self.predict(x, task=task) == np.asarray(y))) for x, y in validation
            )
            if score > best_score + 1e-12:
                best_margin, best_score = margin, score
        self.margin = best_margin
        self.stats.chosen_margins.append(best_margin)
        return best_margin

    def maybe_grow(self, accuracy: float) -> GrowthDecision:
        decision = self.controller.decide(accuracy, self.n_features)
        self.stats.decisions.append(decision)
        if decision.grow:
            self.grow()
        return decision
