"""跨模态联想：一层外积 Hebbian 映射，把一路的表示翻到另一路的空间里。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么必须有这一层

`biosnn-bus` 的两个通道各自经过**独立种子**的随机投影，所以「只有文本」与「只有图像」
的总线输出虽然在同一个 `bus_dim` 维空间里，却是两个互不相干的线性像。直接算余弦，
期望为 0——预注册里把这条写在最前面，免得跑出随机水平被误读成编码坏了。

这一层就是那条「纯局部的联想规则」：**外积 Hebb 学习，没有梯度、没有反向传播**。

```text
W = η · Σ_i  v_i ⊗ t_i          # t_i 文本侧、v_i 图像侧
```

推理时 `文本 → W·t / (1 + λ)`，再与图像侧表示比余弦。反方向用**另一个** W′，
不假设 W 与 W′ 互为转置——局部规则没有理由保证这件事。

## 两个超参怎么定

`η` 与 `λ` **只在训练集上选**（留一法），测试集不参与任何选择。留一法是**精确**的，
不是近似：外积和是线性的，`W_{-i} = W − η·v_i⊗t_i` 一步就能减出来，不必重训。

⚠️ **`η` 单独没有意义**：余弦对全局缩放不敏感，`η` 只有相对 `1/(1+λ)` 才有作用。
所以网格给的是「η × λ」的组合，报的是组合，不是 η 一个数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import recall_at_k

__all__ = ["HebbianAssociator", "select_hyperparameters"]

#: 选参网格。**先写死**，不在看到测试集之后扩大（那就是过拟合到测试集）。
ETA_GRID = (0.001, 0.01, 0.1, 1.0)
DECAY_GRID = (0.0, 0.1, 1.0)


@dataclass
class HebbianAssociator:
    """外积 Hebb 联想：``source → target``。"""

    eta: float = 0.01
    decay: float = 0.0
    weight: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> HebbianAssociator:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source 与 target 都必须是二维 (n, d)。")
        if source.shape[0] != target.shape[0]:
            raise ValueError(f"样本数必须相同，收到 {source.shape[0]} 与 {target.shape[0]}。")
        self.weight = self.eta * (target.T @ source)
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("先 fit() 再 transform()。")
        source = np.asarray(source, dtype=np.float64)
        return (source @ self.weight.T) / (1.0 + self.decay)

    def __repr__(self) -> str:
        shape = None if self.weight is None else self.weight.shape
        return f"HebbianAssociator(eta={self.eta}, decay={self.decay}, weight={shape})"


def _loo_transform(
    source: np.ndarray, target: np.ndarray, index: int, eta: float, decay: float
) -> np.ndarray:
    """第 ``index`` 条样本在**去掉自己**的 W 下的变换结果。"""
    full = eta * (target.T @ source)  # (dt, ds)
    own = eta * np.outer(target[index], source[index])
    reduced = full - own
    return (source[index] @ reduced.T) / (1.0 + decay)


def select_hyperparameters(
    source: np.ndarray,
    target: np.ndarray,
    *,
    eta_grid: tuple[float, ...] = ETA_GRID,
    decay_grid: tuple[float, ...] = DECAY_GRID,
) -> tuple[float, float, dict[tuple[float, float], float]]:
    """在训练集上做留一法选 ``(eta, decay)``，返回最优组合与全网格分数。

    ⚠️ **留一法在这里有它的上限**：训练样本同时是"记忆对象"和"泛化证据"。
    如果最优解是记住训练集（`eta` 很大、`decay` 很小），留一法会看出来——被留出的那一条
    不在和里——但它看不出"这个映射换一个数据集还成不成立"。所以测试集那一栏是必须的，
    不能拿训练集的留一分数替代。
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    scores: dict[tuple[float, float], float] = {}

    for eta in eta_grid:
        for decay in decay_grid:
            hits = 0
            for i in range(n):
                predicted = _loo_transform(source, target, i, eta, decay)
                similarity = (target @ predicted) / (
                    np.linalg.norm(target, axis=1) * np.linalg.norm(predicted) + 1e-12
                )
                order = np.argsort(-similarity, kind="stable")
                rank = int(np.where(order == i)[0][0]) + 1
                hits += rank <= 1
            scores[(eta, decay)] = hits / n

    best = max(scores, key=lambda key: (scores[key], -key[0]))
    return best[0], best[1], scores


def evaluate_direction(
    associator: HebbianAssociator | DeltaAssociator,
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    direction: str,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """把查询翻到候选空间，算 R@K。"""
    predicted = associator.transform(queries)
    return {f"r{k}": recall_at_k(predicted, gallery, k=k, direction=direction).recall for k in ks}


# --------------------------------------------------------------------------
# 第二轮：Widrow–Hoff delta 规则
# --------------------------------------------------------------------------

#: 第二轮的选参网格（预注册写死）。
DELTA_ETA_GRID = (0.001, 0.01, 0.1)
DELTA_EPOCH_GRID = (1, 5, 20)


@dataclass
class DeltaAssociator:
    """带误差修正的局部规则：``W ← W + η(v − W t) tᵀ``，逐样本。

    ⚠️ **它需要教师信号**：误差项里的 ``v`` 是配对训练时给出的目标。所以它是
    **有教师的局部规则**，不是无监督 Hebb，也不等于反向传播——每个突触只用自己
    两端的前/后活动加一个误差信号，不经过链式法则。
    """

    eta: float = 0.01
    epochs: int = 5
    seed: int = 0
    weight: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> DeltaAssociator:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source 与 target 都必须是二维 (n, d)。")
        if source.shape[0] != target.shape[0]:
            raise ValueError(f"样本数必须相同，收到 {source.shape[0]} 与 {target.shape[0]}。")
        if self.epochs < 1:
            raise ValueError(f"epochs 至少为 1，收到 {self.epochs}。")

        weight = np.zeros((target.shape[1], source.shape[1]), dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        diverged = False
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(self.epochs):
                for index in rng.permutation(source.shape[0]):
                    t = source[index]
                    error = target[index] - weight @ t
                    weight += self.eta * np.outer(error, t)
                if not np.all(np.isfinite(weight)):
                    diverged = True
                    break
        self.weight = weight
        self.diverged = diverged
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("先 fit() 再 transform()。")
        return np.asarray(source, dtype=np.float64) @ self.weight.T

    def __repr__(self) -> str:
        shape = None if self.weight is None else self.weight.shape
        return f"DeltaAssociator(eta={self.eta}, epochs={self.epochs}, weight={shape})"


def select_delta_hyperparameters(
    source: np.ndarray,
    target: np.ndarray,
    *,
    eta_grid: tuple[float, ...] = DELTA_ETA_GRID,
    epoch_grid: tuple[int, ...] = DELTA_EPOCH_GRID,
    folds: int = 5,
    seed: int = 0,
) -> tuple[float, int, dict[tuple[float, int], float]]:
    """在训练集上做 **k 折交叉验证**选 ``(eta, epochs)``。

    ⚠️ 这里**不能用留一法**：delta 是迭代的，去掉一条样本后的解不是"整体解减去那一项"
    （外积 Hebb 才是那样）。所以退回 k 折，并在预注册里写明。
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    if folds < 2 or folds > n:
        raise ValueError(f"folds 必须落在 [2, {n}]，收到 {folds}。")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    chunks = [chunk for chunk in np.array_split(order, folds) if len(chunk) > 0]

    scores: dict[tuple[float, int], float] = {}
    for eta in eta_grid:
        for epochs in epoch_grid:
            hits = 0
            total = 0
            for fold in chunks:
                held = {int(i) for i in fold}
                train_idx = [i for i in range(n) if i not in held]
                model = DeltaAssociator(eta=eta, epochs=epochs, seed=seed).fit(
                    source[train_idx], target[train_idx]
                )
                if model.diverged:
                    # 发散 = 这个组合在这折上无效：记 0 分，而不是让它把平均值变成 NaN
                    total += len(held)
                    continue
                predicted = model.transform(source[list(held)])
                scores_fold = recall_at_k(predicted, target[list(held)], k=1, direction="cv").recall
                hits += scores_fold * len(held)
                total += len(held)
            scores[(eta, epochs)] = hits / total

    best = max(scores, key=lambda key: (scores[key], -key[0], -key[1]))
    return best[0], best[1], scores


# --------------------------------------------------------------------------
# 第三轮：岭回归上界探针（**不是**局部规则，不作验收）
# --------------------------------------------------------------------------

#: 第三轮的 λ 网格（预注册写死）。
RIDGE_LAMBDA_GRID = (0.001, 0.1, 1.0, 10.0)


@dataclass
class RidgeAssociator:
    """闭式岭回归：``W = (TᵀT + λI)⁻¹ Tᵀ V``。

    ⚠️ **这不是局部规则，不能用于验收。** 它一次性看整个训练集、要矩阵求逆。
    存在的唯一理由是当**上界探针**：它衡量两条总线表示之间**线性映射**的能力上限，
    用来判断"瓶颈在规则还是在表示"。
    """

    lam: float = 1.0
    weight: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> RidgeAssociator:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source 与 target 都必须是二维 (n, d)。")
        if source.shape[0] != target.shape[0]:
            raise ValueError(f"样本数必须相同，收到 {source.shape[0]} 与 {target.shape[0]}。")
        if self.lam < 0:
            raise ValueError(f"lam 必须非负，收到 {self.lam}。")

        gram = source.T @ source + self.lam * np.eye(source.shape[1])
        self.weight = np.linalg.solve(gram, source.T @ target).T
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("先 fit() 再 transform()。")
        return np.asarray(source, dtype=np.float64) @ self.weight.T

    def __repr__(self) -> str:
        shape = None if self.weight is None else self.weight.shape
        return f"RidgeAssociator(lam={self.lam}, weight={shape})"


def select_ridge_lambda(
    source: np.ndarray,
    target: np.ndarray,
    *,
    lambda_grid: tuple[float, ...] = RIDGE_LAMBDA_GRID,
    folds: int = 5,
    seed: int = 0,
) -> tuple[float, dict[float, float]]:
    """训练集 k 折交叉验证选 λ。"""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    if folds < 2 or folds > n:
        raise ValueError(f"folds 必须落在 [2, {n}]，收到 {folds}。")

    rng = np.random.default_rng(seed)
    chunks = [chunk for chunk in np.array_split(rng.permutation(n), folds) if len(chunk) > 0]

    scores: dict[float, float] = {}
    for lam in lambda_grid:
        hits = 0.0
        total = 0
        for fold in chunks:
            held = {int(i) for i in fold}
            train_idx = [i for i in range(n) if i not in held]
            model = RidgeAssociator(lam=lam).fit(source[train_idx], target[train_idx])
            predicted = model.transform(source[list(held)])
            hits += recall_at_k(predicted, target[list(held)], k=1, direction="cv").recall * len(
                held
            )
            total += len(held)
        scores[lam] = hits / total

    best = max(scores, key=lambda key: (scores[key], -key))
    return best, scores


# --------------------------------------------------------------------------
# 第五轮：k 近邻非线性上界探针（**不是**局部学习规则，不作验收）
# --------------------------------------------------------------------------

#: 第五轮的 k 网格（预注册写死）。
KNN_K_GRID = (1, 3, 5, 10, 20)


@dataclass
class KNNAssociator:
    """k 近邻：记住训练对，查询时按余弦相似度投票。

    ⚠️ **这不是局部学习规则**：没有突触、没有更新。它只作**非线性上界探针**，
    用来区分"信息在、只是非线性地摆着"与"编码里根本没有"。
    """

    k: int = 5
    source: np.ndarray | None = None
    target: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> KNNAssociator:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source 与 target 都必须是二维 (n, d)。")
        if source.shape[0] != target.shape[0]:
            raise ValueError(f"样本数必须相同，收到 {source.shape[0]} 与 {target.shape[0]}。")
        if not 1 <= self.k <= source.shape[0]:
            raise ValueError(f"k 必须落在 [1, {source.shape[0]}]，收到 {self.k}。")
        self.source = source
        self.target = target
        return self

    def transform(self, queries: np.ndarray) -> np.ndarray:
        if self.source is None or self.target is None:
            raise RuntimeError("先 fit() 再 transform()。")
        queries = np.asarray(queries, dtype=np.float64)
        qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
        sn = self.source / (np.linalg.norm(self.source, axis=1, keepdims=True) + 1e-12)
        similarity = qn @ sn.T
        index = np.argsort(-similarity, axis=1, kind="stable")[:, : self.k]
        weights = np.take_along_axis(similarity, index, axis=1).clip(min=0.0)
        totals = weights.sum(axis=1, keepdims=True)
        # 全负相似度时退化为等权，避免除以 0
        weights = np.where(totals > 0, weights / (totals + 1e-12), 1.0 / self.k)
        return (weights[:, :, None] * self.target[index]).sum(axis=1)


def select_knn_k(
    source: np.ndarray,
    target: np.ndarray,
    *,
    k_grid: tuple[int, ...] = KNN_K_GRID,
    folds: int = 5,
    seed: int = 0,
) -> tuple[int, dict[int, float]]:
    """训练集 k 折交叉验证选 k。"""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    rng = np.random.default_rng(seed)
    chunks = [chunk for chunk in np.array_split(rng.permutation(n), folds) if len(chunk) > 0]

    scores: dict[int, float] = {}
    for k in k_grid:
        hits = 0.0
        total = 0
        for fold in chunks:
            held = {int(i) for i in fold}
            train_idx = [i for i in range(n) if i not in held]
            model = KNNAssociator(k=min(k, len(train_idx))).fit(
                source[train_idx], target[train_idx]
            )
            predicted = model.transform(source[list(held)])
            hits += recall_at_k(predicted, target[list(held)], k=1, direction="cv").recall * len(
                held
            )
            total += len(held)
        scores[k] = hits / total

    best = max(scores, key=lambda key: (scores[key], -key))
    return best, scores


# --------------------------------------------------------------------------
# 第六轮：固定随机特征 + delta 读出
# --------------------------------------------------------------------------

#: 第六轮的网格（预注册写死）。
RF_FEATURE_GRID = (256, 512)
RF_ETA_GRID = (0.001, 0.01, 0.1)
RF_EPOCH_GRID = (5, 20)


@dataclass
class RandomFeaturesAssociator:
    """``φ(x) = ReLU(P·x)``，``P`` 固定随机；再在 φ 上跑 delta 规则。

    **被学习的只有读出的 W**。``P`` 不训练、没有梯度穿过 φ——这是"固定随机投影"
    与"局部读出"的组合，不是反向传播。
    """

    n_features: int = 256
    eta: float = 0.01
    epochs: int = 5
    seed: int = 0
    projection: np.ndarray | None = None
    weight: np.ndarray | None = None
    diverged: bool = False

    def _phi(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x @ self.projection.T, 0.0)

    def fit(self, source: np.ndarray, target: np.ndarray) -> RandomFeaturesAssociator:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source 与 target 都必须是二维 (n, d)。")
        if source.shape[0] != target.shape[0]:
            raise ValueError(f"样本数必须相同，收到 {source.shape[0]} 与 {target.shape[0]}。")

        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(size=(self.n_features, source.shape[1])) / np.sqrt(
            source.shape[1]
        )
        inner = DeltaAssociator(eta=self.eta, epochs=self.epochs, seed=self.seed).fit(
            self._phi(source), target
        )
        self.weight = inner.weight
        self.diverged = inner.diverged
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        if self.projection is None or self.weight is None:
            raise RuntimeError("先 fit() 再 transform()。")
        return self._phi(np.asarray(source, dtype=np.float64)) @ self.weight.T


def select_random_features(
    source: np.ndarray,
    target: np.ndarray,
    *,
    feature_grid: tuple[int, ...] = RF_FEATURE_GRID,
    eta_grid: tuple[float, ...] = RF_ETA_GRID,
    epoch_grid: tuple[int, ...] = RF_EPOCH_GRID,
    folds: int = 5,
    seed: int = 0,
) -> tuple[tuple[int, float, int], dict[tuple[int, float, int], float]]:
    """训练集 k 折交叉验证选 ``(n_features, eta, epochs)``。"""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    rng = np.random.default_rng(seed)
    chunks = [chunk for chunk in np.array_split(rng.permutation(n), folds) if len(chunk) > 0]

    scores: dict[tuple[int, float, int], float] = {}
    for n_features in feature_grid:
        for eta in eta_grid:
            for epochs in epoch_grid:
                hits = 0.0
                total = 0
                for fold in chunks:
                    held = {int(i) for i in fold}
                    train_idx = [i for i in range(n) if i not in held]
                    model = RandomFeaturesAssociator(
                        n_features=n_features, eta=eta, epochs=epochs, seed=seed
                    ).fit(source[train_idx], target[train_idx])
                    if model.diverged:
                        total += len(held)
                        continue
                    predicted = model.transform(source[list(held)])
                    hits += recall_at_k(
                        predicted, target[list(held)], k=1, direction="cv"
                    ).recall * len(held)
                    total += len(held)
                scores[(n_features, eta, epochs)] = hits / total

    best = max(scores, key=lambda key: (scores[key], -key[0], -key[1], -key[2]))
    return best, scores


# --------------------------------------------------------------------------
# 第七轮：竞争联想记忆（赢者通吃，纯局部）
# --------------------------------------------------------------------------

#: 第七轮的网格（预注册写死）。
COMPETITIVE_UNITS_GRID = (16, 32, 64)
COMPETITIVE_EPOCH_GRID = (20, 100)
COMPETITIVE_ETA_GRID = (0.05, 0.2)
COMPETITIVE_K_GRID = (1, 3)


@dataclass
class CompetitiveAssociator:
    """竞争联想记忆：M 个单元，每个持有 (文本原型, 图像码)，赢者通吃更新。

    **纯局部**：一个单元的更新只用到它自己的原型与它收到的输入，没有梯度、
    没有反向传播、没有全局误差反传。

    ⚠️ 单元数 M 若接近训练样本数，它就退化成"把训练集存起来"。报告时必须同时给出
    M 与训练样本数之比，否则读者无法判断这是学习还是记忆。
    """

    n_units: int = 32
    epochs: int = 20
    eta: float = 0.05
    k: int = 1
    seed: int = 0
    prototypes: np.ndarray | None = None
    codes: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> CompetitiveAssociator:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source 与 target 都必须是二维 (n, d)。")
        if source.shape[0] != target.shape[0]:
            raise ValueError(f"样本数必须相同，收到 {source.shape[0]} 与 {target.shape[0]}。")

        rng = np.random.default_rng(self.seed)
        prototypes = rng.normal(size=(self.n_units, source.shape[1])) * 0.01
        codes = rng.normal(size=(self.n_units, target.shape[1])) * 0.01

        p_norm = np.linalg.norm(prototypes, axis=1) + 1e-12
        for _ in range(self.epochs):
            for index in rng.permutation(source.shape[0]):
                t = source[index]
                similarity = (prototypes @ t) / (p_norm * (np.linalg.norm(t) + 1e-12))
                winner = int(np.argmax(similarity))
                prototypes[winner] += self.eta * (t - prototypes[winner])
                codes[winner] += self.eta * (target[index] - codes[winner])
                p_norm[winner] = np.linalg.norm(prototypes[winner]) + 1e-12

        self.prototypes = prototypes
        self.codes = codes
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        if self.prototypes is None or self.codes is None:
            raise RuntimeError("先 fit() 再 transform()。")
        source = np.asarray(source, dtype=np.float64)
        pn = self.prototypes / (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-12)
        qn = (source / (np.linalg.norm(source, axis=1, keepdims=True) + 1e-12)) @ pn.T
        index = np.argsort(-qn, axis=1, kind="stable")[:, : self.k]
        weights = np.take_along_axis(qn, index, axis=1).clip(min=0.0)
        totals = weights.sum(axis=1, keepdims=True)
        weights = np.where(totals > 0, weights / (totals + 1e-12), 1.0 / self.k)
        return (weights[:, :, None] * self.codes[index]).sum(axis=1)


def select_competitive(
    source: np.ndarray,
    target: np.ndarray,
    *,
    unit_grid: tuple[int, ...] = COMPETITIVE_UNITS_GRID,
    epoch_grid: tuple[int, ...] = COMPETITIVE_EPOCH_GRID,
    eta_grid: tuple[float, ...] = COMPETITIVE_ETA_GRID,
    k_grid: tuple[int, ...] = COMPETITIVE_K_GRID,
    folds: int = 5,
    seed: int = 0,
) -> tuple[tuple[int, int, float, int], dict[tuple[int, int, float, int], float]]:
    """训练集 k 折交叉验证选 ``(M, epochs, eta, k)``。"""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = source.shape[0]
    rng = np.random.default_rng(seed)
    chunks = [chunk for chunk in np.array_split(rng.permutation(n), folds) if len(chunk) > 0]

    scores: dict[tuple[int, int, float, int], float] = {}
    for units in unit_grid:
        for epochs in epoch_grid:
            for eta in eta_grid:
                for k in k_grid:
                    hits = 0.0
                    total = 0
                    for fold in chunks:
                        held = {int(i) for i in fold}
                        train_idx = [i for i in range(n) if i not in held]
                        model = CompetitiveAssociator(
                            n_units=min(units, len(train_idx)),
                            epochs=epochs,
                            eta=eta,
                            k=min(k, units),
                            seed=seed,
                        ).fit(source[train_idx], target[train_idx])
                        predicted = model.transform(source[list(held)])
                        hits += recall_at_k(
                            predicted, target[list(held)], k=1, direction="cv"
                        ).recall * len(held)
                        total += len(held)
                    scores[(units, epochs, eta, k)] = hits / total

    best = max(scores, key=lambda key: (scores[key], -key[0], -key[1], -key[2], -key[3]))
    return best, scores
