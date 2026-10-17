"""第四轮：表示网格——线性上界与局部规则各自能到哪。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/retrieval_representation_prereg.md`](../../sweep_results/retrieval_representation_prereg.md)。

网格（先写死）：表示 ∈ {`bus@128`, `bus@256`, `bus@512`, `plugin`} × 联想 ∈ {`ridge`, `delta`}。

⚠️ **配置只能按训练集交叉验证选**。8 个配置里挑测试集最高的那个，等于在测试集上做选择，
报告出来的 50% 就不可比了。所以这里同时记录：

* `train_cv`——该配置的联想超参在**训练集**上的交叉验证分数（选配置只看它）；
* `test_r1`——测试集上的结果（只报告，不参与选择）。
"""

from __future__ import annotations

import json

from biosnn_bus.plugins.text_zscore import OllamaEmbedding

from research.common.provenance import collect

from .encode import CachedEmbedding
from .run_retrieval import DEFAULT_CACHE, REPO_ROOT, run_seed

OUT = REPO_ROOT / "sweep_results" / "retrieval_representation"

CONFIGS = [
    ("bus@128", 128, "bus"),
    ("bus@256", 256, "bus"),
    ("bus@512", 512, "bus"),
    ("plugin", 128, "plugin"),
]
ASSOCIATORS = ("ridge", "delta")


def main() -> int:
    cache = CachedEmbedding(OllamaEmbedding(), DEFAULT_CACHE, model="nomic-embed-text")
    results: list[dict] = []

    for associator in ASSOCIATORS:
        for label, bus_dim, representation in CONFIGS:
            per_seed = []
            for seed in range(5):
                record = run_seed(
                    seed=seed,
                    bus_dim=bus_dim,
                    test_fraction=0.2,
                    backend_factory=lambda: cache,
                    associator_kind=associator,
                    representation=representation,
                    verbose=False,
                )
                per_seed.append(record)
            train_cv = sum(
                (r["selection_best_r1"]["text2image"] + r["selection_best_r1"]["image2text"]) / 2
                for r in per_seed
            ) / len(per_seed)
            test_r1 = sum(r["bus_associated"]["mean_r1"] for r in per_seed) / len(per_seed)
            test_r5 = sum(r["bus_associated"]["mean_r5"] for r in per_seed) / len(per_seed)
            entry = {
                "associator": associator,
                "representation": label,
                "bus_dim": bus_dim if representation == "bus" else None,
                "train_cv": train_cv,
                "test_r1": test_r1,
                "test_r5": test_r5,
            }
            results.append(entry)
            print(
                f"{associator:6s} {label:8s} train_cv={train_cv:.3f}  "
                f"test R@1={test_r1:.3f}  R@5={test_r5:.3f}"
            )

    def pick(associator: str) -> dict:
        pool = [r for r in results if r["associator"] == associator]
        return max(pool, key=lambda r: r["train_cv"])

    summary = {
        "experiment": "cross-modal retrieval: representation grid",
        "configs": results,
        "selected_by_train_cv": {assoc: pick(assoc) for assoc in ASSOCIATORS},
        "repro": collect(
            "检索线 W4：表示网格（第四轮）",
            seeds="0,1,2,3,4",
            notes=["配置按训练集 CV 选，测试集只报告", "ridge 是线性上界，非局部规则"],
        ).render_block(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("\n--- 按训练集 CV 选出的配置 ---")
    for assoc, entry in summary["selected_by_train_cv"].items():
        print(
            f"{assoc:6s} → {entry['representation']:8s} "
            f"(train_cv {entry['train_cv']:.3f}, test R@1 {entry['test_r1']:.3f})"
        )
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
