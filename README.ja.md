# BioSNN-Plug

**高い生物学的妥当性を志向した全モダリティ・スパイキングニューラルネットワーク認知プロトタイプ**

> **翻訳されている範囲について。** ドキュメントは中国語・英語・日本語で提供していますが、
> **コードは翻訳されていません**——docstring、コメント、エラーメッセージは中国語のみです。
> そのため非中国語話者がライブラリを使うと、中国語のテキストに触れることになります。
> また、この日本語版は機械支援による翻訳であり、母語話者によるレビューを受けていません。
> 不自然な箇所があれば issue でお知らせください。

[中文](README.md) · [English](README.en.md) · [プロジェクト計画書 v6.2](BioSNN-Plug_项目计划书_v6.2.md) · [v6.3（引用訂正版）](BioSNN-Plug_项目计划书_v6.3.md) · [計画書とリポジトリの乖離監査（中国語）](BioSNN-Plug_计划书差距审计.md) · [プラグイン開発ガイド](docs/plugin_guide.ja.md) · [アーキテクチャ決定記録](docs/adr/README.ja.md)

> **二つの版の関係**：**v6.2 が正式なプロジェクト計画書**です。**v6.3** は最初の引用考証による訂正を反映した版です（変更点は [`docs/references.ja.md`](docs/references.ja.md) の「計画書本文の訂正待ち項目」にあります）。引用は v6.2 を、訂正後の表現（Trace Propagation の位置づけ、TD-LTP の出典など）が必要なときは v6.3 を使ってください。

[![CI](https://github.com/lzmd-arch/BioSNN-Plug/actions/workflows/ci.yml/badge.svg)](https://github.com/lzmd-arch/BioSNN-Plug/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.ja.ipynb)

## これは何か

研究プロトタイプであり、一つの問いを検証する：**知性は純粋に局所的な学習則から育つことができ、外部推論を借りて自らの境界を拡張することを学べるか？**

三つの具体的な主張：

1. **代理勾配をスキップする**——シナプス更新はすべて局所信号によって駆動される（知覚層はカーネル化 IB-Hebbian、認知層は e-prop、実行層は R-STDP）；
2. **モダリティのプラグイン化**——テキスト／画像／音声を独立したプラグインとして共有の認知コアに接続し、モダリティを追加しても既存のアーキテクチャを変更しない；
3. **SNN が自ら LLM を呼び出す**——SNN が認知主体であり、「いつ行動するか、何を関連づけるか」を決める；LLM は交換可能な推論エンジンとして「何を生成するか」だけを担う。

## これは何ではないか

**プロダクションフレームワークではない。** SOTA を目指さず、可用性も約束せず、メンテナは 1 名で、issue に SLA を設けない。

**汎用の SNN ライブラリではない。** 性能は目標ではなく、価値は高リスクな路線が成立するかどうかの検証にある。

**インスピレーションではなく証拠である。** 結論は本リポジトリで再現できなければならない；検証できなかった引用には正直に `unverified` を付ける（[引用リスト](docs/references.ja.md) 参照）。

## 類似プロジェクトとの関係

一目でわかる対照表です。**各セルの根拠とライセンス境界（どの依存に入れられるか、どれは
一行も写せないか）は [`docs/related_projects.ja.md`](docs/related_projects.ja.md) にあります。**

| | 純粋な局所学習 | モダリティプラグイン | 自律トリガ | 継続学習 | オープンソース枠組み |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Javis** | ◐ すべて STDP ファミリ、逆伝播なし（12 種から選択）；注意：**自前の `iter-45` reward ベンチマークで両アームとも偶然水準を超えず**（README 原文「neither arm reaches above-chance accuracy」） | ✗ | ✗ | ✗ | ◐ Rust だが **PolyForm 非商用** |
| **EMBER** | ◐ STDP + 適格性痕跡 × ドーパミンゲーティング | ✗（テキストのみ） | ◐ 観察レベル、**N=1** | ✗ | ✗ 公開リポジトリなし |
| **MEMBRAIN** | ◐ Voja + PES | ✗ | ✗ | ◐ 主張と実装が一致しない | ✅ Nengo（MIT） |
| **eprop-PyTorch / ESPP** | ✅ e-prop；ESPP は別の並列な規則 | ✗ | ✗ | ✗ | ◐ コード断片であり枠組みではない |
| **BioSNN-Plug**（本プロジェクト） | ✅ 三本のラインすべて受入済み | ◐ プラグイン 3 本（画像 / MNIST / テキスト）、**音声は欠** | ○ 第 5 フェーズ | ○ 第 3 フェーズ | ✅ `biosnn-bus` は PyPI に公開済み |

凡例：**✅ 実装済みかつ受入データあり** · **◐ 部分**（機構または証拠が不十分） ·
**○ ロードマップのみ（計画書にタスクはあるが未着手）** · **✗ なし**

⚠️ **最終行の 2 セルは ✅ ではなく ○ です**——「自律トリガ」は第 5 フェーズ、「継続学習」は
第 3 フェーズにあり、**どちらも実装はまだ始まっていません**。記号を分けているのは、この表が
「もう達成した」と読まれないようにするためです。

## 現在の状態

| 部分 | 状態 |
| :--- | :--- |
| `biosnn-bus` スケルトンライブラリ | **0.2.0 利用可能**——プラグインインタフェース、レジストリとディスカバリ、スパイクバス、**TAAF 時間注意誘導融合**（rate-based gating：重みは各経路の発火率からその場で計算し、**逆伝播なし**） |
| モダリティプラグイン | **3 本**：`image_diff`（DVS イベント符号化）、`mnist_pulse`（発火率）、**`text_zscore`**（凍結したローカル embedding ＋ z-score top-k 群体符号化） |
| 研究コード：三つの検証ライン | **第 1 フェーズの実装完了**——W1 Hebbian 知覚 MNIST **97.79%**（既定の閉形式解の読み出し；SGD 読み出しは 98.01%）✓；W2 e-prop 系列 sMNIST **0.7764**（30 エポック × 5 シード）→ **0.8831**（250 エポック、**収束**）✓；W3 R-STDP CartPole 中央値 **237.8 歩**（基準 ≥ 200）✓。数値と再現記録は[再現記録](docs/reproducibility.ja.md) 参照 |
| **ネットワーク規模** | **スパイキングニューロン 256 個**（W2）。W1 には**レート型**ユニットが 3,072 個あります（スパイキングニューロンには数えません）。計画書 §2.2 の認知コア目標は約 5 万——**約 195 倍の差**。詳細は [research/README](research/README.ja.md) |
| 仲裁器 | **部品は揃った**（`research/arbiter/`）：衝突判定（コサイン > 90 度）＋ ES メタ学習仲裁器、**両方の研究ラインに接続済み**（共有の入力シナプス）。注意：共有されるのは入力→隠れ層の射影だけで、「衝突イベント 50% 減」は合成データ上では**自明に成立**します——当該ディレクトリの README 参照 |
| 跨モダリティ検索 | **R@K プロトコルは就位**（`research/retrieval/metrics.py`）、**データセットは未接続** |
| 認知コア / LLM 連携 | **未着手**、[プロジェクト計画書 §七](BioSNN-Plug_项目计划书_v6.2.md) 参照 |

**第二段階の成功基準の実態**：

| 基準 | 状態 |
| :--- | :--- |
| プラグインを独立に差し替えられる | ✅ 3 本のプラグイン ＋ entry points、スケルトンライブラリは PyPI に公開済み |
| TAAF の微調整サンプル数 < ゼロからの学習の 1/10 | ⚠️ **基準は [ADR-0012](docs/adr/ADR-0012-taaf-locality-and-bus-output-contract.ja.md) により消滅したのであって、達成ではない**——融合重みはレッドラインに従い一切学習しないため、この基準には**指す対象がありません**（微調整が無い以上、「微調整サンプル数」も無い）。計画書の改訂記録 27 参照 |
| 跨モダリティ検索 R@1 50%+ | ❌ **データセットとプロトコルの範囲が未定**（プロトコルのコードはある、データがない） |
| 仲裁器が双規則の衝突イベントを 50% 以上減らす | ✅ **実際の共有シナプス上で達成**（10 → 0、`retention` 0.3） |

明確にしておくべき境界が 3 つある：

- **TAAF は実装済みだが、既定の融合戦略は依然として連結**：`TemporalAttentionFusion` は**明示的に選ぶ**融合戦略（`SpikeBus(strategy=...)`）で、既定は単純連結のままです。注意：その注意重みは**各経路の発火率から計算し、学習しません**（rate-based gating）——このレッドラインは **AST による構造的アサーション**で固定しています。[ADR-0012](docs/adr/ADR-0012-taaf-locality-and-bus-output-contract.ja.md) 参照。
- **知覚層は凍結した事前学習済みフロントエンドを使ってよい**（境界宣言、[計画書 §2.3](BioSNN-Plug_项目计划书_v6.3.md) 参照）：「純粋な局所学習」という主張の範囲は**シナプス学習規則**であって、**入力エンコーダではありません**。判定は「パラメータが勾配に入るか」であって「モデルが大きいか」ではない——凍結した 2GB の事前学習済み embedding は、固定ランダム射影とこの点で等価です。逆に、100 行のローカルエンコーダが逆伝播で自身のパラメータを学習するなら、この境界に**違反**します。テキストプラグインの `nomic-embed-text`（Ollama、768 次元）は凍結です。
- （スケルトンライブラリがなぜ独立したパッケージなのか、なぜこれらを意図的に含めないのかは [ADR-0001](docs/adr/ADR-0001-skeleton-as-separate-library.ja.md) 参照。）
- スケルトンライブラリは**PyPI にリリース済み**です：`pip install biosnn-bus`（現在 `0.2.0`）。リリースは PyPI の trusted publishing（OIDC、トークンを保存しません）で行い、手順は [release.yml](.github/workflows/release.yml) にあります。

## アーキテクチャ

データは下から上へ流れる。✅ は現在のリポジトリで既に動くもの；◐ は**その層の単一の学習規則が第 1 フェーズで検証済み**で、モジュール全体（多層ネットワーク、モダリティデコーダ、ワーキングメモリ回路）は未実装のもの；⬜ はプロジェクト計画書にあり未着手のもの——同じ図に描いているのは、この図が同時にロードマップでもあるためだ。

```mermaid
flowchart TB
    subgraph L5["LLM 調度・連携層"]
        MCP["MCP / API インターフェース<br/>差し替え可能な推論エンジン<br/>⬜ 第 5 フェーズ"]
    end

    subgraph L4["実行層"]
        ACT["行動生成 ＋ モダリティデコーダ<br/>R-STDP ＋ 報酬予測 Critic<br/>◐ 第 1 フェーズ：学習規則を検証済み（W3）<br/>⬜ モダリティデコーダ"]
    end

    subgraph L3["認知層（認知コア）"]
        WM["作業記憶<br/>RSNN ＋ ALIF ＋ e-prop<br/>◐ 第 1 フェーズ：学習規則を検証済み（W2）<br/>⬜ 多層 RSNN と作業記憶回路"]
        EM["情景記憶<br/>パターン分離 ＋ 神経新生<br/>⬜ 第 4 フェーズ"]
        MG["メタ認知ゲート<br/>不確実性の監視<br/>⬜ 第 4 フェーズ"]
    end

    subgraph ARB["規則間の仲裁 research/arbiter"]
        CF["衝突判定：コサイン > 90 度<br/>✅ 実装済み"]
        ES["ES メタ学習仲裁器<br/>✅ 両ラインに接続済み（共有入力シナプス）<br/>⚠ 「50% 減」は合成データ上で自明に成立"]
    end

    subgraph L2["スケルトンライブラリ biosnn-bus"]
        BUS["SpikeBus<br/>融合チャネルでグループ化 · 時間格子を整列<br/>射影してから融合 · スパースランダム射影<br/>TAAF 時間注意誘導融合（rate-based gating、学習なし）<br/>✅ 実装済み"]
    end

    subgraph L1["知覚層（プラグイン化）"]
        IMG["画像 image_diff<br/>DVS イベント符号化<br/>✅ 実装済み"]
        MN["MNIST mnist_pulse<br/>発火率符号化<br/>✅ 実装済み"]
        TXT["テキスト text_zscore<br/>凍結 embedding ＋ z-score top-k<br/>✅ 実装済み"]
        AUD["音声プラグイン<br/>蝸牛モデル周波数分解<br/>⬜ 第 3 フェーズ"]
        THIRD["サードパーティプラグイン<br/>entry points で接続<br/>✅ 仕組みは整備済み"]
        HEB["学習規則：核化 IB-Hebbian ＋ 除法正規化<br/>◐ 第 1 フェーズ検証済み（W1、MNIST 97.79%）"]
    end

    IMG --> BUS
    MN --> BUS
    TXT --> BUS
    AUD -.-> BUS
    THIRD -.-> BUS
    BUS --> WM
    BUS -.->|知覚層を接続| HEB
    HEB -.->|特徴| WM
    WM --> EM
    EM --> MG
    MG --> ACT
    MG -.->|呼び出しを起動| MCP
    MCP -.->|結果を符号化して再注入| EM
    ARB -.->|層間衝突を調停| WM
    ARB -.->|層間衝突を調停| ACT
```

層間の衝突は **ES メタ学習アービタ**が調停する（⬜ 第 2 フェーズ）。各層の学習則の完全な説明は[プロジェクト計画書](BioSNN-Plug_项目计划书_v6.2.md) §2、§3 参照。

層間の衝突は **ES メタ学習アービタ**が調停します：衝突判定と仲裁器はいずれも実装済みで、
**両方の研究ラインにも接続済み**です（`research/arbiter/shared_synapse.py`、二つの規則が
入力→隠れ層のシナプスを共有）。注意：「衝突イベント 50% 減」は合成データ上では**自明に
成立**します——最適解は一方の規則を切ることなので、減少量は `retention` と**一緒に**
読んでください。[`research/arbiter/README`](research/arbiter/README.ja.md) 参照。

注意：**図の ✅ は「実装済み」であって、「路線図が置く場所に接続済み」という意味では
ありません**：TAAF は実装されていますが**明示的に選ぶ**戦略で（既定は連結のまま）、
テキストプラグインは実装されていますが**凍結した事前学習済みエンコーダ**を前に置いて
います（境界宣言は[計画書 §2.3](BioSNN-Plug_项目计划书_v6.3.md)——「純粋な局所学習」は
シナプス規則の話で、入力エンコーダの話ではない）。各層の学習則の完全な説明は
[プロジェクト計画書](BioSNN-Plug_项目计划书_v6.2.md) §2、§3 参照。

## クイックスタート

numpy だけに依存し、GPU は不要です：

```bash
pip install biosnn-bus
```

（`main` 上の**未リリース**の変更を入れる場合のみ Git 経由で：`pip install "biosnn-bus @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-bus"`）

モダリティプラグインは 3 つのメソッドと 2 つのプロパティでできている：

```python
import numpy as np

from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeBus, SpikeTrain


class LevelEncoder(ModalityPlugin):
    """一次元信号をレベルに応じて集団スパイクに符号化する。"""

    def __init__(self, spike_dim: int = 16) -> None:
        self._spike_dim = spike_dim
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    @property
    def modality_name(self) -> str:
        return "level"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        radius = 0.5 / (self._spike_dim - 1)
        return SpikeTrain(data=np.abs(signal[:, None] - self.levels) <= radius)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output: SpikeTrain):
        return spike_output.data.astype(float) @ self.levels


bus = SpikeBus(bus_dim=64, seed=0)
bus.register(LevelEncoder())
output = bus.step({"level": np.linspace(0, 1, 12)})
print(output)
```

[Colab デモ](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.ja.ipynb)（5 分、GPU 不要）は、プラグインから認知コアの入口までの完全なスパイクラスタプロットを描く。

サードパーティのパッケージにプラグインを提供させたい？**本リポジトリのコードを 1 行も変更する必要はない**——[プラグイン開発ガイド](docs/plugin_guide.ja.md#サードパーティパッケージからプラグインを提供する) 参照。

## リポジトリ構成

```text
packages/biosnn-bus/   スケルトンライブラリ：独立した配布パッケージ、semver に従い、研究コードに一切依存しない
research/              研究コード：第 1 フェーズから埋める、12 か月の破壊的変更免責期間
examples/              実行できるサンプル；スクリプトが唯一の真実の源で、notebook はそこから生成する
docs/                  プラグインガイド、ADR、再現チェックリスト、引用リスト
scripts/               CI と pre-commit が使うチェックスクリプト
```

## メンテナンス方針

メンテナは 1 名。完全な再現手順を伴う報告を優先して扱う。

- `biosnn-bus` は semver に従い、現在は `0.x`——慣例により 0.x ではマイナーバージョンで破壊的変更を許容し、最初に外部から依存されるバージョンで `1.0.0` に上げる；
- `research/` 配下のコードは最初の 12 か月（2027-09 まで）は互換性を約束しない；
- サードパーティのモダリティプラグインは独立したパッケージにすればよく、本リポジトリに PR を出す必要はない（[ADR-0003](docs/adr/ADR-0003-entry-point-plugin-discovery.ja.md)）。

## ライセンスと引用

コードは [Apache-2.0](LICENSE)、ドキュメントとプロジェクト計画書は [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)。データセットは各データソースのライセンスに従い、本リポジトリはデータミラーを提供しない；ダウンロードと前処理スクリプトは [scripts/download_data.py](scripts/download_data.py)。

研究ラインが依存する **SpikingJelly は啓智オープンソースライセンス 1.0（OIOSL）を使い、Apache-2.0 ではない**。研究用途では発動しないが、**商用利用・再配布には AITISA への開示申告が必要**。その取舍と Python 下限への影響は [ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.ja.md) を参照。

引用には [CITATION.cff](CITATION.cff) を使う。
