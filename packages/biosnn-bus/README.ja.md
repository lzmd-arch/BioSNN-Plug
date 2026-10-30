# biosnn-bus

[中文](README.md) · [English](README.en.md)

[BioSNN-Plug](https://github.com/lzmd-arch/BioSNN-Plug) の**モダリティプラグインスケルトンライブラリ**です。

担うのはただ一つ：**新しいモダリティを追加するときに、既存のアーキテクチャを変更しなくてよいようにすること**です。

> **これはスケルトンであり、認知コアではありません。** 本ライブラリにはいかなる学習則
> （e-prop / R-STDP / カーネル化 IB-Hebbian）も含まれず、スパイクニューロンモデルも
> 含まれません。GPU にも依存せず、PyTorch にも依存しません。提供するのは、プラグインの
> インターフェースコントラクト、登録と発見のメカニズム、そしてスパイクバスの継ぎ目です。

## インストール

```bash
pip install biosnn-bus

# torch ブリッジが必要なとき（SpikeTrain.to_torch / from_torch）
pip install "biosnn-bus[torch]"
```

`main` ブランチの**未リリース**の変更（レビュー中の修正など）を入れる場合は Git から：

```bash
pip install "biosnn-bus @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-bus"
```

依存するのは numpy だけです。

## クイックスタート

```python
import numpy as np
from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeBus, SpikeTrain


class MyPlugin(ModalityPlugin):
    @property
    def modality_name(self) -> str:
        return "my_modality"

    @property
    def spike_dim(self) -> int:
        return 64

    def encode(self, raw_input):
        return SpikeTrain(data=np.asarray(raw_input) > 0.5, channel=self.channel)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output):
        return spike_output.rates


bus = SpikeBus(bus_dim=128, seed=0)
bus.register(MyPlugin())
out = bus.step({"my_modality": np.random.default_rng(0).random((10, 64))})
print(out)
```

完全に実行できる版は [`examples/quickstart_register_plugin.py`](https://github.com/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.py)、
および [Colab notebook](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.ja.ipynb)（GPU なしで実行できます）にあります。

実行すると下の図が出ます——1 次元の正弦信号が時間チャネル（32 ユニットのレベル符号化）、8×8 の
輝度変化列が意味チャネル（128 チャネル、ON/OFF）を通り、最下段が融合と射影のあとに
**実際に認知コアへ送られる入力**です：

![三パネルのスパイクラスタ：時間チャネル sine_wave / 意味チャネル image_diff / SpikeBus 融合出力](../../figures/f1-spike-raster.png)

図中のラベルが英語なのは、既定のフォントに CJK 字形がなく、日本語ラベルが豆腐（□）になるためです。
図の**真実の源はただ一つ**——[`scripts/make_figures.py`](../../scripts/make_figures.py) が上の例を
再実行し、例自身が PNG を書き出します（描画コードを別に持たない）。論文用の図 F1 です。

## 中核となる概念

| 概念 | 説明 |
| :--- | :--- |
| `ModalityPlugin` | モダリティプラグインの抽象基底クラスです。`encode` / `decode` / `get_membrane` の 3 メソッドと、`modality_name` / `spike_dim` の 2 プロパティを持ちます |
| `SpikeTrain` | 形状 `(T, N)` のスパイク列コンテナです。ブール値（スパイク）でも浮動小数点（発火率）でも保持できます |
| `FusionChannel` | `temporal` / `semantic` の 2 本の融合チャネルです。プロジェクト計画書 §2.2 の「二重チャネル融合」に対応します |
| `SpikeBus` | スパイクバスです。符号化 → 時間整列 → チャネルごとの融合 → 認知コア次元へのスパースランダム射影、という流れです |
| `register_plugin` | プロセス内登録デコレータです |
| `discover_plugins` | サードパーティの配布パッケージが宣言したエントリポイントをスキャンします |

## 内蔵プラグイン

| プラグイン | モダリティ | 符号化 | 融合チャネル |
| :--- | :--- | :--- | :--- |
| `image_diff` | 画像 | DVS 形式の ON/OFF イベント（フレーム差分） | semantic |
| `mnist_pulse` | MNIST | 画素値＝発火確率（ベルヌーイ） | semantic |
| `text_zscore` | テキスト | 凍結 embedding ＋ z-score top-k 群体符号化 | temporal |
| `audio_cochlea` | 音声 | 対数帯域分解 ＋ 発火率 | temporal |

## サードパーティプラグインの組み込み方

自分のパッケージでエントリポイントを宣言します。**BioSNN-Plug のコードを一行も変更する必要はありません**：

```toml
# 自分のパッケージの pyproject.toml
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

利用者側がそれをスキャンして組み込みます。次のコードはそのまま実行できます——サードパーティプラグインが一つもインストールされていなくても：

```python
from biosnn_bus import SpikeBus, discover_plugins, get_plugin, list_plugins
from biosnn_bus.plugins import DiffImagePlugin

discover_plugins()  # サードパーティの配布パッケージが宣言したプラグインをスキャンします。一つもインストールされていなくてもエラーになりません
print(list_plugins())  # 組み込みのサンプルプラグインは import 時にレジストリへ登録済みです

bus = SpikeBus(bus_dim=128, seed=0)
bus.register(get_plugin("image_diff")())  # bus.register(DiffImagePlugin()) と等価です
print(bus)
```

## スケルトンの境界

スパイクバスの現在の既定の融合戦略は**単純連結**（`ConcatenateFusion`）であり、
プロジェクト計画書 §2.2 が記述する TAAF 時間注意誘導融合では**ありません**——後者は
第 2 フェーズの研究タスクです。バスにはすでに `FusionStrategy` プロトコルと二重チャネルルーティングが
用意されており、戦略を差し替えても呼び出し側のコードは変わりません。

理由は [ADR-0001](https://github.com/lzmd-arch/BioSNN-Plug/blob/main/docs/adr/ADR-0001-skeleton-as-separate-library.ja.md) を参照してください。

## SpikeBus 仕様 v0.1（ホスト契約、2026-10-29 草案）

**範囲**：プラグイン作者とホスト（製品）の接縫。学習規則・ニューロンモデル・スケジューリングは
**標準化しない**。

| ホスト原語 | 意味 | 引数 | 返り値／エラー |
| :--- | :--- | :--- | :--- |
| `inject(modality, raw, t=None)` | 一つのモダリティの生入力をバスへ送る | `modality: str`、`raw: Any`、`t: int \| None` | `None`；未知のモダリティ → `UnknownModality` |
| `read_state()` | 融合状態の**読み取り専用スナップショット** | — | `dict`（チャネル、次元、時刻）；呼び出し側は破壊的変更をしてはならない |
| `wait_trigger(timeout_ms)` | SNN が「行動する価値がある」と判断するまでブロック | `timeout_ms >= 0` | `TriggerEvent(t, salience, associations)`、タイムアウトなら `None` |

**データ契約**：時間軸は整数ミリ秒、同一ステップ内で各モダリティを整列；スパイクはブール、
浮動小数点の発火率は `[0, 1]`；NaN/inf 禁止；融合後の次元は常に `bus_dim`。

**プラグイン契約**：`encode` / `decode` / `get_membrane`、`modality_name` は一意、
`spike_dim >= 1`；torch 非依存（`[torch]` は任意のブリッジのみ）。

**バージョン**：セマンティックバージョニング。`v0.x` は破壊的変更を許し、リリースノートに明記する
こと。ホストは minor を固定する。

**適合性テスト（CI）**：C1 登録後の `step` は `(bus_dim,)` を返す；C2 未知モダリティは例外；
C3 時刻整列；C4 次元の恒常性；C5 スナップショットは読み取り専用。

**読み書きの境界（三面マトリクス、2026-10-30 確定）**：開放度＝信頼境界——**書き込みに責任を持つ者だけが書き込み権を持つ**。

| 面 | 書き込める者 | 制約 |
| :--- | :--- | :--- |
| 知覚面 | 任意のプラグイン（「感覚」の書き込みは開放） | TAAF 融合 + 呼び出しごとの監査 + **プラグインは既定で無効** |
| 記憶面 | **力学コアのみ**（ホストも LLM も直接は書かない） | 外部は `propose` のみ。二段階キュー + 統合ゲート付き消費を経由 |
| 知識面 | 最大に開放（文書・仕様・データ） | ただし機構の中核は非公開で、バス契約の外 |

MCP についての三行：**読み取り専用は立場**（LLM は決して直接書かない）、**propose は弁**（二段階キュー）、**バスが唯一の扉**。命名規律：**実際には書かないツールを `write_*` と呼んではならない**（ADR-0027 を参照）。

**符号化のクレジット**：次元非依存のテキスト符号化（z-score 標準化 top-k 集団符号化）は EMBER のグループ（Savage, 2026, arXiv:2604.12167）に由来します。本リポジトリの `text_zscore` はその**後続実装であり、引用済み**です（帰属の向きは EMBER → BioSNN）。本仕様でこの符号化に触れる記述のクレジットは EMBER のグループに帰します。

## ライセンス

Apache-2.0。ドキュメントは CC-BY 4.0 を採用しています。
