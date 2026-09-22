# 再現チェックリスト

[中文](reproducibility.md) · [English](reproducibility.en.md)

> プロジェクト計画書 §12.4 は「各フェーズの正確な環境（commit hash、依存パッケージの
> バージョン、乱数シード）」を要求しています。

## なぜこれが必要か

再現が失敗する最も一般的な原因はコードの誤りではなく、**環境が一致しない**ことです：
numpy のバージョンが変わった、PyTorch のバージョンが変わった、乱数シードを記録して
いない、ローカルの未コミットの変更のまま実行した。

したがって、実験の結論を外部に報告するたびに、以下の内容を添付する必要があります。
**添付のない結論は未再現とみなします。**

## テンプレート

これをコピーして記入し、実験記録または issue に入れてください：

```text
実験名：
日付：
Git commit：          # git rev-parse HEAD
Git 状態：            # git status --porcelain の出力；空でなければ未コミットの変更がある
Python：              # python -V
OS / アーキテクチャ：
ハードウェア：        # GPU モデル + GPU メモリ
乱数シード：          # 実験で使うすべて。それぞれの用途を明記する
依存関係スナップショット：  # uv.lock のハッシュ、または `uv pip freeze` の出力
実行コマンド：
所要時間：
```

## これらの情報の取得方法

本リポジトリは依存関係の管理に `uv` を使い、`uv.lock` をバージョン管理にコミットしています
——これが再現の基盤です：**同じ lock ファイルからは同じ依存パッケージのバージョンが
解決されます**。

```bash
git rev-parse HEAD                     # commit hash
git status --porcelain                 # 有输出 = 工作区不干净，复现结果不可信
python -V
uv pip freeze                          # 已安装依赖的精确版本
shasum -a 256 uv.lock                  # Windows 用 Get-FileHash uv.lock
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

1 つのコマンドでまとめて取得する場合（Windows / Git Bash）：

```bash
echo "commit: $(git rev-parse HEAD)"; \
echo "dirty: $(git status --porcelain | wc -l) files"; \
python -V; uv pip freeze | shasum -a 256; \
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

## 乱数シード

**スケルトンライブラリのランダム性は 1 か所だけです**：スパイクバス末端の
スパースランダム射影（`SparseRandomProjection`）。そのシードは `SpikeBus(seed=...)` で
決まります。

約束事：

- スケルトンライブラリの内部では、プロセス状態に依存する乱数源を**使用しません**。
  具体的には、射影行列のシードは `zlib.crc32` によって `(seed, チャネル, モダリティ構成)`
  から導出されており、Python 組み込みの `hash()` では**ありません**——後者は文字列に対して
  プロセス単位でソルトを付与するため（`PYTHONHASHSEED`）、同じ設定でもプロセスが異なれば
  異なる射影行列が得られ、「シードを固定すれば再現できる」という一文は単一プロセス内で
  しか成立しません。

  これを監視している回帰テストがあります：`packages/biosnn-bus/tests/test_bus.py::test_projection_seed_survives_a_process_boundary`
  は 2 つの異なる `PYTHONHASHSEED` で子プロセスを起動し、出力が一致することを検証します。

- プラグイン自身のランダム性はプラグインが自分で責任を持ちます。プラグインを書くときは
  `seed` をコンストラクタ引数として公開し、`np.random.random()` を直接呼び出さないで
  ください。

**研究コード（`research/`）のランダム性**は
[`research/common/seeding.py`](../research/common/seeding.py) を通します：まず用途を登録し、
それから適用します。ここの `derive_seed` が組み込みの `hash()` ではなく `zlib.crc32` を
使うのはスケルトンライブラリと同じ理由です（後者はプロセスごとに加塩されるため、プロセス
をまたぐと再現できません）。同じクロスプロセスの回帰テストが監視しています。各検証線は
`np.random.seed(...)` を**直接呼ぶべきではありません**——そうするとシードが再現記録に
現れなくなります。

## 提出前セルフチェック

[`CONTRIBUTING.md`](../CONTRIBUTING.ja.md) の共通チェックリストのほかに、再現に関しては
次の 1 項目も確認してください：

```bash
uv sync --locked      # lock ファイルと pyproject が一致している必要があります
```

上記の「同じ lock ファイルからは同じ依存パッケージのバージョンが解決されます」という
一文は、この手順が通ることが前提です。

## スケルトンライブラリフェーズの環境記録

第 0 フェーズ（スケルトンライブラリとオープンソース基盤）は GPU 実験を伴わず、結論は
すべて**決定論的**です——テストスイートは CPU 上で実行され、結果はハードウェアに依存しません。
したがってこのフェーズでは依存パッケージのバージョンを記録するだけで十分です。

| 項目 | 値 |
| :--- | :--- |
| スケルトンライブラリのバージョン | `biosnn-bus` 0.1.0 |
| Python | >= 3.10（CI は 3.10 / 3.11 / 3.12 をカバー） |
| 実行時依存 | `numpy>=1.24` のみ |
| ハードウェア要件 | なし。CPU で可 |

## 第 1 フェーズ（単一規則検証）の環境記録

このフェーズの実験には **GPU が関与します**。したがってもはや純粋に決定論的ではありません：
同じ設定でも CPU と CUDA で異なる浮動小数点結果が出ることがあります。よって依存関係の
バージョンに加えて、ハードウェアと GPU メモリも記録しなければなりません。

| 項目 | 値 |
| :--- | :--- |
| Python | **>= 3.11**（SpikingJelly 2.0.0rc1 の下限、`docs/adr/ADR-0008` を参照） |
| 主要な依存 | `torch`、`torchvision`、`torchaudio`、`spikingjelly==2.0.0rc1`、`gymnasium`（`research` 依存グループ） |
| torch の入手元 | プラットフォーム別：Windows → `download.pytorch.org/whl/cu130`（sm_120 をカバー）；その他 → `whl/cpu`。理由は `pyproject.toml` の `[tool.uv.sources]` のコメントにあります |
| ハードウェア | NVIDIA GeForce RTX 5060、8,151 MiB、sm_120 |
| スケルトンライブラリ | 影響なし：numpy のみに依存し、`requires-python >=3.10` のままです（CI の `package` job が 3.10 の脚でカバーします） |

**依存グループの分離**：`research` グループは**既定ではインストールされません**。
`uv sync --locked` は torch のない第 0 フェーズの環境を与え、`uv sync --locked --group
research` が第 1 フェーズの実験環境です。したがってスケルトンライブラリの「torch に依存
しない」（ADR-0002）という宣言は、ローカルでも検証可能なままです。

各モジュールの docstring の先頭には、12 か月の破壊的変更の免責期間と満了日（2027-09 まで）
を記載します。

### これまでに完了した実験記録

| 検証線 | 記録の場所 | 受入の数値 |
| :--- | :--- | :--- |
| W1 カーネル化 IB-Hebbian 知覚層 | [`research/ib_hebbian/README.ja.md`](../research/ib_hebbian/README.ja.md) | MNIST 98.01%（閾値 70%） |
| W2 e-prop 認知層 | [`research/eprop/README.ja.md`](../research/eprop/README.ja.md) | sMNIST 77.48%；活動ニューロン比率 1.0000（閾値 > 60%） |
| W3 R-STDP + TD-LTP Critic | [`research/rstdp/README.ja.md`](../research/rstdp/README.ja.md) | ✅ **§七 の二基準を達成**：CartPole 中央値 **260.0 歩**（ホールドアウトシード 25–44、二十個、基準 ≥ 200）；偏移比の最大 **0.0359**（全シード < 0.10 ✓）。**ただし最小値は 74** で、自ら加えた「安定」条件（≥ 100、二十個中二個が 100 未満）は未達。設定と根拠は [ADR-0009](adr/ADR-0009-w3-behaviour-policy-and-trace-centring.ja.md) 参照 |

## 以降のフェーズでの記録場所

第 1 フェーズ（単一規則検証）以降は、実験記録ごとに完全なテンプレートを付ける必要があります。
この部分は [`research/common/provenance.py`](../research/common/provenance.py) が生成します：

```python
from research.common.provenance import DegradationLog, collect
from research.common.seeding import SeedBook

book = SeedBook(base=0)
book.derive("权重初始化")

record = collect(
    "ib_hebbian/mnist",
    seeds=book.render(),
    elapsed_s=123.4,
    peak_mb=None,  # 実実験では measure_peak_memory() が返す stats["peak_mb"] を渡します
    degradation=DegradationLog(scale_reduction=True, notes=["5 万 → 1 万ニューロン"]),
)
print(record.render_block())  # Markdown の text フェンスブロックとしてそのまま貼れます
```

**手で正しく書くことのできない** 4 つのフィールドを自動で読み出します：`git rev-parse
HEAD`、`git status --porcelain`（空でなければ「作業ツリーが汚れており、この結論は信頼
できません」と明示します）、`uv.lock` の SHA-256、そしてハードウェアと GPU メモリです。

GPU 実験ではさらに次を記録する必要があります：

- GPU メモリのピーク（プロジェクト計画書 §6.2 は 8GB をハード制約として挙げています）——
  `research/common/device.py` の `measure_peak_memory()` で測定し、`nvidia-smi` ではなく
  PyTorch アロケータの口径を用います
- プロジェクト計画書 §6.2 の縮退パスを発動したかどうか（規模縮退 / 分割学習 / INT8 痕跡量子化）
  ——`DegradationLog` で登録します；**発動しないのが常態**なので、明示的に登録されたものだけが
  数えられます
- 学習時間とエネルギー消費に関わる量（§9 のエネルギー効率指標は「呼び出し頻度-エネルギー消費曲線」の形式で報告します）

## データ

プロジェクト計画書 §12.1 により、本リポジトリは**データミラーを提供しません**。

ダウンロードと前処理のスクリプトは [`scripts/download_data.py`](../scripts/download_data.py) です：

```bash
uv run python scripts/download_data.py --list        # 登録済みデータセットの一覧
uv run python scripts/download_data.py mnist         # ダウンロード + 検証 + .npy へ変換
uv run python scripts/download_data.py --verify-only # 既存ファイルの検証のみ、ネットワーク不要
```

データは `.gitignore` で無視される `data/` に置かれ、各ファイルは**個別にハッシュを検証**
してからでないと使用可能とはみなしません。検証を通った SHA-256 は再現記録のために出力します。

**チェックサムが MD5 なのは意図的です**：これはセキュリティ機構ではなく完全性の検査
（ダウンロードの切断とミラーのドリフトに対するもの）です。MNIST について MD5 は
**独立した第三者が公表している唯一のチェックサム**だから選んでいます——
`torchvision.datasets.MNIST` は 4 ファイルの MD5 を自身のソースにハードコードしており、
本スクリプトは同じ値を固定します。したがって相互に照合できます。自前で SHA-256 を計算して
固定するのは、自分で自分を保証するだけです。
