# プラグイン開発ガイド

[中文](plugin_guide.md) · [English](plugin_guide.en.md)

> このドキュメントの各 Python コードブロックは CI によって**実際に実行**されます
> （`scripts/check_doc_code_blocks.py` を参照）。先頭から末尾までコピー＆ペーストできます。
> ブロック同士はつながっています。単独では実行できない断片には、フェンスに
> `python no-run` を付けてください。

## 何を書くのか

モダリティプラグインが答えるのは 3 つの問いです：

| メソッド | 答える問い |
| :--- | :--- |
| `encode(raw_input)` | 自分のモダリティのデータをどうやってスパイクに変えるか？ |
| `get_membrane()` | 自分のモダリティのニューロンはどんな形をしているか？ |
| `decode(spike_output)` | スパイクをどうやって自分に読める形に戻すか？ |

これに 2 つのメタ情報が加わります：`modality_name`（名前）と `spike_dim`（何本出力するか）。

これで全部です。学習則を気にする必要はありませんし、バスがどう融合するかを気にする必要も、
認知コアが何であるかを気にする必要もありません——それらはスケルトンライブラリと認知コアの仕事です。

## 完全な最小プラグイン

以下はそのまま実行できます。これは**集団レベル符号化**を実装します：1 次元信号を
ニューロン集団の発火に符号化し、信号があるニューロンの好むレベルに近いほど、その
ニューロンが発火しやすくなります。

```python
import numpy as np

from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeTrain


class LevelEncoder(ModalityPlugin):
    """1 次元信号をレベルに応じて集団スパイクに符号化する。"""

    def __init__(self, spike_dim: int = 16) -> None:
        if spike_dim < 2:
            raise ValueError("spike_dim は 2 以上である必要があります。でないと信号レベルを区別できません。")
        self._spike_dim = spike_dim
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    # ---- 2 つのプロパティ ----

    @property
    def modality_name(self) -> str:
        return "level"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    # ---- 3 つのメソッド ----

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        radius = 0.5 / (self._spike_dim - 1)  # 隣接レベルの半間隔
        distance = np.abs(signal[:, None] - self.levels[None, :])
        return SpikeTrain(data=distance <= radius, channel=self.channel)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output: SpikeTrain):
        active = spike_output.data.astype(np.float64)
        total = active.sum(axis=1)
        return np.divide(
            active @ self.levels, total, out=np.zeros(active.shape[0]), where=total > 0
        )
```

試してみましょう：

```python
plugin = LevelEncoder(spike_dim=16)
signal = np.linspace(0.0, 1.0, 8)

train = plugin.encode(signal)
print(train)  # 形状 (8, 16)、ブール、channel=temporal
print("解码回来:", np.round(plugin.decode(train), 3))
```

### `spike_dim` は `encode` の実際の出力と一致しなければならない

バスがこれを検査します。16 と宣言しておいて 32 本を返しても、登録時にはエラーになりませんが、
最初の `step()` で止められます：

```python
from biosnn_bus import PluginContractError, SpikeBus


class WrongDim(LevelEncoder):
    @property
    def spike_dim(self) -> int:
        return 16

    def encode(self, raw_input) -> SpikeTrain:
        return SpikeTrain(data=np.zeros((4, 99), dtype=bool))


bus = SpikeBus(bus_dim=32, seed=0)
bus.register(WrongDim())  # 登録時には検査されない。まだ encode を一度も呼んでいないから
try:
    bus.step({"level": signal})
except PluginContractError as exc:
    print("如期被拦下:", exc)
```

> 登録時に宣言と出力が一致するかは調べられません：`spike_dim` は静的宣言であり、`encode` の
> 出力は実行時の挙動だからです。だからこそ `encode` の中では `self.spike_dim` を使って出力を
> 構築すれば、両者がずれることはありません（上の `LevelEncoder.encode` はそう書かれています）。
> 検証を `__init_subclass__` に置かない理由は
> [ADR-0006](../docs/adr/ADR-0006-plugin-interface-fidelity.ja.md) を参照してください。

## 2 つの任意プロパティ

### `fusion_channel`：どの融合チャネルを使うか

プロジェクト計画書 §2.2 のスパイクバスは**二重チャネル**です。既定は `'temporal'` で、
もう一方の値は `'semantic'` です。どちらを選ぶかは**このモダリティの情報がどのように
組織されているか**で決まります：

| 値 | 適するもの | プロジェクト計画書 §2.3 の例 |
| :--- | :--- | :--- |
| `'temporal'` | 情報が主に時間構造にある（リズム、順序、持続時間） | テキスト、音声 |
| `'semantic'` | 情報が主に内容／空間構造にある | 画像 |

```python
class SemanticEncoder(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "semantic"


print(SemanticEncoder().channel)  # "semantic" と表示される
```

間違って書くとどうなるか：

```python
class BadChannel(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "auditory"  # そんなチャネルは存在しない


try:
    SpikeBus(bus_dim=32).register(BadChannel())
except PluginContractError as exc:
    print("如期被拦下:", exc)
```

### `temporal_scale`：このモダリティの時間スケール

単位はミリ秒で、既定は `10.0` です。プロジェクト計画書はこれを「そのモダリティの
典型的な時間スケールであり、TAAF モジュールの初期化に用いるもの」と定義しています。
スケルトン段階では、これはマーカーとしてスパイク列とともに伝達され、融合時には各モダリティの
**最大値**を取ります（最も遅いモダリティが融合後の時間スケールを決めます）。

```python
class FastEncoder(LevelEncoder):
    @property
    def temporal_scale(self) -> float:
        return 1.0


print(FastEncoder().temporal_scale)
```

## コントラクトのセルフチェック

`SpikeBus.register()` はプラグインの `validate()` を呼びます。以下の 4 種類の問題は
その時点で止められます：

```python
class BadName(LevelEncoder):
    @property
    def modality_name(self) -> str:
        return "   "


class BadDim(LevelEncoder):
    @property
    def spike_dim(self) -> int:
        return -3


class BadScale(LevelEncoder):
    @property
    def temporal_scale(self) -> float:
        return 0.0


class BadChannel(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "auditory"


for bad in (BadName(), BadDim(), BadScale(), BadChannel()):
    try:
        SpikeBus(bus_dim=32).register(bad)
    except PluginContractError as exc:
        print(f"{type(bad).__name__}: {exc}")
```

## バスに接続する

```python
bus = SpikeBus(bus_dim=64, seed=0)
bus.register(LevelEncoder(spike_dim=16))

output = bus.step({"level": signal})
print(output)  # 形状 (8, 64)、float32
```

注意点が 2 つあります：

1. `output.data` は 0/1 のスパイクではなく**浮動小数点の電流**です。バス末尾の
   スパースランダム射影は ±1 の重みを持ち、負の値は抑制性入力を表します。スパイクが
   必要なときは `output.binary()` を呼んでください。
2. 同じバス上のモダリティ名は一意でなければなりません。同じモダリティで複数の構成を
   登録したい場合（たとえば解像度の異なる 2 つの画像エンコーダ）、`name=` で明示的に
   区別します：

```python
bus.register(LevelEncoder(spike_dim=16), name="level_low")
bus.register(LevelEncoder(spike_dim=64), name="level_high")
print(bus.registered_modalities)
```

## サードパーティパッケージからプラグインを提供する

**このリポジトリのコードを 1 行も書き換える必要はありません。** 自分のパッケージの
`pyproject.toml` で宣言します：

```toml
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

ユーザーがあなたのパッケージをインストールした後は：

```python no-run
# この断片は実在のサードパーティパッケージに依存しており、このリポジトリには当然
# インストールされていないので実行できません。
from biosnn_bus import SpikeBus, discover_plugins, get_plugin

discover_plugins()  # インストール済みの配布パッケージが宣言したプラグインをすべて走査して登録する
bus = SpikeBus(bus_dim=256, seed=0)
bus.register(get_plugin("audio")())
```

`discover_plugins()` は冪等で、繰り返し呼んでも二重登録されません。あるエントリポイントの
指す先が `ModalityPlugin` のサブクラスでない場合、または読み込み時に ImportError を
送出する場合、エラーメッセージにはそのエントリポイントの名前とインポートパスが含まれます。

## テストテンプレート

プラグインはテストすべきものです：コントラクト違反は学習の途中で気づくのではなく、
**登録時**に露見させなければなりません。これは本リポジトリ同梱のサンプルプラグイン
`DiffImagePlugin` のテスト構成で、そのまま流用できます
（`packages/biosnn-bus/tests/test_plugins_image_diff.py` を参照）：

```python
import numpy as np
import pytest

from biosnn_bus import SpikeBus, SpikeTrain
from biosnn_bus.plugins import DiffImagePlugin


@pytest.fixture
def plugin():
    return DiffImagePlugin(height=4, width=5, threshold=0.25)


def test_shape_contract(plugin):
    train = plugin.encode(np.zeros((6, 4, 5)))
    assert train.data.shape == (6, plugin.spike_dim)
    assert train.is_binary


def test_rejects_wrong_shape(plugin):
    with pytest.raises(ValueError, match="期望形状"):
        plugin.encode(np.zeros((6, 4, 6)))


def test_survives_a_round_trip(plugin):
    """符号化は不可逆でもよいが、復号した形状は正しくなければならず、誤差には明確な上界がある。"""
    frames = np.zeros((4, 4, 5))
    frames[1, 0, 0] = 0.25  # しきい値ちょうどなので、イベントが 1 つ発火するはず
    restored = plugin.decode(plugin.encode(frames))
    assert restored.shape == frames.shape
    assert np.abs(restored - frames).max() <= 0.25


def test_plugs_into_the_bus(plugin):
    bus = SpikeBus(bus_dim=32, seed=0)
    bus.register(plugin)
    assert bus.step({"image_diff": np.zeros((6, 4, 5))}).data.shape == (6, 32)
```

**形状コントラクトと誤った形状の拒否、この 2 つは必須です**——これらは
「プラグインの書き間違い」を、学習の途中で初めて露見するものから、登録時の 1 行の
エラーへと変えます。

## さらに先へ

- プラグインインタフェースの完全なシグネチャと意味：`packages/biosnn-bus/src/biosnn_bus/plugin.py`
- スパイク列コンテナ `SpikeTrain` の完全な API：`packages/biosnn-bus/src/biosnn_bus/types.py`
- なぜグローバルなレジストリではなくエントリポイントなのか：`docs/adr/ADR-0003-entry-point-plugin-discovery.ja.md`
- スケルトンライブラリが**含まない**もの（そしてその理由）：`docs/adr/ADR-0001-skeleton-as-separate-library.ja.md`
