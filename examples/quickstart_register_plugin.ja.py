# %% [markdown]
# # 5 分で：カスタムモダリティプラグインを登録する
#
# この例は `biosnn-bus` が解こうとしている唯一の問題を示します：
# **モダリティの追加に既存アーキテクチャの変更は不要**。
#
# 順に 4 つのことを行います：新しいモダリティプラグインを書き（1 次元信号を集団スパイクに
# 符号化します）、登録し、
# 同梱の画像プラグインと一緒に `SpikeBus` へ接続し、二重チャネルルーティングと
# スパイクラスタを見ます。
#
# すべて CPU で動作し、依存は numpy のみです（描画に matplotlib）。

# %%
# **Colab のようなまっさらな環境には、このパッケージはまだ入っていません。** ローカル開発では
# すでに入っているのでこのブロックはスキップされ、import が実際に失敗したときだけインストールします。
import subprocess
import sys

try:
    import biosnn_bus
except ModuleNotFoundError:
    print("この環境に biosnn-bus が無いので、PyPI からインストールします……")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "biosnn-bus"])
    import biosnn_bus  # noqa: F401

# %%
import numpy as np
from biosnn_bus import (
    ModalityPlugin,
    PassThroughMembrane,
    SpikeBus,
    SpikeTrain,
    register_plugin,
)
from biosnn_bus.plugins import DiffImagePlugin

# %% [markdown]
# ## 1. プラグインを定義する
#
# プロジェクト計画書 §2.3 が定めるインターフェースを実装します：`encode` /
# `get_membrane` / `decode` の 3 メソッド、
# および `modality_name` / `spike_dim` の 2 プロパティです。`fusion_channel` と
# `temporal_scale` には既定値があり、必要に応じて上書きします。
#
# ここでは**集団レベル符号化**を使います：`spike_dim` 個のニューロンがそれぞれ
# 一つの信号レベルを「好み」、
# 信号がどのニューロンの担当地域に入るかで発火が決まります。バスが何をしているかを
# 見るのにちょうどよい符号化です。


# %%
# override=True は、このセルを何度でも再実行できるようにするためです（Colab ではよくあります）。
@register_plugin("sine_wave", override=True)
class SineWavePlugin(ModalityPlugin):
    """1 次元信号をレベルに応じて集団スパイクに符号化する（カスタムモダリティの例）。"""

    def __init__(self, spike_dim: int = 32) -> None:
        if spike_dim < 2:
            raise ValueError(
                "spike_dim は 2 以上である必要があります。でないと信号レベルを区別できません。"
            )
        self._spike_dim = spike_dim
        # 各ニューロンが好む信号レベル。[0, 1] に均等に並べる
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    @property
    def modality_name(self) -> str:
        return "sine_wave"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    @property
    def fusion_channel(self) -> str:
        # 時系列は時間チャネルへ、画像プラグインは意味チャネルへ。
        # バスの二重チャネルルーティングはこのプロパティで決まります。
        return "temporal"

    @property
    def temporal_scale(self) -> float:
        return 5.0

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(signal)):
            raise ValueError("入力信号に NaN または Inf が含まれています。")
        # 各時間ステップで、信号レベルに最も近いニューロン（または隣接する 2 個）が発火します。
        # 半径は隣接レベルの半間隔とし、どのような信号値にも少なくとも 1 個の
        # ニューロンが応答するようにします。
        radius = 0.5 / (self._spike_dim - 1)
        distance = np.abs(signal[:, None] - self.levels[None, :])
        data = distance <= radius
        return SpikeTrain(data=data, channel=self.channel, temporal_scale=self.temporal_scale)

    def decode(self, spike_output: SpikeTrain):
        active = spike_output.data.astype(np.float64)
        total = active.sum(axis=1)
        # 発火したニューロンが好むレベルの加重平均。ステップ全体が無発火なら 0
        return np.divide(
            active @ self.levels,
            total,
            out=np.zeros(active.shape[0]),
            where=total > 0,
        )

    def get_membrane(self) -> PassThroughMembrane:
        # スケルトンライブラリに学習則は含まれません。認知コア側がここに実際の
        # スパイキングニューロン層を接続します。
        return PassThroughMembrane()


# %% [markdown]
# ## 2. スパイクバスへ接続する
#
# `SpikeBus` は任意個のプラグインを受け取り、それぞれの `fusion_channel` に従って
# 時間路と意味路に分け、
# 同一の時間グリッドに揃えたうえで融合し、認知コアの入力次元へ射影します。

# %%
bus = SpikeBus(bus_dim=128, seed=0)
bus.register(SineWavePlugin(spike_dim=32))
bus.register(DiffImagePlugin(height=8, width=8, threshold=0.1))

print(bus)
for name, info in bus.describe()["modalities"].items():
    print(f"  {name:12s} dim={info['spike_dim']:4d} channel={info['channel']}")
print("二重チャネルルーティング:", bus.describe()["modalities_by_channel"])

# %% [markdown]
# ## 3. 1 ステップ実行する
#
# 時間チャネルには 1 次元の正弦信号を、意味チャネルには 8×8 の輝度変化列を入力します。

# %%
steps = 24
t = np.linspace(0, 2 * np.pi, steps)
signal = 0.5 + 0.45 * np.sin(t)  # [0.05, 0.95] に収まる

rng = np.random.default_rng(0)
frames = np.zeros((steps, 8, 8))
frames[:, 2:6, 2:6] = 0.8  # 中央の明るいブロック
frames[12:, 0:2, 0:2] = 0.6  # 後半で左上が明るくなる

output = bus.step({"sine_wave": signal, "image_diff": frames})
print("バス出力:", output)
print(
    f"ゼロの割合 {(output.data == 0).mean():.1%}（ランダム射影の重みは ±1 なので、"
    f"出力は入力電流であり 0/1 のスパイクではありません）"
)

# %% [markdown]
# ## 4. スパイクラスタを見る

# %%

import matplotlib.pyplot as plt  # noqa: E402

sine_plugin = bus.get("sine_wave")
image_plugin = bus.get("image_diff")

# 正の電流を持つユニット、つまり実際に押し上げられている認知コアのニューロンを
# 取り出します（前節の説明を参照）。
output_units = output.binary(threshold=0.0)

# 図のラベルは英語のままにします。既定の DejaVu Sans には CJK の字形がなく、
# 非ラテン文字のラベルは多くの環境（Colab を含む）で豆腐になります。
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

for ax, train, title in (
    (axes[0], sine_plugin.encode(signal), "temporal channel - sine_wave (32 units, level code)"),
    (axes[1], image_plugin.encode(frames), "semantic channel - image_diff (128 ch, ON/OFF)"),
    (
        axes[2],
        output_units,
        f"cognitive core input - SpikeBus out ({bus.bus_dim}-d, fused, positive current only)",
    ),
):
    rows, cols = np.nonzero(train.data)
    ax.scatter(cols, rows, s=6, marker="|", linewidths=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("time step")
    ax.set_xlim(-0.5, train.n_neurons - 0.5)

axes[-1].set_xlabel("unit index")
fig.suptitle("biosnn-bus: plugin -> channel -> fusion -> cognitive core input", fontsize=11)
fig.tight_layout()

# ヘッドレスなバックエンド（CI）で show() を呼ぶと警告が出るだけです。
# 対話的なバックエンド（Colab の inline など）でのみ必要です
if plt.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
    plt.show()

# %% [markdown]
# ## 5. モダリティ空間へ復号する
#
# 符号化は可逆である必要はありませんが、元の構造を反映しているべきです。
# ここでは集団レベル符号化を信号値へ復号します。

# %%
restored = sine_plugin.decode(sine_plugin.encode(signal))
print("元の信号の先頭 5 値:  ", np.round(signal[:5], 4))
print("復号後の先頭 5 値:    ", np.round(restored[:5], 4))
print(
    f"最大誤差 {np.abs(restored - signal).max():.4f}（レベル符号化の量子化ステップは {1 / 32:.4f}）"
)

# %% [markdown]
# ## 次のステップ
#
# - サードパーティパッケージにプラグインを提供させたい場合は、その `pyproject.toml` に
#   entry point を宣言します。利用者側は
#   `discover_plugins()` を呼ぶだけです。**BioSNN-Plug のコードは 1 行も変更不要**。
# - 融合戦略を差し替えたい場合は `SpikeBus(strategy=...)` に `FusionStrategy` プロトコルを
#   満たす任意のオブジェクトを渡せます。
#   計画書にある TAAF 時間注意誘導融合は、将来ここに接続されます。
#
# 詳しくは `docs/plugin_guide.ja.md` のプラグイン開発ガイドを参照してください。
