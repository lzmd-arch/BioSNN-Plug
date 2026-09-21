# %% [markdown]
# # Five minutes: registering a custom modality plugin
#
# This example shows the one problem `biosnn-bus` exists to solve: **adding a modality
# requires no change to the existing architecture**.
#
# It does four things in order: write a brand-new modality plugin (encoding a 1-D signal
# as a population of spikes), register it,
# wire it into a `SpikeBus` alongside the bundled image plugin, and look at the
# dual-channel routing and the spike raster.
#
# Everything runs on CPU; the only dependency is numpy (matplotlib for the plot).

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
# ## 1. Define the plugin
#
# Implement the interface the project plan §2.3 specifies: the three methods `encode` /
# `get_membrane` / `decode`,
# plus the two properties `modality_name` / `spike_dim`. `fusion_channel` and
# `temporal_scale` have defaults you override as needed.
#
# This uses **population level coding**: each of the `spike_dim` neurons "prefers" one
# signal level, and
# whichever neuron's range the signal falls into fires — which makes it easy to see what
# the bus is doing.


# %%
# override=True lets this cell be re-run repeatedly (common in Colab).
@register_plugin("sine_wave", override=True)
class SineWavePlugin(ModalityPlugin):
    """Encode a 1-D signal as a population of spikes by level (custom modality example)."""

    def __init__(self, spike_dim: int = 32) -> None:
        if spike_dim < 2:
            raise ValueError("spike_dim must be at least 2, otherwise levels cannot be told apart.")
        self._spike_dim = spike_dim
        # Signal level each neuron prefers, spread evenly over [0, 1]
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    @property
    def modality_name(self) -> str:
        return "sine_wave"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    @property
    def fusion_channel(self) -> str:
        # Time series take the temporal channel; the image plugin takes the semantic one.
        # This property is what the bus's dual-channel routing keys on.
        return "temporal"

    @property
    def temporal_scale(self) -> float:
        return 5.0

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(signal)):
            raise ValueError("input signal contains NaN or Inf.")
        # At each time step the neuron closest to the signal level (or the two neighbours) fires.
        # The radius is half the spacing between adjacent levels, so every signal value gets at
        # least one neuron to respond.
        radius = 0.5 / (self._spike_dim - 1)
        distance = np.abs(signal[:, None] - self.levels[None, :])
        data = distance <= radius
        return SpikeTrain(data=data, channel=self.channel, temporal_scale=self.temporal_scale)

    def decode(self, spike_output: SpikeTrain):
        active = spike_output.data.astype(np.float64)
        total = active.sum(axis=1)
        # Weighted mean of the active neurons' preferred levels; 0 when the whole step is silent
        return np.divide(
            active @ self.levels,
            total,
            out=np.zeros(active.shape[0]),
            where=total > 0,
        )

    def get_membrane(self) -> PassThroughMembrane:
        # The skeleton library contains no learning rules; the cognitive core plugs a real
        # spiking neuron layer in here.
        return PassThroughMembrane()


# %% [markdown]
# ## 2. Wire it into the spike bus
#
# `SpikeBus` takes any number of plugins, splits them into temporal and semantic paths by
# their `fusion_channel`,
# aligns them onto one time grid, then fuses and projects to the cognitive core's input
# dimension.

# %%
bus = SpikeBus(bus_dim=128, seed=0)
bus.register(SineWavePlugin(spike_dim=32))
bus.register(DiffImagePlugin(height=8, width=8, threshold=0.1))

print(bus)
for name, info in bus.describe()["modalities"].items():
    print(f"  {name:12s} dim={info['spike_dim']:4d} channel={info['channel']}")
print("dual-channel routing:", bus.describe()["modalities_by_channel"])

# %% [markdown]
# ## 3. Run one step
#
# The temporal channel takes a 1-D sine signal; the semantic channel takes an 8x8 sequence
# of brightness changes.

# %%
steps = 24
t = np.linspace(0, 2 * np.pi, steps)
signal = 0.5 + 0.45 * np.sin(t)  # stays within [0.05, 0.95]

rng = np.random.default_rng(0)
frames = np.zeros((steps, 8, 8))
frames[:, 2:6, 2:6] = 0.8  # a bright block in the middle
frames[12:, 0:2, 0:2] = 0.6  # top-left lights up in the second half

output = bus.step({"sine_wave": signal, "image_diff": frames})
print("bus output:", output)
print(
    f"zero fraction {(output.data == 0).mean():.1%} (the random projection has +/-1 weights, so"
    f" the output is an input current, not 0/1 spikes)"
)

# %% [markdown]
# ## 4. Look at the spike raster

# %%

import matplotlib.pyplot as plt  # noqa: E402

sine_plugin = bus.get("sine_wave")
image_plugin = bus.get("image_diff")

# Take the units carrying positive current — the cognitive-core neurons actually being
# driven up (see the note in the previous section).
output_units = output.binary(threshold=0.0)

# Plot labels stay in English: the default DejaVu Sans has no CJK glyphs, so non-Latin
# labels render as boxes in most environments
# (Colab included).
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

# A headless backend (CI) only warns on show(); an interactive one (Colab inline) needs it
if plt.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
    plt.show()

# %% [markdown]
# ## 5. Decode back into modality space
#
# Encoding need not be lossless, but it should reflect the original structure. Here the
# population level code is decoded back into signal values.

# %%
restored = sine_plugin.decode(sine_plugin.encode(signal))
print("first 5 values of the original signal:", np.round(signal[:5], 4))
print("first 5 values after decoding:      ", np.round(restored[:5], 4))
print(
    f"max error {np.abs(restored - signal).max():.4f}"
    f" (the level code's quantisation step is {1 / 32:.4f})"
)

# %% [markdown]
# ## Next steps
#
# - Want a third-party package to ship a plugin? Declare an entry point in its
#   `pyproject.toml`; users then call
#   `discover_plugins()`. **No line of BioSNN-Plug needs to change.**
# - Want a different fusion strategy? `SpikeBus(strategy=...)` accepts any object
#   satisfying the `FusionStrategy` protocol.
#   The TAAF temporal-attention-guided fusion in the project plan plugs in here.
#
# See `docs/plugin_guide.en.md` for the full plugin guide.
