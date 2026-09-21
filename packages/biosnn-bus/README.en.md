# biosnn-bus

[中文](README.md) · [日本語](README.ja.md)

**The modality-plugin skeleton library** for [BioSNN-Plug](https://github.com/lzmd-arch/BioSNN-Plug).

It is responsible for exactly one thing: **adding a new modality must not require modifying the existing architecture**.

> **This is the skeleton, not the cognitive core.** This library contains no learning rules
> (e-prop / R-STDP / kernelized IB-Hebbian), no spiking neuron models, and depends on
> neither GPU nor PyTorch. What it provides is the plugin interface contract, the
> registration and discovery mechanism, and the seams of the spike bus.

## Installation

Not published to PyPI yet; install from Git for now:

```bash
pip install "biosnn-bus @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-bus"

# When you need the torch bridge (SpikeTrain.to_torch / from_torch)
pip install "biosnn-bus[torch] @ git+https://github.com/lzmd-arch/BioSNN-Plug.git#subdirectory=packages/biosnn-bus"
```

Depends only on numpy.

## Quickstart

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

For a complete, runnable version see [`examples/quickstart_register_plugin.py`](https://github.com/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.py),
and the [Colab notebook](https://colab.research.google.com/github/lzmd-arch/BioSNN-Plug/blob/main/examples/quickstart_register_plugin.en.ipynb) (runs without a GPU).

## Core concepts

| Concept | Description |
| :--- | :--- |
| `ModalityPlugin` | Abstract base class for modality plugins: three methods `encode` / `decode` / `get_membrane`, two properties `modality_name` / `spike_dim` |
| `SpikeTrain` | Spike-train container of shape `(T, N)`; boolean (spikes) or float (firing rates) both work |
| `FusionChannel` | The two fusion channels `temporal` / `semantic`, corresponding to "dual-channel fusion" in the project plan §2.2 |
| `SpikeBus` | The spike bus: encode → align in time → fuse per channel → sparse random projection to the cognitive-core dimension |
| `register_plugin` | In-process registration decorator |
| `discover_plugins` | Scans the entry points declared by third-party distribution packages |

## How third-party plugins are integrated

Declare an entry point in your package, and **no line of BioSNN-Plug code needs to change**:

```toml
# your package's pyproject.toml
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

The user side scans and attaches it. The snippet below runs as-is — even if not a single third-party plugin is installed:

```python
from biosnn_bus import SpikeBus, discover_plugins, get_plugin, list_plugins
from biosnn_bus.plugins import DiffImagePlugin

discover_plugins()  # scans the plugins declared by third-party distribution packages; no error even if none is installed
print(list_plugins())  # the built-in example plugin is already registered in the registry on import

bus = SpikeBus(bus_dim=128, seed=0)
bus.register(get_plugin("image_diff")())  # equivalent to bus.register(DiffImagePlugin())
print(bus)
```

## The boundaries of the skeleton

The spike bus's current default fusion strategy is **plain concatenation** (`ConcatenateFusion`), **not** the TAAF temporal-attention-guided fusion described in the project plan §2.2 — the latter is a phase-2 research task. The bus already has the `FusionStrategy` protocol and the dual-channel routing in place; replacing the strategy will not change caller code.

See [ADR-0001](https://github.com/lzmd-arch/BioSNN-Plug/blob/main/docs/adr/ADR-0001-skeleton-as-separate-library.md) for the reasoning.

## License

Apache-2.0. Documentation is CC-BY 4.0.
