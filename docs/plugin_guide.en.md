# Plugin development guide

[中文](plugin_guide.md) · [日本語](plugin_guide.ja.md)

> Every Python code block in this document is **actually executed** by CI (see
> `scripts/check_doc_code_blocks.py`). You can copy and paste them from top to bottom —
> they are contiguous. Fragments that cannot run on their own must be marked
> `python no-run` on the fence.

## What you are writing

A modality plugin answers three questions:

| Method | Question it answers |
| :--- | :--- |
| `encode(raw_input)` | How does my modality's data become spikes? |
| `get_membrane()` | What do the neurons for my modality look like? |
| `decode(spike_output)` | How do spikes turn back into something you can read? |

Plus two pieces of metadata: `modality_name` (what it is called) and `spike_dim` (how many
channels it outputs).

That is all of it. You do not need to care about learning rules, about how the bus fuses
channels, or about what the cognitive core is — those are the business of the skeleton
library and the cognitive core.

## A complete minimal plugin

The following runs as-is. It implements **population-level encoding**: a one-dimensional
signal is encoded into the firing of a group of neurons, and the closer the signal is to a
neuron's preferred level, the more likely that neuron is to fire.

```python
import numpy as np

from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeTrain


class LevelEncoder(ModalityPlugin):
    """Encode a one-dimensional signal as a population of spikes by level."""

    def __init__(self, spike_dim: int = 16) -> None:
        if spike_dim < 2:
            raise ValueError("spike_dim must be at least 2, otherwise signal levels cannot be told apart.")
        self._spike_dim = spike_dim
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    # ---- two properties ----

    @property
    def modality_name(self) -> str:
        return "level"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    # ---- three methods ----

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        radius = 0.5 / (self._spike_dim - 1)  # half the spacing between adjacent levels
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

Try it:

```python
plugin = LevelEncoder(spike_dim=16)
signal = np.linspace(0.0, 1.0, 8)

train = plugin.encode(signal)
print(train)  # shape (8, 16), boolean, channel=temporal
print("decoded:", np.round(plugin.decode(train), 3))
```

### `spike_dim` must match what `encode` actually outputs

The bus checks this. Declare 16 but return 32 channels and registration will not complain,
but the first `step()` will stop you:

```python
from biosnn_bus import PluginContractError, SpikeBus


class WrongDim(LevelEncoder):
    @property
    def spike_dim(self) -> int:
        return 16

    def encode(self, raw_input) -> SpikeTrain:
        return SpikeTrain(data=np.zeros((4, 99), dtype=bool))


bus = SpikeBus(bus_dim=32, seed=0)
bus.register(WrongDim())  # not checked at registration: encode has not been called yet
try:
    bus.step({"level": signal})
except PluginContractError as exc:
    print("rejected as expected:", exc)
```

> Registration cannot check the declaration against the output: `spike_dim` is a static
> declaration, while the output of `encode` is runtime behaviour. So build the output in
> `encode` from `self.spike_dim` and the two cannot drift apart (that is how
> `LevelEncoder.encode` above is written). For why the validation is not placed in
> `__init_subclass__`, see
> [ADR-0006](../docs/adr/ADR-0006-plugin-interface-fidelity.en.md).

## Two optional properties

### `fusion_channel`: which fusion channel to take

The spike bus of project plan §2.2 is **dual-channel**. The default is `'temporal'`; the
other value is `'semantic'`. Which to pick depends on **how this modality's information is
organised**:

| Value | Suits | Example in project plan §2.3 |
| :--- | :--- | :--- |
| `'temporal'` | Information lives mainly in temporal structure (rhythm, order, duration) | Text, audio |
| `'semantic'` | Information lives mainly in content/spatial structure | Images |

```python
class SemanticEncoder(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "semantic"


print(SemanticEncoder().channel)  # prints "semantic"
```

What happens if you get it wrong:

```python
class BadChannel(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "auditory"  # no such channel


try:
    SpikeBus(bus_dim=32).register(BadChannel())
except PluginContractError as exc:
    print("rejected as expected:", exc)
```

### `temporal_scale`: the time scale of this modality

The unit is milliseconds; the default is `10.0`. The project plan defines it as "the typical
time scale of this modality, used to initialise the TAAF module". At the skeleton stage it
travels along with the spike train as a marker, and fusion takes the **maximum** across
modalities (the slowest modality decides the time scale after fusion).

```python
class FastEncoder(LevelEncoder):
    @property
    def temporal_scale(self) -> float:
        return 1.0


print(FastEncoder().temporal_scale)
```

## Contract self-check

`SpikeBus.register()` calls the plugin's `validate()`, and the following four kinds of
problem are stopped at that moment:

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

## Plugging into the bus

```python
bus = SpikeBus(bus_dim=64, seed=0)
bus.register(LevelEncoder(spike_dim=16))

output = bus.step({"level": signal})
print(output)  # shape (8, 64), float32
```

Two things to note:

1. `output.data` is a **floating-point current**, not a 0/1 spike. The sparse random
   projection at the end of the bus carries ±1 weights, and a negative value means
   inhibitory input. Call `output.binary()` when you need spikes.
2. Modality names on the same bus must be unique. To register several configurations of the
   same modality (say image encoders at two resolutions), tell them apart explicitly with
   `name=`:

```python
bus.register(LevelEncoder(spike_dim=16), name="level_low")
bus.register(LevelEncoder(spike_dim=64), name="level_high")
print(bus.registered_modalities)
```

## Third-party plugin integration

**You do not need to change a single line of this repository.** Declare it in your package's
`pyproject.toml`:

```toml
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

Once a user has installed your package:

```python no-run
# This depends on a third-party package that really exists; it is of course not
# installed in this repository, so it cannot be executed.
from biosnn_bus import SpikeBus, discover_plugins, get_plugin

discover_plugins()  # scan and register plugins declared by every installed distribution
bus = SpikeBus(bus_dim=256, seed=0)
bus.register(get_plugin("audio")())
```

`discover_plugins()` is idempotent; calling it repeatedly does not register anything twice.
If an entry point does not point at a `ModalityPlugin` subclass, or raises ImportError while
loading, the error message carries the entry point's name and its import path.

## Test template

Plugins are worth testing: a contract violation should surface **at registration**, not
halfway through training. This is the test structure of this repository's bundled example
plugin `DiffImagePlugin`, and you can copy it as-is
(see `packages/biosnn-bus/tests/test_plugins_image_diff.py`):

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
    """Encoding may be lossy, but the decoded shape must be right and the error must have a clear upper bound."""
    frames = np.zeros((4, 4, 5))
    frames[1, 0, 0] = 0.25  # exactly at the threshold: should trigger one event
    restored = plugin.decode(plugin.encode(frames))
    assert restored.shape == frames.shape
    assert np.abs(restored - frames).max() <= 0.25


def test_plugs_into_the_bus(plugin):
    bus = SpikeBus(bus_dim=32, seed=0)
    bus.register(plugin)
    assert bus.step({"image_diff": np.zeros((6, 4, 5))}).data.shape == (6, 32)
```

**The shape contract and the wrong-shape rejection are the two mandatory ones** — they turn
"the plugin is written wrong" from something that only surfaces midway through training into
a one-line error at registration.

## Going further

- Full signatures and semantics of the plugin interface: `packages/biosnn-bus/src/biosnn_bus/plugin.py`
- Full API of the spike train container `SpikeTrain`: `packages/biosnn-bus/src/biosnn_bus/types.py`
- Why entry points rather than a global registry: `docs/adr/ADR-0003-entry-point-plugin-discovery.en.md`
- What the skeleton library does **not** include (and why): `docs/adr/ADR-0001-skeleton-as-separate-library.en.md`
