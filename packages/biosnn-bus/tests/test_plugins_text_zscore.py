"""Tests for TextZScorePlugin: coding math, dimension independence, and the temporal channel.

These pin the properties that make the plugin worth having -- not just "it runs":

* sparsity is exactly floor(s * N) neurons per step (plan section 4.3);
* spike_dim is independent of the embedding dimension;
* one token = one time step (the temporal channel is a sequence);
* the coding is reproducible under a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import SpikeBus, SpikeTrain, TemporalAttentionFusion
from biosnn_bus.plugins import MnistPulsePlugin, TextZScorePlugin

N_NEURONS = 256
SPARSITY = 0.14
K = int(SPARSITY * N_NEURONS)  # 35


class FakeBackend:
    """Deterministic embedding backend -- tests must not depend on Ollama being installed."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self.calls += 1
        rng = np.random.default_rng(abs(hash(texts[0])) % (2**32))
        return [rng.standard_normal(self.dim) for _ in texts]


@pytest.fixture
def plugin() -> TextZScorePlugin:
    return TextZScorePlugin(
        n_neurons=N_NEURONS, sparsity=SPARSITY, backend=FakeBackend(dim=32), seed=0
    )


class TestParameters:
    @pytest.mark.parametrize("n_neurons", [0, -1, 1.5, True])
    def test_rejects_bad_n_neurons(self, n_neurons):
        with pytest.raises(ValueError, match="n_neurons"):
            TextZScorePlugin(n_neurons=n_neurons, backend=FakeBackend())

    @pytest.mark.parametrize("sparsity", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_bad_sparsity(self, sparsity):
        with pytest.raises(ValueError, match="sparsity"):
            TextZScorePlugin(sparsity=sparsity, backend=FakeBackend())

    def test_spike_dim_is_population_size(self, plugin):
        assert plugin.spike_dim == N_NEURONS

    def test_channel_is_temporal(self, plugin):
        assert plugin.channel == "temporal"


class TestCoding:
    def test_exactly_k_neurons_fire_per_step(self, plugin):
        """Plan 4.3: k = floor(s * N) neurons per step. Here 35 of 256."""
        train = plugin.encode("one two three")
        data = np.asarray(train.data)
        fired = {int((data[t] != 0).sum()) for t in range(train.n_steps)}
        assert fired == {K}, f"expected exactly {K} neurons per step, got {fired}"

    def test_one_token_one_time_step(self, plugin):
        assert plugin.encode("a b c d").n_steps == 4

    def test_rates_are_in_unit_range(self, plugin):
        data = np.asarray(plugin.encode("hello world").data)
        assert data.min() >= 0.0
        assert data.max() <= 1.0

    def test_peak_rate_hits_r_max(self):
        plugin = TextZScorePlugin(
            n_neurons=N_NEURONS, sparsity=SPARSITY, r_max=0.7, backend=FakeBackend(32), seed=0
        )
        data = np.asarray(plugin.encode("hello").data)
        assert data.max() == pytest.approx(0.7)

    def test_dimension_independence(self):
        """Different embedding dims, same N -- that is what dimension independence means."""
        dims = {}
        for dim in (16, 32, 128, 768):
            plugin = TextZScorePlugin(
                n_neurons=N_NEURONS, sparsity=SPARSITY, backend=FakeBackend(dim=dim), seed=0
            )
            dims[dim] = plugin.encode("same text").n_neurons
        assert set(dims.values()) == {N_NEURONS}

    def test_fixed_seed_is_reproducible(self):
        def run():
            plugin = TextZScorePlugin(
                n_neurons=N_NEURONS, sparsity=SPARSITY, backend=FakeBackend(32), seed=7
            )
            return np.asarray(plugin.encode("the cat sat").data)

        assert np.array_equal(run(), run())


class TestInputs:
    def test_list_of_tokens_works(self, plugin):
        assert plugin.encode(["a", "b"]).n_steps == 2

    def test_rejects_non_text(self, plugin):
        with pytest.raises(ValueError, match="str or list"):
            plugin.encode(123)

    def test_rejects_empty(self, plugin):
        with pytest.raises(ValueError, match="token"):
            plugin.encode("   ")

    def test_decode_returns_rates_not_text(self, plugin):
        """decode returns (T, N) rates -- the coding emits rates, not text."""
        train = plugin.encode("hello world")
        out = plugin.decode(train)
        assert out.shape == (2, N_NEURONS)  # two tokens -> two time steps
        assert isinstance(out, np.ndarray)

    def test_decode_rejects_wrong_dim(self, plugin):
        with pytest.raises(ValueError, match="spike_dim"):
            plugin.decode(SpikeTrain(data=np.zeros((1, 5)), channel="temporal"))


class TestWithBus:
    def test_multimodal_fusion_closes_the_taaf_gap(self):
        """Text + image through TAAF -- this is what gives TAAF a real caller."""
        bus = SpikeBus(bus_dim=128, seed=0, strategy=TemporalAttentionFusion())
        bus.register(TextZScorePlugin(n_neurons=N_NEURONS, seed=0, backend=FakeBackend(32)))
        bus.register(MnistPulsePlugin(height=28, width=28, n_steps=4, seed=0))

        frames = np.zeros((4, 28, 28))
        frames[0, 5:15, 5:15] = 0.9
        out = bus.step({"text_zscore": "the cat sat", "mnist_pulse": frames})
        assert out.n_neurons == 128 == bus.bus_dim
        assert out.channel == "temporal"  # text is temporal; image is semantic -> summed

    def test_projection_seeds_are_dumped(self):
        bus = SpikeBus(bus_dim=64, seed=0, strategy=TemporalAttentionFusion())
        bus.register(TextZScorePlugin(n_neurons=N_NEURONS, seed=0, backend=FakeBackend(32)))
        bus.step({"text_zscore": "hello world"})
        assert bus.describe()["projection_seeds"]
