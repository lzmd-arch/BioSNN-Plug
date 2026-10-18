"""音频插件：对数频带、确定性、静音、单音落带、总线接线。"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import SpikeBus
from biosnn_bus.plugins import AudioCochleaPlugin
from biosnn_bus.plugins.audio_cochlea import log_spaced_bands


def _tone(freq_hz: float, *, seconds: float = 0.3, rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return np.sin(2 * np.pi * freq_hz * t)


class TestBands:
    def test_edges_are_monotonic_and_bounded(self):
        edges = log_spaced_bands(32, 16_000, 512)
        assert edges[0] == 0.0
        assert edges[-1] == pytest.approx(8_000.0)
        assert np.all(np.diff(edges) > 0)

    def test_log_spacing_is_roughly_geometric(self):
        """相邻带的比值应当在同一量级——这是"对数频带"的可检验含义。"""
        edges = log_spaced_bands(16, 16_000, 512)
        widths = np.diff(edges[1:])  # 跳过从 0 起的第一带
        ratios = widths[1:] / widths[:-1]
        assert np.all(ratios > 1.0) and np.all(ratios < 2.0)

    def test_every_band_covers_at_least_one_bin(self):
        edges = log_spaced_bands(64, 16_000, 256)
        bin_hz = 16_000 / 256
        assert np.all(np.diff(edges) >= bin_hz - 1e-9)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="n_bands"):
            log_spaced_bands(0, 16_000, 512)


class TestEncoding:
    def test_shape_and_dim(self):
        plugin = AudioCochleaPlugin(n_bands=16, n_steps=8)
        train = plugin.encode(_tone(440))
        assert train.data.shape == (8, 16)
        assert plugin.spike_dim == 16
        assert plugin.fusion_channel == "temporal"

    def test_same_seed_same_spikes(self):
        a = AudioCochleaPlugin(seed=0).encode(_tone(440))
        b = AudioCochleaPlugin(seed=0).encode(_tone(440))
        assert np.array_equal(a.data, b.data)

    def test_different_seed_differs(self):
        a = AudioCochleaPlugin(seed=0).encode(_tone(440))
        b = AudioCochleaPlugin(seed=1).encode(_tone(440))
        assert not np.array_equal(a.data, b.data)

    def test_silence_produces_no_spikes(self):
        plugin = AudioCochleaPlugin(n_bands=8, n_steps=4)
        train = plugin.encode(np.zeros(4_800))
        assert train.data.sum() == 0.0

    def test_a_tone_peaks_in_the_band_that_contains_it(self):
        """单音的能量应落在包含该频率的那一带——这是"频带分解"的最小可检验性质。"""
        plugin = AudioCochleaPlugin(sample_rate=16_000, n_bands=16, n_steps=8, frame_size=512)
        energies = plugin.band_energies(_tone(2_000))
        per_band = energies.mean(axis=0)
        peak = int(np.argmax(per_band))
        edges = log_spaced_bands(16, 16_000, 512)
        assert edges[peak] <= 2_000 <= edges[peak + 1]

    def test_short_input_is_zero_padded_not_an_error(self):
        plugin = AudioCochleaPlugin(n_bands=8, n_steps=8, frame_size=512)
        train = plugin.encode(np.ones(10))
        assert train.data.shape == (8, 8)

    def test_multi_channel_takes_the_first_column(self):
        plugin = AudioCochleaPlugin(n_bands=4, n_steps=4, frame_size=256)
        stereo = np.stack([_tone(440), np.zeros(4_800)], axis=1)
        mono = plugin.band_energies(_tone(440))
        assert np.allclose(plugin.band_energies(stereo), mono)

    def test_rejects_empty_and_wrong_rank(self):
        plugin = AudioCochleaPlugin()
        with pytest.raises(ValueError, match="空"):
            plugin.encode(np.array([]))
        with pytest.raises(ValueError, match="一维"):
            plugin.encode(np.zeros((2, 3, 4)))


class TestDecode:
    def test_decode_returns_rates_of_the_same_shape(self):
        plugin = AudioCochleaPlugin(n_bands=8, n_steps=4)
        train = plugin.encode(_tone(440))
        rates = plugin.decode(train)
        assert rates.shape == train.data.shape

    def test_decode_rejects_wrong_dim(self):
        plugin = AudioCochleaPlugin(n_bands=8, n_steps=4)
        other = AudioCochleaPlugin(n_bands=4, n_steps=4)
        with pytest.raises(ValueError, match="spike_dim"):
            plugin.decode(other.encode(_tone(440)))


class TestBusIntegration:
    def test_registers_and_produces_the_declared_dim(self):
        bus = SpikeBus(bus_dim=16, seed=0)
        bus.register(AudioCochleaPlugin(n_bands=8, n_steps=6))
        out = bus.step({"audio_cochlea": _tone(440)})
        assert out.data.shape == (6, 16)
