"""Text modality plugin: frozen local embedding + z-score top-k population coding.

Plan section 2.3 modality table, text row; fusion channel **temporal** (sequence-led
modality -- token order lives in the time structure; the criterion is in
`docs/plugin_guide.md`). The coding scheme is specified in plan section **4.3** and is
attributed to [Savage, 2026] (`docs/references.md` entry [9], EMBER). Section 8 row 17
of the plan selects the same scheme ("embedding coding = z-score top-k").

## What the coding does

For each token with frozen embedding e:

1. cosine similarity against N sensory-neuron preference directions p_i, then z-score
   z_i = (cos(e, p_i) - mu) / sigma, where mu/sigma are the mean and std of all of them;
2. keep the top-k neurons (k = floor(s * N), sparsity s = 0.14, sparse-distributed-memory);
3. scale firing rates by r_i = max(z_i, 0) / max(z_top-k) * r_max.

**Dimension independence** is the selling point: the encoded N does not depend on the
embedding dimension, so swapping the embedding model (768 / 1024 / 4096) does not change
downstream code. EMBER measured 82.2% discriminability retained at 1024 dims and 83.8%
at 384 (a 1.6% difference).

## Three implementation choices (plan section 4.3 does not fix these; this does)

1. **Preference directions p_i are fixed-seed random unit vectors**, not learned -- the
   same trade-off as ADR-0012 option **A (fixed random projection)**: zero training,
   bit-for-bit reproducible, seed goes into the config dump. **Do not learn p_i** -- that
   would make the coding layer require training and puncture the pure-local-learning claim.
2. **One token = one time step** (T = token count). Text is a sequence; this matches the
   **temporal** channel.
3. **Tokenization is whitespace splitting** (text.split()), no tokenizer dependency.

**The inherent cost of this scheme**: nomic-embed is a **sentence** embedding model, so
per-token embedding gets "this word as a standalone sentence" and **loses context**. This
is the cost of the (a) frozen-embedding route, not an implementation defect. Getting
context requires embedding the whole span (T = 1, degenerating to non-sequential) or
switching to a model with per-token outputs. Path (b) (character/token-level sparse coding
from scratch) is downgraded to a **purification exploration item** per plan revision note.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from ..membrane import PassThroughMembrane
from ..plugin import ModalityPlugin
from ..registry import register_plugin
from ..types import SpikeTrain

__all__ = ["EmbeddingBackend", "OllamaEmbedding", "TextZScorePlugin"]

#: plan section 4.3: sparsity s = 0.14 (sparse distributed memory)
DEFAULT_SPARSITY = 0.14
#: population size after coding. Independent of embedding dim.
DEFAULT_N_NEURONS = 256


class EmbeddingBackend:
    """Minimal swappable interface: turn text into a frozen embedding vector.

    Implementers need only `embed(text) -> np.ndarray`. The skeleton default is
    `OllamaEmbedding`; tests use a fake backend so they do not depend on whether Ollama
    is installed on the machine.
    """

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError


class OllamaEmbedding(EmbeddingBackend):
    """Fetch a frozen embedding over the local Ollama HTTP API.

    Args:
        model: model name (default nomic-embed-text, present locally, 768 dims).
        base_url: Ollama address (default http://127.0.0.1:11434).
        timeout: per-request timeout in seconds.

    Note:
        Uses /api/embed (the batch endpoint), not /api/embeddings -- the latter takes one
        input at a time. **This is a frozen front end**: its parameters never enter a
        gradient and never take part in local learning, so it does not affect the
        pure-local-learning claim (which targets **synaptic learning rules**, not the
        input encoder).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed("warm up").shape[0])
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"cannot fetch embedding ({self.model} @ {self.base_url}): {error}."
                " Is Ollama running? Is the model imported?"
            ) from error
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding count mismatch: want {len(texts)}, got {len(vectors or [])}."
            )
        return [np.asarray(v, dtype=np.float64) for v in vectors]


@register_plugin("text_zscore")
class TextZScorePlugin(ModalityPlugin):
    """Frozen embedding -> z-score top-k population coding -> spike train.

    Args:
        n_neurons: population size N. **Independent of embedding dim**.
        sparsity: sparsity s, with k = floor(s * N). Default 0.14 (plan 4.3).
        n_steps: time steps each token occupies. Default 1 (one token, one step).
        r_max: upper bound for firing-rate scaling.
        backend: an EmbeddingBackend; None means OllamaEmbedding.
        seed: seed for the preference directions p_i. Fixed seed is the precondition of
            reproducibility.

    Note:
        spike_dim = n_neurons -- independent of embedding dimension, which is exactly
        what dimension independence means.
    """

    def __init__(
        self,
        n_neurons: int = DEFAULT_N_NEURONS,
        *,
        sparsity: float = DEFAULT_SPARSITY,
        n_steps: int = 1,
        r_max: float = 1.0,
        backend: EmbeddingBackend | None = None,
        seed: int = 0,
    ) -> None:
        for label, value in (("n_neurons", n_neurons), ("n_steps", n_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive int, got {value!r}.")
        if not 0 < sparsity < 1:
            raise ValueError(f"sparsity must be in (0, 1), got {sparsity!r}.")
        if not r_max > 0:
            raise ValueError(f"r_max must be positive, got {r_max!r}.")

        self.n_neurons = n_neurons
        self.sparsity = float(sparsity)
        self.n_steps = int(n_steps)
        self.r_max = float(r_max)
        self.backend = backend
        self.seed = int(seed)
        self._preferences: dict[int, np.ndarray] = {}

    @property
    def modality_name(self) -> str:
        return "text_zscore"

    @property
    def spike_dim(self) -> int:
        return self.n_neurons

    @property
    def fusion_channel(self) -> str:
        # Text is a sequence-led modality (token stream) -> temporal channel.
        return "temporal"

    @property
    def temporal_scale(self) -> float:
        return float(self.n_steps)

    def encode(self, raw_input: Any) -> SpikeTrain:
        """str or list[str] -> (T, N) firing-rate train.

        A str is split on whitespace into tokens; one token = one time step
        (T = tokens * n_steps). A list[str] is taken as tokens directly.
        """
        tokens = self._tokenize(raw_input)
        backend = self._backend()
        vectors = self._embed(backend, tokens)

        rows = []
        for vector in vectors:
            rates = self._encode_one(vector)
            for _ in range(self.n_steps):
                rows.append(rates)
        data = np.stack(rows, axis=0) if rows else np.zeros((1, self.n_neurons))
        return SpikeTrain(data=data, channel=self.channel, temporal_scale=self.temporal_scale)

    def decode(self, spike_output: SpikeTrain) -> np.ndarray:
        """(T, N) train -> (T, N) firing rates.

        **Not** a decoding back to text or to the embedding. This coding emits
        population firing rates; inverting it to text is ill-posed (top-k truncation
        loses information). It returns the rates themselves, for sanity checks and for
        downstream readout.
        """
        if not isinstance(spike_output, SpikeTrain):
            raise TypeError(f"decode needs a SpikeTrain, got {type(spike_output).__name__}.")
        if spike_output.n_neurons != self.spike_dim:
            raise ValueError(
                f"decode expects spike_dim={self.spike_dim}, got {spike_output.n_neurons}."
            )
        return np.asarray(spike_output.data, dtype=np.float64)

    def get_membrane(self) -> PassThroughMembrane:
        """The skeleton library ships no learning rule; this is a placeholder."""
        return PassThroughMembrane()

    def _encode_one(self, vector: np.ndarray) -> np.ndarray:
        """z-score top-k population coding for one embedding -> (N,) rates."""
        preferences = self._preferences_for(vector.shape[0])
        cosine = preferences @ vector / (np.linalg.norm(vector) + 1e-12)

        mu = float(cosine.mean())
        sigma = float(cosine.std())
        z = (cosine - mu) / sigma if sigma > 0 else np.zeros_like(cosine)

        k = max(1, int(self.sparsity * self.n_neurons))
        top = np.argpartition(z, -k)[-k:]
        peak = float(z[top].max())
        rates = np.zeros(self.n_neurons, dtype=np.float64)
        if peak > 0:
            rates[top] = np.maximum(z[top], 0.0) / peak * self.r_max
        return rates

    def _preferences_for(self, dim: int) -> np.ndarray:
        """(N, dim) preference directions: fixed-seed random unit vectors.

        Cached per dim -- swapping the embedding model (768 -> 1024) swaps the preference
        directions but keeps N, so the coding stays dimension independent.
        """
        cached = self._preferences.get(dim)
        if cached is not None:
            return cached
        rng = np.random.default_rng(self.seed)
        raw = rng.standard_normal((self.n_neurons, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        unit = raw / np.where(norms > 0, norms, 1.0)
        self._preferences[dim] = unit
        return unit

    def _backend(self) -> EmbeddingBackend:
        if self.backend is None:
            self.backend = OllamaEmbedding()
        return self.backend

    def _embed(self, backend: EmbeddingBackend, tokens: list[str]) -> list[np.ndarray]:
        batch = getattr(backend, "embed_batch", None)
        if callable(batch):
            return batch(tokens)
        return [backend.embed(token) for token in tokens]

    def _tokenize(self, raw_input: Any) -> list[str]:
        if isinstance(raw_input, str):
            tokens = raw_input.split()
        elif isinstance(raw_input, (list, tuple)) and all(isinstance(x, str) for x in raw_input):
            tokens = list(raw_input)
        else:
            raise ValueError(
                f"{type(self).__name__} expects str or list[str], got {type(raw_input).__name__}."
            )
        tokens = [tok for tok in tokens if tok.strip()]
        if not tokens:
            raise ValueError("text must have at least one non-empty token.")
        return tokens

    def __repr__(self) -> str:
        return (
            f"TextZScorePlugin(n_neurons={self.n_neurons}, sparsity={self.sparsity:g}, "
            f"n_steps={self.n_steps}, seed={self.seed})"
        )
