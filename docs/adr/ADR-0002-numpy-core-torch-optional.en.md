# ADR-0002: Skeleton library core on numpy, torch as an optional bridge

[中文](ADR-0002-numpy-core-torch-optional.md) · [日本語](ADR-0002-numpy-core-torch-optional.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-21
- **Related**: project plan §7 (phase 0 success criteria), §1.4 (technical constraints); [ADR-0001](ADR-0001-skeleton-as-separate-library.en.md)

## Context

Among the hard success criteria the project plan §7 sets for the skeleton library in phase 0 is
this one:

> The demo runs in a Colab environment **without a GPU**.

Meanwhile, project plan §1.4 and the technology-selection table in §8 fix the whole project's
stack as **SpikingJelly + PyTorch + eprop-PyTorch**. The cognitive core will use torch, without
question.

The question is whether the skeleton library has to follow suit.

The `torch` wheel is about 2GB (the CUDA build). If `pip install biosnn-bus` drags it in, then:

- someone who only wants to write a modality plugin has to download 2GB;
- in Colab, `pip install` takes several minutes, and "get the demo running in 5 minutes" is
  ruined outright;
- the skeleton library contradicts its own positioning as a "standalone lightweight library".

There is a less visible problem too: the interface specification in the project plan §2.3 says

```python no-run
@abstractmethod
def get_membrane(self) -> nn.Module: ...
```

`nn.Module` is a torch type. As long as it appears in the signature, the skeleton library must
`import torch`, and there is no way around the dependency.

## Decision

**The skeleton library core depends on numpy only.**

1. `SpikeTrain` stores numpy arrays internally and is backend-agnostic; torch tensors are
   bridged through `SpikeTrain.to_torch()` / `SpikeTrain.from_torch()`, implemented under the
   optional extra `biosnn-bus[torch]` with a lazy import.
2. The return type of `get_membrane` changes from `nn.Module` to the structural protocol
   [`Membrane`](../../packages/biosnn-bus/src/biosnn_bus/types.py) — it only requires that a
   `forward` method exist. The `nn.Module` subclasses the cognitive core returns
   **automatically satisfy** that protocol structurally, and plugin authors' code is
   unaffected. See [ADR-0006](ADR-0006-plugin-interface-fidelity.en.md) for details.
3. CI asserts this explicitly: after installing the wheel into a clean venv, `torch` must
   **not** be present.

## Consequences

**Benefits**

- `pip install biosnn-bus` pulls only numpy and finishes in seconds;
- the Colab demo needs no GPU and no waiting, so the phase 0 hard criterion is met directly;
- the skeleton library can be tested in any Python environment, and a CI matrix running
  3.10/3.11/3.12 is no burden at all;
- it forces a good outcome in return: the interface **must** be decoupled from any particular
  tensor library. If you later want to switch to JAX or talk to Loihi/Lava directly, the
  interface does not have to change.

**Costs (accepted)**

- moving data between libraries requires an explicit conversion (`.to_torch()`), one step more
  than passing a tensor directly;
- the `Membrane` protocol is a structural type: an `isinstance` check only looks at whether
  `forward` exists, and cannot catch a signature mismatch. This is an inherent limitation of
  Python protocols;
- the numeric operations in the bus run on numpy and are slower than torch. At the skeleton
  phase the data volumes are small, so this is acceptable; if it ever does become a bottleneck,
  the cognitive core side can bypass `SpikeBus` and take the raw `SpikeTrain` directly.

**Explicit non-goals**

- the skeleton library is **not** a lightweight SNN simulator. It does no neuron dynamics and
  no learning rules. Those are the business of the cognitive core and SpikingJelly.

## Alternatives

**A. The core depends on torch.**
Rejected. It directly violates the phase 0 hard success criterion, and a 2GB install size
conflicts with the positioning as a "standalone lightweight library".

**B. The core depends on no array library at all, using plain Python lists.**
Rejected. Spike data is a dense `(T, N)` array; plain Python would be too slow to run a demo,
and numpy itself is only a dozen-odd MB, which is not a burden.

**C. Use `array-api-compat` for backend-agnostic abstraction.**
Rejected (at this stage). What it solves is "supporting numpy/torch/jax at the same time",
while all we need is "numpy in the core, bridging torch at the boundary". One more layer of
abstraction buys no real benefit; introducing it once multiple backends are actually needed is
soon enough.
