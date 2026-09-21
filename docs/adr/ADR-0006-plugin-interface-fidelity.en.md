# ADR-0006: `ModalityPlugin` keeps the original signature from §2.3 of the project plan

[中文](ADR-0006-plugin-interface-fidelity.md) · [日本語](ADR-0006-plugin-interface-fidelity.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-21
- **Related**: project plan §2.3 (modality plugin interface specification); [ADR-0002](ADR-0002-numpy-core-torch-optional.en.md)

## Context

Project plan §2.3 gives the complete interface definition of `ModalityPlugin`, including three
methods and two properties, plus two attributes with default values. This specification is the
foundation of the whole plugin design, and it is one of the few parts of the project plan that
is written specifically enough to be copied directly.

But [ADR-0002](ADR-0002-numpy-core-torch-optional.en.md) decides that the skeleton library core
does not depend on torch, and the specification in the project plan has this line:

```python no-run
@abstractmethod
def get_membrane(self) -> nn.Module: ...
```

`nn.Module` is a torch type. Copy it as written, and the skeleton library must `import torch`.

So there is a real trade-off: **change the interface, or change the type hint?**

## Decision

**The signature is not changed by a single character; only the return type becomes a structural
protocol.**

All five members are implemented exactly as the project plan has them:

| Member | Text in the project plan | Skeleton library implementation |
| :--- | :--- | :--- |
| `encode(raw_input)` | abstract method | identical |
| `get_membrane()` | abstract method, returns `nn.Module` | returns the `Membrane` protocol (**the only deviation**) |
| `decode(spike_output)` | abstract method | identical |
| `modality_name` | abstract property | identical |
| `spike_dim` | abstract property | identical |
| `temporal_scale` | default `10.0` | identical |
| `fusion_channel` | default `'temporal'` | identical |

`Membrane` is a `runtime_checkable` `Protocol` that only requires a `forward` method to exist:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Membrane(Protocol):
    def forward(self, x): ...
```

The `nn.Module` subclasses the cognitive core returns **automatically satisfy** this protocol
structurally — `nn.Module` already requires subclasses to implement `forward`. So for the
cognitive core and for plugin authors, how you write the code has not changed at all.

Two further members that the project plan does not have are also added; both are pure additions
and do not affect compatibility:

- the `channel` property: the enumeration form of `fusion_channel`, for use inside the bus. It
  also carries the validation duty — when `fusion_channel` returns an illegal string, the error
  is raised here;
- the `validate()` method: a self-check of the interface contract. `SpikeBus.register()` calls
  it.

## Consequences

**Benefits**

- the interface specification in the project plan is still the external contract, so reading the
  project plan and reading the code shows you the same set of things;
- plugin authors who write their code the way the project plan describes will find it just
  runs; they do not need to understand the relationship between Protocol and `nn.Module`;
- the skeleton library keeps zero torch dependencies.

**Costs (accepted)**

- **weaker type checking**: `Membrane` is a structural protocol, so `isinstance(x, Membrane)`
  only checks whether a `forward` attribute exists and cannot catch a signature mismatch (say,
  `forward(self)` with one parameter too few). This is an inherent limitation of Python
  Protocol, not something this decision introduces;
- strictly speaking this is **a deviation from the project plan**, if a small one. Someone
  reading the code who sees `Membrane` rather than `nn.Module` should be able to find the reason
  here, rather than assuming something was copied wrongly.

**Why validation lives in `validate()` rather than `__init_subclass__`**

The discussions around the project plan once imagined validating constraints such as `spike_dim`
inside `__init_subclass__`. That cannot be done: they are **instance attributes**, and at class
definition time there is no value to check. There are two real opportunities — at instantiation,
or at registration. Registration was chosen (`SpikeBus.register` calls `validate()`), because:

- an ABC should not force subclasses to implement `__init__` (many plugins do not need one);
- registration is the only entry point through which a plugin enters the system, so intercepting
  there covers the most ground, and the error message carries the most context (it can say which
  bus rejected it).

## Alternatives

**A. Copy the project plan exactly, `get_membrane() -> nn.Module`.**
Rejected. It directly violates [ADR-0002](ADR-0002-numpy-core-torch-optional.en.md), and in turn
violates the phase 0 hard success criterion of "running in Colab without a GPU".

**B. Remove `get_membrane` and let the cognitive core manage the membrane potential itself.**
Rejected. This is a **substantive** change to the interface in the project plan, not an
adjustment at the type level. `get_membrane` is the only way for a plugin to expose to the
cognitive core "what the neurons for my modality look like"; remove it, and the cognitive core
would in turn have to know the internal structure of every modality, and the boundary of
pluginization breaks.

**C. Type the return value as `Any`.**
Rejected. It looks like the least effort and is in fact the worst: the type checker stops doing
anything at all, and the reader does not know what the returned value is supposed to satisfy.
The Protocol at least writes the contract down, and it is executable documentation.

**D. Define your own `Membrane` abstract base class (ABC) and require plugins to inherit from
it.**
Rejected. Then an `nn.Module` subclass **could not** be returned directly, and the cognitive core
would have to write an adapter layer. The benefit of a structural protocol is precisely that no
explicit inheritance is needed.
