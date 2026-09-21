# ADR-0003: Plugin discovery via entry points rather than a global registry

[中文](ADR-0003-entry-point-plugin-discovery.md) · [日本語](ADR-0003-entry-point-plugin-discovery.ja.md)

- **Status**: Accepted
- **Date**: 2026-09-21
- **Related**: project plan §1.2 core goal 2, §12.6 open-source success metrics; [ADR-0001](ADR-0001-skeleton-as-separate-library.en.md)

## Context

The project plan §1.2 lists "**adding a new modality requires no changes to the existing
architecture**" as one of its core goals, and §12.6 turns it into a measurable metric: by month 24,
"externally contributed modality plugins ≥ 1" (organic goal); by month 36, "a third-party plugin
ecosystem has formed" (vision).

For both of these to hold, the precondition is that **a modality plugin written by someone else
does not need to submit a PR to this repository**. If every added modality required changing the
source of `biosnn_bus`, that is not pluginisation, that is "shoving code into the main repository".

So the real question is not "how do you register a plugin", but "**how does a standalone
third-party package get discovered by this library**".

There is a second constraint as well: the phase 0 success criteria in the project plan §7 require a
demonstration of "registering a custom modality plugin", with the user defining a plugin in their
own script and using it immediately — that route has to be short as well.

## Decision

**Two channels coexist**:

1. **In-process registration**: the `@register_plugin("name")` decorator, which writes into a
   module-level global registry. Suitable for same-repository plugins and interactive exploration;
   the shortest path.
2. **Entry point discovery**: `discover_plugins()` uses
   `importlib.metadata.entry_points(group="biosnn_bus.plugins")` to scan installed third-party
   distribution packages.

How a third-party package plugs in:

```toml
# the third-party package's own pyproject.toml — no line of BioSNN-Plug code needs to change
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

`discover_plugins()` is **idempotent**: an entry with a name that is already registered is skipped,
so you can call it repeatedly without worry. If an entry point points at something that is not a
`ModalityPlugin` subclass, or raises ImportError while loading, the error message carries the entry
point's name and value — a failed import in a third-party package should not show up as "the plugin
mysteriously disappeared".

## Consequences

**Benefits**

- Third-party plugins are **fully independent**: their own repository, their own CI, their own
  release cadence. This repository changes by zero lines;
- They automatically get all the benefits of the Python packaging ecosystem (version constraints,
  dependency resolution, private-index distribution);
- Combining a global registry with entry points lets the two paths share the same lookup logic
  (`get_plugin`); you do not need to know where a plugin came from.

**Costs (accepted)**

- **Global mutable state**: the module-level registry is process-wide. Tests must be isolated from
  one another (see the `isolated_registry` fixture in
  `packages/biosnn-bus/tests/conftest.py`), otherwise a plugin registered by one test leaks into
  the next. This is a real pitfall, and it is already handled in the test fixture;
- A name collision errors out by default rather than silently overwriting (only `override=True`
  overwrites). When two modality plugins collide on a name, the developer knows at the moment of
  registration, rather than discovering halfway through training that they are using someone else's
  encoder;
- The `@register_plugin` decorator is an **import side effect** — if you do not import the plugin
  module, nothing is registered. This is the standard practice for Python plugins, and the error
  message from `get_plugin` specifically points you at `discover_plugins()`.

**What this does not settle**

- There is no plugin version compatibility check. It is enough for a third-party plugin to declare
  a requirement of `biosnn-bus>=0.1`; the skeleton library does not verify which version of the API
  the plugin actually used. That can wait until there really is a third-party plugin.

## Alternatives

**A. Only the decorator-based global registry.**
Rejected. It forces a "third-party plugin" to `import` this library before it can itself be
imported, which in practice rules out standalone distribution — community members could only
submit PRs, exactly the situation to be avoided.

**B. Only entry points.**
Rejected. Even writing a plugin you can use immediately would first mean creating a package and
installing it; that is too heavy for both interactive exploration and example demos. What the
project plan §7 phase 0 asks for is precisely the short path of "register a plugin in 5 minutes".

**C. Plugins declare a path through a configuration file (YAML/JSON).**
Rejected. It sidesteps Python's packaging and dependency management: you would have to make sure
yourself that the dependencies are installed and the path is right, and static check tools cannot
see these strings at all.

**D. Use an off-the-shelf plugin framework such as `pluggy` (pytest's plugin system).**
Rejected (at this phase). It adds one more runtime dependency, when all we need is the single
capability of "find a class by name", and `importlib.metadata` is in the standard library. Revisit
it if a complex hook requirement actually appears.
