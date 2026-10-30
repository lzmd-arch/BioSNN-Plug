# Contributing

[中文](CONTRIBUTING.md) · [日本語](CONTRIBUTING.ja.md)

Thank you for being willing to spend the time. Three things you need to know first, so that the
effort is not wasted:

1. **This is a research prototype, not a production framework.** Some features that look
   "obviously worth adding" will be shelved, on the grounds that they are not on the route in the
   project plan.
2. **One maintainer, and no SLA on issue or PR response.** Reports with complete reproduction
   steps get priority.
3. **Writing a modality plugin does not require a PR here.** A separate package is enough — see
   the "Third-party plugins" section below.

## Development environment

You need [uv](https://docs.astral.sh/uv/) (Python package and project manager) and Python **3.11+** — the root project's floor comes from SpikingJelly ([ADR-0008](docs/adr/ADR-0008-spikingjelly-license-and-python-floor.en.md)); the skeleton package itself still declares 3.10+, covered by CI's 3.10 leg.

```bash
git clone https://github.com/lzmd-arch/BioSNN-Plug.git
cd BioSNN-Plug
uv sync                    # installs every dev dependency in one command (skeleton library included, editable)
uv run pre-commit install  # installs the commit hooks (fetched from the network on first run)
```

To verify the environment:

```bash
uv run pytest -q                                  # all tests pass
uv run python examples/quickstart_register_plugin.py   # the example runs
```

## Pre-submit checklist

The project plan §12.5 requires a "**citation spot-check + code-block check**" before a PR.
Confirm each item:

```bash
uv run pre-commit run --all-files    # every hook, in one pass
uv run pytest -q                     # unit tests are not a hook -- run them separately
```

| Check | Script | Blocks what |
| :--- | :--- | :--- |
| Style | `ruff check` / `ruff format` | Formatting and common bugs |
| Documentation code blocks | `scripts/check_doc_code_blocks.py` | Code in the documentation that does not run |
| Version number consistency | `scripts/check_version_consistency.py` | Version numbers in filenames and content out of step |
| Reference list | `scripts/check_references.py` | Numbering gaps, dead-link format, duplicate entries |
| Notebook sync | `scripts/build_notebook.py` | Drift between the Colab notebook and the example script |
| Markdown table structure | `scripts/check_md_tables.py` | broken rows, pipes swallowing content |
| Tri-language consistency | `scripts/check_translations.py` | translation drift, structure mismatch |
| Gap-report reconciliation | `scripts/gap_report.py --check` | gap numbers in docs drifting from the record |

CI runs all of it again, and adds a **lychee link liveness check** and a **dependency license
audit**.

### Documentation code blocks: executed by default, exceptions marked explicitly

`scripts/check_doc_code_blocks.py` **really does execute** the Python code blocks in Markdown.
The reason is that a pure syntax check (`ast.parse`) cannot catch the class of problem the
project plan §12.3 calls out — `self.threshold` is perfectly legal syntax; the error is that the
attribute does not exist at run time.

- Code blocks within one document **share a namespace and execute in order of appearance**. When
  you write a tutorial you can keep writing straight down, the way you would write a script;
- For a snippet that cannot run on its own (it depends on a third-party package, or it is only
  illustrative), mark it on the fence:

  ````text
  ```python no-run
  bus.register(get_plugin("audio")())   # depends on a third-party package this repository does not have
  ```
  ````

**`no-run` is an escape hatch for code that by definition cannot run, not a back door around the
check.** If a piece of code could run inside this repository but is marked `no-run`, review will
ask you to change it back.

### Citation spot-check

Whenever you add or modify an external citation:

Register it in [`docs/references.en.md`](docs/references.en.md) and state its **verification
status**.

The verification rules (which four things to check, what each status means, what to do when you
cannot verify something) are defined by that file — it is the single home of the citation rules.
An attribution error really did happen once in this repository, and the cause is recorded there.

## Change types and compatibility

| Change | Requirement |
| :--- | :--- |
| Skeleton library `packages/biosnn-bus/` | Follows semver. A breaking change requires a version number bump; prefer adding new capability, do not change existing signatures |
| Research code `research/` | No compatibility promises for the first 12 months (until 2027-09); refactor freely |
| Documentation / toolchain | No special requirements |

The skeleton library is currently `0.x`. By semver convention, minor version numbers in `0.x` may
contain breaking changes — this is deliberate, so that it can iterate quickly while it has no
external users yet. The first version that something outside depends on will be bumped to `1.0.0`.

### Made a significant trade-off?

If this PR makes a decision that someone later will ask "why not write it that way", **add an
ADR**. The format and the criteria for deciding are in
[`docs/adr/README.en.md`](docs/adr/README.en.md).

Once an ADR is recorded, its body is not edited — when you change your mind, write a new one and
link the two to each other.

## PR process

1. Fork, or open a branch (suggested branch names: `feat/...`, `fix/...`, `docs/...`);
2. Make the change, plus tests. **A bug fix must come with a test that reproduces the bug**;
3. Get the pre-submit checklist passing;
4. Open the PR, and fill in the checklist in the PR template.

CI must be fully green. CI currently covers Python 3.10 / 3.11 / 3.12, all running on CPU — the
skeleton library deliberately does not depend on a GPU (see
[ADR-0002](docs/adr/ADR-0002-numpy-core-torch-optional.en.md)), so if you find that your change
needs a GPU to be tested, that most likely means the abstraction level is wrong.

## Third-party plugins

**Do not open a PR against this repository.** Declare an entry point in your own `pyproject.toml`
(with the group `biosnn_bus.plugins`) and that is enough; zero changes to this repository.

For how to write the configuration and how users install it, see
[`docs/plugin_guide.en.md`](docs/plugin_guide.en.md#third-party-plugin-integration).

If you have built a modality plugin, you are welcome to open an issue and tell people about it.

## Release

`biosnn-bus` is released through [`.github/workflows/release.yml`](.github/workflows/release.yml):
pushing a `v<version>` tag triggers it. The process, and the PyPI-side configuration that must be
done before the first release (registering trusted publishing), are both written in the comment at
the top of that file.

Maintainer summary:

```bash
# 1. Change the version in packages/biosnn-bus/pyproject.toml and sync __version__ in __init__.py
#    (scripts/check_version_consistency.py checks these two plus the project plan's filename)
# 2. After committing and pushing, tag it
git tag v0.1.0 && git push origin v0.1.0
```

The `verify` job first confirms that the tag name matches the package version — a wrong tag is
stopped at this step, and the "released v0.2.0 but the package contents are still 0.1.0" situation,
which is hard to remedy after the fact, does not arise.

Until PyPI is configured, you can trigger the pipeline manually with
`gh workflow run release.yml`: it only runs validation and the build, it does not publish.

`biosnn-mcp` has its **own** pipeline, [`release-mcp.yml`](.github/workflows/release-mcp.yml): its tag
prefix is `mcp-v<version>` (separate from the skeleton library's `v<version>` - the two packages
version independently, and one tag cannot express two versions).

```bash
# 1. Change the version in packages/biosnn-mcp/pyproject.toml (match src/biosnn_mcp/_version.py)
# 2. Write .github/releases/mcp-v<version>.md (release notes are hand-written)
# 3. After committing and pushing, tag it
git tag mcp-v0.2.0 && git push origin mcp-v0.2.0
```

Before the first release, register a pending publisher for `biosnn-mcp` on PyPI (fields are in the
header of that file); the **environment name is `pypi-mcp`**, not the `pypi` used by the skeleton
library. Before registering you can run `gh workflow run release-mcp.yml` to validate and build only;
afterwards `-f publish=true` publishes without re-tagging.

## Documentation translations

The documentation exists in Chinese, English and Japanese. When you change any managed
document, **change all three** — a stale translation is worse than no translation, because
readers take it to be current.

`scripts/check_translations.py` catches four kinds of drift. Run it locally to check:

```bash
uv run python scripts/check_translations.py
```

| What it catches | How it happens |
| :--- | :--- |
| Missing translation | A new document was added without its `.en.md` / `.ja.md` |
| Heading structure drift | A section was dropped, or `###` became `##` |
| Broken code block | A parameter value, identifier or command was changed while translating |
| Internal link pointing at another language | The link kept the old filename |

**Prose is translated; code is not.** Comments and docstrings inside code blocks may be
translated; variable names, commands, paths and string literals such as the registry name
in `get_plugin("audio")` may not — translating that one breaks the example. CI executes the
code blocks in the documentation, so this kind of mistake surfaces immediately.

Terminology follows [`docs/GLOSSARY.md`](docs/GLOSSARY.md); add a new term there before
using it.

Issues and pull requests in any language are welcome.

## Reporting problems

Use the [issue template](https://github.com/lzmd-arch/BioSNN-Plug/issues/new/choose). Bug reports
must include a **minimal reproduction script** and **environment information** — the template
gives you ready-made commands.

## License

By contributing you agree that your contribution is released under [Apache-2.0](LICENSE) (code) or
[CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) (documentation).

> **Every `undefined reference` / `undefined citation` warning in CI logs must be triaged**
> (first-pass BibTeX, or genuinely broken). **Never dismiss the batch as "just the first pass".**
> On 2026-10-24 a `\ref` was eaten by an escape sequence and split across two lines; the warning
> was in the log and was waved through as first-pass noise.

> **Workspace hygiene**: one-off scripts, drafts and commit-message drafts go in `data/`
> (gitignored), **never in the repository root**. Evidence (run logs, fetched sources, old caches)
> also stays under `data/` and is not committed. `sweep_results/` holds only **artifacts that are
> cited** (pre-registrations, summary.json, result documents). A cleanup on 2026-10-24 removed 58
> one-off patch scripts and logs that had accumulated in `data/`.
