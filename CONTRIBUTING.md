# Contributing to Firefly DataScience

Thanks for helping build the framework. This guide gets you from clone to green PR.

## Development setup

Requires **Python 3.13** and [`uv`](https://docs.astral.sh/uv/). On macOS, the boosting libraries need
OpenMP:

```bash
brew install libomp                                   # macOS only (xgboost / lightgbm)
git clone https://github.com/fireflyframework/fireflyframework-datascience
cd fireflyframework-datascience
uv sync --extra tabular --extra data --extra validation --group dev
```

The single framework dependency, `fireflyframework-agentic`, resolves from its public git repo.
Add extras as you work on them: `--extra dl` (PyTorch), `--extra nlp` (HuggingFace), `--extra tabfm`
(TabPFN), `--extra genai` (the agentic accelerators), or `--extra full`.

## The quality gate

Every change must pass the same gate CI runs:

```bash
uv run ruff check src/ tests/          # lint
uv run ruff format --check src/ tests/ # format
uv run pyright                         # type-check
uv run pytest                          # tests (integration/nightly excluded by default)
```

- **`-m integration`** runs network/heavy tests (OpenML, HuggingFace downloads).
- **`-m nightly`** runs long-running suites (full AMLB, GPU).

## Conventions

- **CalVer** `YY.MM.PATCH`; the version lives in `src/fireflyframework_datascience/_version.py`.
- **Apache-2.0 header** on every `.py`: `# Copyright 2026 Firefly Software Foundation.`
- **Hexagonal**: each module is a light `__init__.py` (ports = `Protocol`s + DTOs), heavy `adapters.py`
  (concrete impls, lazy-importing optional libraries), and a light `auto_configuration.py` that
  registers beans via `@bean`, gated by `@conditional_on_class` / `@conditional_on_property`.
- **Lazy imports** of optional heavy dependencies are deliberate (keeps the core importable without any
  extra). The `PLC0415` rule is therefore relaxed for the DataScience subtree.

## Adding an adapter

1. Define (or reuse) the `Protocol` port in the module's `__init__.py`.
2. Implement the adapter in `adapters.py`, importing the heavy library *inside* the method and raising
   `AdapterUnavailableError("MyAdapter", "<extra>")` when it is missing.
3. Register it in `auto_configuration.py` behind `@conditional_on_class("<library>")`.
4. Add the entry point under `[project.entry-points."firefly_datascience.auto_configuration"]`.
5. Add the optional dependency to `[project.optional-dependencies]`.
6. Write a test (mark it `integration`/`nightly` if it needs network/GPU).

## Docs

Docs are an [mkdocs Material](https://squidfunk.github.io/mkdocs-material/) site under `docs/`.

```bash
uv sync --only-group docs
uv run mkdocs serve              # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict     # must pass (no broken links)
```

Diagrams are generated — edit `assets/tools/gen_diagrams.py`, run it, and commit the SVGs:

```bash
uv run python assets/tools/gen_diagrams.py   # writes docs/img/*.svg
```

## Commits & PRs

Keep the gate green, write a clear commit message, open a PR against `main`, and make sure CI passes.
Thank you! 🐝
