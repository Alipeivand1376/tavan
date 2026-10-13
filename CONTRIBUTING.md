# Contributing to tavan

Contributions are welcome. A few guidelines:

## Development setup

```bash
git clone https://github.com/Alipeivand1376/tavan
cd tavan
pip install -e ".[dev]"
```

## Before opening a pull request

- `pytest` — all tests must pass (new behaviour needs a test).
- `ruff check src` — lint must be clean.
- Keep the **NumPy-only core**: new models go on `tavan.nn`, not on an external
  deep-learning framework. Optional features may use optional dependencies
  (declared as extras) and must import them lazily.
- Public functions and classes use NumPy-style docstrings.
- New models register with `@tavan.registry.register(kind, name)` and declare
  `_state_attrs` so they can be saved and loaded.

## Scope

`tavan` is deliberately small and transparent. Prefer clear, well-documented
reference implementations over micro-optimised or feature-maximal code.
