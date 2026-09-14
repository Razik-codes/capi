# Contributing

Contributions that improve reproducibility, correctness, documentation, or
evaluation coverage are welcome.

## Development setup

Fast checks for public repository structure and validation-script syntax:

```bash
python -m unittest discover -s tests
python -m py_compile scripts/check_environment.py scripts/validate_pretrained.py
```

Project-environment checks after installing the CUDA-oriented dependencies:

```bash
uv sync
uv run python scripts/check_environment.py --json artifacts/environment.json
```

The dependencies are CUDA-oriented. Fast repository tests deliberately avoid
model downloads, datasets, project imports, and GPU requirements; expensive
checks belong in a documented experiment command and should save output under
`artifacts/`.

## Pull requests

1. Create a focused branch from `main`.
2. Explain the research or engineering motivation.
3. Add or update tests for behavior that can be checked automatically.
4. Record the exact command and hardware for GPU-dependent validation.
5. Run the relevant checks and update documentation when interfaces change.
6. Keep upstream work and original contributions clearly attributed.

Do not commit datasets, checkpoints, access tokens, generated caches, or
personally identifying experiment paths.

Open issues with a minimal example, OS, Python and PyTorch versions,
accelerator model, command, and complete error. By contributing, you agree that
your contribution is licensed under Apache License 2.0.
