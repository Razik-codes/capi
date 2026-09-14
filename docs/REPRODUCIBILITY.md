# Reproducibility protocol

This repository separates checks that can run on a normal laptop from
experiments that require a larger accelerator, datasets, and network access.

## Current validation status

Full CAPI pretraining and paper-scale evaluation have not yet been reproduced
locally in this fork. The currently available laptop GPU has 4 GB VRAM,
which is below the practical requirement for ViT-L/14 pretrained inference,
ADE20K/ImageNet evaluation, and far below the multi-GPU setup used for full
pretraining.

The benchmark values in the README are therefore labeled as values reported by
the original authors. New results should only be added after the exact command,
hardware, software environment, and output artifact are recorded.

## Validation tiers

### Tier 0: repository contract

Runs without CUDA, datasets, pretrained weights, or project dependencies:

```bash
python -m unittest discover -s tests
python -m py_compile scripts/check_environment.py scripts/validate_pretrained.py
```

This tier checks that the public-facing repository structure is coherent and
that validation scripts remain syntactically valid.

### Tier 1: environment report

Runs after installing the project environment:

```bash
uv sync
uv run python scripts/check_environment.py --json artifacts/environment.json
```

On a GPU machine, make the hardware requirement explicit:

```bash
uv run python scripts/check_environment.py \
  --require-cuda \
  --min-vram-gb 16 \
  --json artifacts/environment_cuda.json
```

### Tier 2: pretrained inference

Requires network access for checkpoint download unless the checkpoint is already
cached or passed through `--weights`.

```bash
uv run python scripts/validate_pretrained.py \
  --model capi_vitl14_in1k \
  --device cuda \
  --output artifacts/capi_vitl14_in1k_cuda.json
```

For constrained machines, CPU validation may be possible but can be slow:

```bash
uv run python scripts/validate_pretrained.py \
  --model capi_vitl14_in1k \
  --device cpu \
  --output artifacts/capi_vitl14_in1k_cpu.json
```

The command exits non-zero if loading, inference, shape capture, or finite-value
validation fails. The JSON report should be kept with any claim based on the
run.

### Tier 3: evaluation

Requires accepted dataset terms, local dataset storage, and enough accelerator
memory for feature extraction.

```bash
uv run python benchmark.py \
  model_loader_kwargs.config_path=default_pretrain_config.yaml \
  pretrained_weights=/path/to/checkpoint.pth
```

Record the dataset paths, checkpoint hash or URL, GPU model, driver/CUDA
versions, and resulting logs under `artifacts/`.

### Tier 4: pretraining

The default configuration is a large distributed experiment:

```bash
uv run python train_distributed.py default_pretrain_config.yaml \
  train.output_dir=/path/to/output
```

The default launcher uses four nodes and the optimizer configuration assumes a
large effective batch size. A reduced configuration would be useful for code
debugging, but it should not be described as reproducing the paper-scale result.

## Artifact naming

Use stable, descriptive paths:

```text
artifacts/
  environment_cuda.json
  capi_vitl14_in1k_cuda.json
  benchmark_ade20k_<date>.log
```

Do not commit datasets, checkpoints, access tokens, generated caches, or private
machine paths.
