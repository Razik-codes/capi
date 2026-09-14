# CAPI: Reproducibility Notes and Validation Tools

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

This fork of **CAPI (Cluster and Predict Latent Patches)** adds a few tools I
use to check a research environment, validate pretrained-model inference, and
record what was run. CAPI is a masked-image-modeling method from Meta FAIR.

> **Attribution.** The CAPI method, model implementation, pretrained weights,
> figures, and published results are the work of Darcet et al. This repository
> is based on [`facebookresearch/capi`](https://github.com/facebookresearch/capi)
> and does not claim authorship of the original research. See
> [Project scope](docs/PROJECT_SCOPE.md) for a precise separation of
> upstream work and changes in this fork.

## Status

I have not reproduced full CAPI training or the paper-scale evaluations. My
laptop GPU has **4 GB VRAM**, so it cannot reliably run ViT-L/14 evaluation and
is far below the multi-GPU setup required for pretraining. The benchmark values
below are reported by the original CAPI authors, not measured in this fork.

The additions here are small engineering changes: an environment report, a
pretrained-inference check that writes JSON, CPU-safe tests, CI, and notes on
what can and cannot be validated on this machine.

## Research context

CAPI trains an image encoder by clustering latent patch representations and
predicting their cluster assignments from masked views. The original study asks
whether discrete targets learned online can improve masked image modeling
without a separately trained tokenizer.

- Paper: [Cluster and Predict Latent Patches for Improved Masked Image Modeling](https://arxiv.org/abs/2502.08769)
- Official code: [facebookresearch/capi](https://github.com/facebookresearch/capi)
- What changed in this fork: [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)
- Reproduction protocol: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

![CAPI architecture from the original project](imgs/poule_fig.png)

## Changes in this fork

- a committed `uv.lock` for the reference environment;
- checks that can run without CUDA, datasets, or checkpoint downloads;
- a JSON report for a pretrained-model inference check;
- GitHub Actions for the lightweight checks;
- notes on the original code, local limitations, and next validation steps.

I will add local benchmark numbers only alongside the command, environment, and
saved output from the run.

## Quick start

Run the checks that do not need CUDA, datasets, pretrained weights, or project
dependencies:

```bash
python -m unittest discover -s tests
python -m py_compile scripts/check_environment.py scripts/validate_pretrained.py
```

Create the reference project environment:

```bash
uv sync
uv run python scripts/check_environment.py --json artifacts/environment.json
```

The reference environment targets Python 3.11.9, PyTorch 2.5.1, CUDA 12.1, and
Linux. A CUDA-capable machine with substantially more than 4 GB VRAM is
recommended; the pretrained ViT-L/14 model has 302M parameters.

Validate a pretrained model and save a JSON report:

```bash
uv run python scripts/validate_pretrained.py \
  --model capi_vitl14_in1k \
  --device cuda \
  --output artifacts/capi_vitl14_in1k_cuda.json
```

The command exits non-zero if CUDA, weight download, loading, inference, shape
capture, or finite-value validation fails. It never converts an offline or
out-of-memory failure into a passing result.

## Pretrained checkpoints

These numbers are reported by the original CAPI authors; they are not results
from this fork.

| Architecture | Parameters | Pretraining data | ADE20K k-NN | ImageNet-1k attentive | Weights |
|---|---:|---|---:|---:|---|
| ViT-L/14 | 302M | Places205 | 35.2 | 79.2 | [checkpoint](https://dl.fbaipublicfiles.com/capi/capi_vitl14_p205.pth) |
| ViT-L/14 | 302M | LVD-142M | 32.1 | 83.8 | [checkpoint](https://dl.fbaipublicfiles.com/capi/capi_vitl14_lvd.pth) |
| ViT-L/14 | 302M | ImageNet-22k | 29.7 | 83.6 | [checkpoint](https://dl.fbaipublicfiles.com/capi/capi_vitl14_i22k.pth) |
| ViT-L/14 | 302M | ImageNet-1k | 29.2 | 82.9 | [checkpoint](https://dl.fbaipublicfiles.com/capi/capi_vitl14_in1k.pth) |

Minimal feature extraction:

```python
import torch

model = torch.hub.load("facebookresearch/capi:main", "capi_vitl14_in1k")
model.eval()

images = torch.zeros(1, 3, 224, 224)
with torch.inference_mode():
    global_repr, registers, feature_map = model(images)
```

## Training and evaluation

The default configuration reproduces the original ViT-L/14 setup. It is a
large-scale distributed experiment, not a laptop-sized tutorial.

```bash
# Direct training
uv run python train_capi.py default_pretrain_config.yaml \
  train.output_dir=/path/to/output

# Slurm launcher (four nodes by default)
uv run python train_distributed.py default_pretrain_config.yaml \
  train.output_dir=/path/to/output

# Evaluation suite
uv run python benchmark.py \
  model_loader_kwargs.config_path=default_pretrain_config.yaml \
  pretrained_weights=/path/to/checkpoint.pth
```

Dataset identifiers use URL-like syntax. Examples include
`custom://ADE20K?split='training'`,
`torchvision://ImageFolder?root='/path/to/imagenet/train'`, and Hugging Face
datasets such as `hf://timm/imagenet-22k-wds?...`. Some datasets require prior
acceptance of their terms and an access token.

## Repository map

| Path | Purpose |
|---|---|
| `model.py` | Vision Transformer, decoder, clustering head, and model loader |
| `train_capi.py` | CAPI training loop |
| `data.py` | Datasets, augmentation, and masking |
| `eval_classification.py` | Linear and attentive classification evaluation |
| `eval_segmentation.py` | k-NN and logistic-regression segmentation evaluation |
| `eval_visualizations.py` | PCA and dictionary-learning feature visualizations |
| `fsdp.py` | SimpleFSDP implementation used for distributed training |
| `scripts/check_environment.py` | Environment and hardware provenance report |
| `scripts/validate_pretrained.py` | Pretrained checkpoint inference validation |
| `tests/` | Fast repository-contract checks used in CI |
| `.github/workflows/quality.yml` | CPU-only GitHub Actions checks |

## Limitations

- Full pretraining requires a multi-GPU or multi-node setup.
- Local full-scale experiments have not yet been run because the available GPU
  has 4 GB VRAM.
- The reference dependencies are CUDA-specific and primarily tested on Linux.
- Dataset licenses and authentication are not automated.
- CI checks repository contracts and added tooling; it does not claim to
  reproduce large-scale paper results on a hosted runner.

## Contributing and citation

See [CONTRIBUTING.md](CONTRIBUTING.md). When using the method, implementation,
or weights, cite the original paper:

```bibtex
@article{darcet2025capi,
  title   = {Cluster and Predict Latent Patches for Improved Masked Image Modeling},
  author  = {Darcet, Timoth{\'e}e and Baldassarre, Federico and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal = {arXiv preprint arXiv:2502.08769},
  year    = {2025}
}
```

The code and model weights remain under the [Apache License 2.0](LICENSE).
