# Project scope and provenance

## Why this repository exists

This fork contains small reproducibility and validation additions around CAPI.
It keeps a clear link to the original work and does not claim the original
method or results as its own.

## Provenance

The repository was initialized from the official Meta FAIR implementation at
[`facebookresearch/capi`](https://github.com/facebookresearch/capi). The method,
core training code, architecture, checkpoints, paper results, and figures
originate upstream and retain their original attribution and license.

The Git history is preserved so changes can be audited. `git diff` against the
recorded upstream commit is the authoritative contribution record.

## Changes in this fork

- portable handling of the optional Intel scikit-learn extension;
- environment and pretrained-inference validation commands;
- structured JSON experiment provenance;
- fast repository-contract tests and GitHub Actions checks;
- reproducibility, limitations, and contribution documentation.

These are engineering changes, not new machine-learning research. Upstream
benchmark values are labeled as reported rather than reproduced.

## How to describe it

Describe the environment checks, inference validation, test coverage, and
documentation you added. Do not imply authorship of CAPI or its paper unless
you are a named author.

Before publishing, confirm that any CI badge or remote link points to the
published repository, and keep benchmark claims tied to saved artifacts.
