#!/usr/bin/env python3
"""Report whether the local machine can run CAPI experiments.

This command is intentionally diagnostic. Missing Python packages fail the
check; missing or small CUDA devices are reported clearly so CPU-only repository
checks can still be used on constrained laptops.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "omegaconf": "omegaconf",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "xformers": "xformers",
    "jaxtyping": "jaxtyping",
    "einops": "einops",
    "matplotlib": "matplotlib",
    "pillow": "Pillow",
    "rich": "rich",
    "nvidia-cuda-runtime-cu12": "nvidia-cuda-runtime-cu12",
    "cuml-cu12": "cuml-cu12",
    "tabulate": "tabulate",
    "torchmetrics": "torchmetrics",
    "huggingface-hub": "huggingface-hub",
    "timm": "timm",
    "datasets": "datasets",
    "filelock": "filelock",
}


def distribution_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def collect_package_versions() -> dict[str, str | None]:
    return {name: distribution_version(distribution) for name, distribution in PACKAGE_DISTRIBUTIONS.items()}


def collect_cuda_report() -> dict[str, Any]:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        return {
            "torch_importable": False,
            "available": False,
            "devices": [],
            "message": "torch is not importable",
        }

    import torch

    report: dict[str, Any] = {
        "torch_importable": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if not torch.cuda.is_available():
        report["message"] = "CUDA is not available to PyTorch"
        return report

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        report["devices"].append(
            {
                "index": index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_gb": round(total_bytes / 1024**3, 2),
                "free_vram_gb": round(free_bytes / 1024**3, 2),
            }
        )
    return report


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional path for the full JSON report.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available.")
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=0.0,
        help="Fail if the largest visible CUDA device has less than this much VRAM.",
    )
    args = parser.parse_args()

    package_versions = collect_package_versions()
    missing_packages = [name for name, version in package_versions.items() if version is None]
    cuda = collect_cuda_report()

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
        "missing_packages": missing_packages,
        "cuda": cuda,
        "checks": {
            "require_cuda": args.require_cuda,
            "min_vram_gb": args.min_vram_gb,
        },
        "ok": True,
        "warnings": [],
    }

    if missing_packages:
        report["ok"] = False
        report["warnings"].append("Some project dependencies are not installed.")

    if args.require_cuda and not cuda["available"]:
        report["ok"] = False
        report["warnings"].append("CUDA was required but is not available.")

    largest_vram = max((device["total_vram_gb"] for device in cuda["devices"]), default=0.0)
    if args.min_vram_gb and largest_vram < args.min_vram_gb:
        report["ok"] = False
        report["warnings"].append(
            f"Largest visible CUDA device has {largest_vram:.2f} GB VRAM; requested {args.min_vram_gb:.2f} GB."
        )

    print("CAPI environment report")
    print(f"Python: {report['python']}")
    print(f"Platform: {report['platform']}")
    print(f"Missing packages: {', '.join(missing_packages) if missing_packages else 'none'}")
    if cuda["available"]:
        for device in cuda["devices"]:
            print(
                "CUDA device {index}: {name}, {total_vram_gb:.2f} GB total, {free_vram_gb:.2f} GB free".format(
                    **device
                )
            )
    else:
        print(f"CUDA: unavailable ({cuda.get('message', 'no CUDA device reported')})")

    for warning in report["warnings"]:
        print(f"WARNING: {warning}")

    if args.json:
        write_json(args.json, report)
        print(f"Wrote JSON report to {args.json}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
