#!/usr/bin/env python3
"""Validate one pretrained CAPI checkpoint and write a provenance report."""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_WEIGHTS = {
    "capi_vitl14_p205": "https://dl.fbaipublicfiles.com/capi/capi_vitl14_p205.pth",
    "capi_vitl14_lvd": "https://dl.fbaipublicfiles.com/capi/capi_vitl14_lvd.pth",
    "capi_vitl14_in22k": "https://dl.fbaipublicfiles.com/capi/capi_vitl14_i22k.pth",
    "capi_vitl14_in1k": "https://dl.fbaipublicfiles.com/capi/capi_vitl14_in1k.pth",
}


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def output_summary(output: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        return tensor_summary(output)
    if isinstance(output, (tuple, list)):
        return [output_summary(item) for item in output]
    return {"type": type(output).__name__}


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_WEIGHTS), default="capi_vitl14_in1k")
    parser.add_argument("--weights", help="Optional local checkpoint or URL. Overrides --model weights.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete CUDA device such as cuda:0.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, required=True, help="Path for the JSON validation report.")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "model": args.model,
        "weights": args.weights or MODEL_WEIGHTS[args.model],
        "requested_device": args.device,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "ok": False,
    }

    start = time.perf_counter()
    try:
        import torch

        from model import __model_loader__

        device = resolve_device(args.device)
        report["resolved_device"] = device
        report["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))

        model = __model_loader__(config_path=None, pretrained_weights=report["weights"], device=device)
        model.eval()

        images = torch.zeros(args.batch_size, 3, args.image_size, args.image_size, device=device)
        with torch.inference_mode():
            output = model(images)

        report["outputs"] = output_summary(output)
        if isinstance(report["outputs"], list):
            report["all_outputs_finite"] = all(
                item.get("finite", True) for item in report["outputs"] if isinstance(item, dict)
            )
        else:
            report["all_outputs_finite"] = report["outputs"].get("finite", True)

        if device.startswith("cuda"):
            report["peak_cuda_memory_gb"] = round(torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3, 3)

        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        report["ok"] = bool(report["all_outputs_finite"])
    except Exception as exc:  # noqa: BLE001 - this script must record failures as provenance.
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    write_report(args.output, report)
    print(f"Wrote validation report to {args.output}")
    print("Validation status: " + ("ok" if report["ok"] else "failed"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
