"""PyTorch device selection for AMD/Intel/NVIDIA GPUs on Windows via DirectML."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def torch_backend() -> str:
    """Return configured backend: auto | directml | dml | cpu | cuda."""
    return os.getenv("CV_TORCH_BACKEND", "auto").strip().lower()


def dml_device_id() -> int:
    """DirectML adapter index (see torch_directml.device_count())."""
    return int(os.getenv("DML_DEVICE_ID", "0"))


def is_directml_requested() -> bool:
    return torch_backend() in ("directml", "dml", "privateuseone")


def directml_available() -> bool:
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        return False
    try:
        import torch_directml as dml

        return dml.device_count() > 0
    except Exception:
        return False


def get_torch_device() -> torch.device:
    """
    Resolve the PyTorch device for inference.

    On Windows with AMD GPUs, use CV_TORCH_BACKEND=directml (torch-directml).
    Note: ROCm is the Linux stack; DirectML is the supported Windows path for AMD.
    """
    import torch

    backend = torch_backend()

    if backend in ("cpu",):
        return torch.device("cpu")

    if backend in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")

    if backend in ("auto", "directml", "dml", "privateuseone"):
        if directml_available():
            import torch_directml

            device_id = dml_device_id()
            count = torch_directml.device_count()
            if device_id >= count:
                print(
                    f"  WARNING: DML_DEVICE_ID={device_id} out of range "
                    f"(0..{count - 1}), using 0"
                )
                device_id = 0
            dev = torch_directml.device(device_id)
            print(f"  PyTorch DirectML device: {dev} (adapter {device_id}/{count})")
            return dev

        if backend in ("directml", "dml", "privateuseone"):
            print("  WARNING: torch-directml not available, falling back to CPU")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


@contextmanager
def dml_safe_no_grad():
    """
    DirectML does not support torch.inference_mode(); use no_grad instead.

    Ultralytics uses inference_mode by default, which breaks on DirectML.
    """
    import torch

    with torch.no_grad():
        yield


def patch_ultralytics_for_directml() -> None:
    """Force ultralytics to use torch.no_grad instead of torch.inference_mode."""
    import ultralytics.utils.torch_utils as tu

    tu.TORCH_1_10 = False
