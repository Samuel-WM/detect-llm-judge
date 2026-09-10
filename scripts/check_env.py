"""Print environment diagnostics: torch/CUDA/GPU state and installed package versions.

Run first, before any phase. Paste output into RESULTS.md.
"""
import importlib.metadata as md
import platform
import sys

import torch

REQUIRED = [
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "matplotlib",
    "tqdm",
]


def main() -> None:
    print("=" * 60)
    print("PYTHON / PLATFORM")
    print("=" * 60)
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())

    print()
    print("=" * 60)
    print("TORCH / CUDA")
    print("=" * 60)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        print("gpu name:", torch.cuda.get_device_name(idx))
        print("cuda runtime (torch build):", torch.version.cuda)
        print("total vram (GB):", round(props.total_memory / 1024**3, 3))
        print("free vram (GB):", round(free_b / 1024**3, 3))
        print("cudnn version:", torch.backends.cudnn.version())
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    print()
    print("=" * 60)
    print("PACKAGE VERSIONS")
    print("=" * 60)
    for pkg in REQUIRED:
        try:
            print(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")


if __name__ == "__main__":
    main()
