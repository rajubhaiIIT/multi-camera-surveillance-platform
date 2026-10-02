#!/usr/bin/env python3
"""Is PyTorch using your NVIDIA GPU?   python scripts/check_gpu.py"""
import sys

try:
    import torch
except ImportError:
    sys.exit("torch is not installed. See README, section 'Phase 1 setup'.")

print("torch:", torch.__version__, "| built for CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    print("[FAIL] CUDA not available. You probably have the CPU-only torch build.")
    print("       Fix: python -m pip install --force-reinstall torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu126")
    print("       (check pytorch.org/get-started/locally for the current command; update your NVIDIA driver if it still fails)")
    sys.exit(1)
print("[ OK ] GPU:", torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
x = torch.randn(2048, 2048, device="cuda")
print("[ OK ] matmul on GPU:", float((x @ x).sum().abs() > 0))
try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except ImportError:
    print("ultralytics not installed yet: python -m pip install -r requirements-ai.txt")
