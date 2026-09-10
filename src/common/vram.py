"""VRAM bookkeeping helpers. Exactly one model may be resident on the GPU at a time (Section 1,
rule 2). Callers must `del` their own references before calling `reclaim()`; this module cannot
drop a caller's local variable for them.
"""
import gc

import torch


def free_vram_gb(device: int = 0) -> float:
    free_b, _total_b = torch.cuda.mem_get_info(device)
    return free_b / 1024**3


def reclaim(device: int = 0) -> float:
    """Run after `del model` (and any other tensors) to release CUDA memory. Returns free VRAM
    in GB after reclaiming, so callers can log or assert against it."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return free_vram_gb(device)


def assert_baseline(baseline_gb: float, tol_gb: float = 0.3, device: int = 0) -> None:
    """Fail loudly if free VRAM has not returned to (near) the recorded baseline after unload.
    A shrinking baseline across judges means something is leaking on the GPU.
    """
    current = free_vram_gb(device)
    assert current >= baseline_gb - tol_gb, (
        f"VRAM did not return to baseline: baseline={baseline_gb:.3f} GB, "
        f"current={current:.3f} GB, tol={tol_gb} GB. Something is still resident on the GPU."
    )
