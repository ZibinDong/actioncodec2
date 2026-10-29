"""Data parallelism, or none at all, behind one shape."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

__all__ = ["Distributed", "setup", "cleanup"]

#: Ranks wait at a barrier while the primary rank writes a checkpoint, which can
#: take longer than NCCL's ten-minute default on a network filesystem.
COLLECTIVE_TIMEOUT = timedelta(minutes=60)


@dataclass(frozen=True)
class Distributed:
    """Who this process is in the job."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        """Only this process prints, logs to the tracker and writes files."""
        return self.rank == 0

    @property
    def device(self) -> str:
        on_cpu = not torch.cuda.is_available() or (
            dist.is_initialized() and dist.get_backend() == "gloo"
        )
        return "cpu" if on_cpu else f"cuda:{self.local_rank}"

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def sum_(self, *values: float) -> tuple[float, ...]:
        """Sum scalars across ranks; identity when there is only one."""
        if not self.enabled:
            return tuple(float(value) for value in values)
        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tuple(tensor.tolist())


def setup() -> Distributed:
    """Join the job described by the environment, if there is one.

    ``torchrun`` sets ``RANK``/``LOCAL_RANK``/``WORLD_SIZE``; without them the
    process runs as a single rank.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return Distributed()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                timeout=COLLECTIVE_TIMEOUT,
                device_id=torch.device("cuda", local_rank),
            )
        else:  # CPU-only ranks (used to test the data-parallel path)
            dist.init_process_group(backend="gloo", timeout=COLLECTIVE_TIMEOUT)
    return Distributed(rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
