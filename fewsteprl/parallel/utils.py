"""Distributed training utilities: process group setup, rank helpers, FSDP / DDP
wrapping and device meshes for hybrid sharding."""

from __future__ import annotations

import os
import functools
from datetime import timedelta
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.device_mesh import init_device_mesh


# --- process group and rank helpers ---

def setup_distributed() -> None:
    """Initialize the NCCL process group from the torchrun environment variables."""
    if dist.is_initialized():
        return
    # NCCL's default 600s timeout is too short for eval phases such as VBench,
    # where ranks can wait at a barrier for a long time; configurable via env.
    timeout_hours = float(os.environ.get("FEWSTEPRL_DIST_TIMEOUT_HOURS", "6"))
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=timeout_hours))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """Destroy the default process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


# --- FSDP sharding strategy ---

_STRATEGY_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid": ShardingStrategy.HYBRID_SHARD,
    "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "no_shard": ShardingStrategy.NO_SHARD,
}

_HYBRID_SHARDING_NAMES = {"hybrid", "hybrid_zero2"}
_DEVICE_MESH_CACHE: dict[tuple[bool, int, int], Any] = {}


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Map "full_shard" / "shard_grad_op" / "hybrid" / "hybrid_zero2" / "no_shard"
    to the FSDP ``ShardingStrategy``."""
    if name not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown sharding strategy: {name}. "
            f"Choose from: {list(_STRATEGY_MAP.keys())}"
        )
    return _STRATEGY_MAP[name]


# --- mixed precision ---

_DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def get_mixed_precision_policy(fsdp_precision: str) -> MixedPrecision | None:
    """FSDP ``MixedPrecision`` for "bf16" / "fp16"; None for "fp32" / "no".

    PyTorch-style names ("bfloat16", ...) are also accepted.
    """
    fsdp_precision = _DTYPE_ALIASES.get(fsdp_precision, fsdp_precision)
    if fsdp_precision == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif fsdp_precision == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif fsdp_precision in {"no", "fp32"}:
        return None
    else:
        raise ValueError(
            f"Unknown precision: {fsdp_precision}. Choose from: bf16, fp16, fp32, no"
        )


# --- FSDP wrapping ---

def fsdp_wrap_model(
    model: torch.nn.Module,
    sharding_strategy: str = "full_shard",
    fsdp_precision: str = "bf16",
    auto_wrap_policy: Callable | None = None,
    device_id: int | None = None,
    use_orig_params: bool = True,
) -> FSDP:
    """Wrap a model with FSDP.

    Args:
        fsdp_precision: param/reduce/buffer dtype policy, deliberately independent
            of the autocast precision.
        auto_wrap_policy: defaults to wrapping modules with >1M parameters.
        device_id: CUDA device; defaults to LOCAL_RANK.
        use_orig_params: keep original parameters instead of FlatParameters;
            required for LoRA (mixed frozen / trainable parameters).
    """
    if device_id is None:
        device_id = get_local_rank()

    strategy = get_sharding_strategy(sharding_strategy)
    mp_policy = get_mixed_precision_policy(fsdp_precision)
    device_mesh = (
        get_device_mesh(use_hybrid=True)
        if sharding_strategy in _HYBRID_SHARDING_NAMES
        else None
    )

    if auto_wrap_policy is None:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000_000
        )

    fsdp_kwargs = dict(
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.device("cuda", device_id),
        limit_all_gathers=True,
        use_orig_params=use_orig_params,
    )
    if device_mesh is not None:
        fsdp_kwargs["device_mesh"] = device_mesh

    wrapped = FSDP(
        model,
        **fsdp_kwargs,
    )
    return wrapped


def get_transformer_wrap_policy(transformer_block_cls: type | set[type] | tuple[type, ...]) -> Callable:
    """FSDP auto-wrap policy that wraps each instance of ``transformer_block_cls``."""
    if isinstance(transformer_block_cls, type):
        transformer_layer_cls = {transformer_block_cls}
    else:
        transformer_layer_cls = set(transformer_block_cls)

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )


# --- DDP wrapping ---

def ddp_wrap_model(
    model: torch.nn.Module,
    device_id: int | None = None,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> torch.nn.parallel.DistributedDataParallel:
    """Wrap a model (already on its device) with DDP; ``device_id`` defaults to
    LOCAL_RANK."""
    if device_id is None:
        device_id = get_local_rank()
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


# --- device mesh (hybrid sharding across nodes) ---

def _set_mesh_group_timeouts(mesh: Any) -> None:
    """Apply the long collective timeout to a device mesh's per-dim sub-groups.

    ``init_device_mesh`` creates them without a timeout, so they would keep NCCL's
    600s default even though the default group uses a longer one (see
    :func:`setup_distributed`). Best-effort.
    """
    try:
        from torch.distributed.distributed_c10d import _set_pg_timeout
    except Exception:  # noqa: BLE001
        return
    timeout_hours = float(os.environ.get("FEWSTEPRL_DIST_TIMEOUT_HOURS", "6"))
    timeout = timedelta(hours=timeout_hours)
    for name in (getattr(mesh, "mesh_dim_names", None) or ()):
        try:
            _set_pg_timeout(timeout, mesh.get_group(name))
        except Exception:  # noqa: BLE001
            pass


def get_device_mesh(use_hybrid: bool = False) -> torch.distributed.device_mesh.DeviceMesh:
    """Cached device mesh: 2D (replicate across nodes x shard within a node) when
    ``use_hybrid``, else 1D (shard only)."""
    world_size = get_world_size()
    local_size = get_local_world_size()
    cache_key = (use_hybrid, world_size, local_size)
    if cache_key in _DEVICE_MESH_CACHE:
        return _DEVICE_MESH_CACHE[cache_key]

    if use_hybrid:
        if local_size <= 0:
            raise ValueError(f"LOCAL_WORLD_SIZE must be positive for HSDP, got {local_size}")
        if world_size % local_size != 0:
            raise ValueError(
                f"WORLD_SIZE ({world_size}) must be divisible by LOCAL_WORLD_SIZE "
                f"({local_size}) for HSDP hybrid sharding"
            )
        n_nodes = world_size // local_size
        mesh = init_device_mesh(
            "cuda",
            (n_nodes, local_size),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    _set_mesh_group_timeouts(mesh)
    _DEVICE_MESH_CACHE[cache_key] = mesh
    return mesh
