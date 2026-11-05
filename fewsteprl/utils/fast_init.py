"""Fast model construction for loading pretrained weights.

Modules are built on the meta device and materialized empty instead of being
randomly initialized, which is faster, uses less memory and consumes no RNG state::

    with fast_init(torch.device("cpu")):
        model = SomeModel.from_pretrained("path/to/model")
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn


# Original __init__ methods, patched inside fast_init.
_ORIGINAL_INITS: dict[type[nn.Module], Any] = {
    nn.Linear: nn.Linear.__init__,
    nn.Embedding: nn.Embedding.__init__,
    nn.LayerNorm: nn.LayerNorm.__init__,
}


def _get_fast_init(cls: type[nn.Module], device: torch.device):
    """Create a patched __init__ that constructs on meta device then materializes."""
    assert cls in _ORIGINAL_INITS

    def _fast_init(self, *args, **kwargs):
        kwargs.pop("device", None)
        _ORIGINAL_INITS[cls](self, *args, **kwargs, device="meta")
        self.to_empty(device=device)

    return _fast_init


@contextlib.contextmanager
def no_init_weights():
    """Turn the common ``nn.init.*`` functions into no-ops inside the context."""
    init_funcs = [
        "kaiming_uniform_",
        "kaiming_normal_",
        "zeros_",
        "ones_",
        "normal_",
        "uniform_",
        "xavier_uniform_",
        "xavier_normal_",
        "constant_",
    ]
    originals = {name: getattr(nn.init, name) for name in init_funcs if hasattr(nn.init, name)}

    for name in originals:
        setattr(nn.init, name, lambda *a, **kw: None)

    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(nn.init, name, orig)


@contextlib.contextmanager
def fast_init(device: torch.device, init_weights: bool = False):
    """Construct ``nn.Linear`` / ``nn.Embedding`` / ``nn.LayerNorm`` as uninitialized
    tensors on ``device``; weights must then be loaded (e.g. ``from_pretrained``).

    ``init_weights=False`` also disables ``nn.init.*`` (see ``no_init_weights``).
    """
    for cls in _ORIGINAL_INITS:
        cls.__init__ = _get_fast_init(cls, device)

    ctx = contextlib.nullcontext() if init_weights else no_init_weights()
    try:
        with ctx:
            yield
    finally:
        for cls in _ORIGINAL_INITS:
            cls.__init__ = _ORIGINAL_INITS[cls]
