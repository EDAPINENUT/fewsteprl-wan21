"""Flow-map scheduler, equivalent to AnyFlow's ``FlowMapDiscreteScheduler``.

``step`` takes an arbitrary ``(timestep, r_timestep)`` pair and applies the
flow-map jump::

    prev_sample = sample - (timestep - r_timestep) / num_train_timesteps * model_output

where ``model_output`` is the average velocity over ``[r, t]``; ``r -> t`` gives an
Euler step and ``r = 0`` maps ``z_t`` straight to ``z_0``. Timesteps are in raw
model units ``[0, T]`` with ``t = T`` pure noise, and forward noising is
``z_t = (1 - t/T) * x + (t/T) * noise``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class FlowMapScheduler(SchedulerMixin, ConfigMixin):
    """Flow-map scheduler with arbitrary ``(t, r)`` steps.

    Args:
        num_train_timesteps: T; sigmas elsewhere in fewsteprl are ``t / T``.
        shift: static timestep shift (1.0 = none; 5.0 for AnyFlow-Wan).
        weight_type: training loss weight over t: 'gaussian' (peaked at T/2),
            'beta08' (``t * (1-t)^0.5``) or 'uniform'.
    """

    order = 1
    _compatibles: list[str] = []

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        weight_type: str = "gaussian",
    ):
        self.set_timesteps(num_train_timesteps, device="cpu")
        self.set_train_weight(weight_type)

    # --- static shift: sigma' = alpha * sigma / (1 + (alpha - 1) * sigma) ---

    def apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Apply the configured static shift to sigmas in ``[0, 1]``."""
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (
            1.0 + (self.config.shift - 1.0) * sigmas
        )

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        """Set ``timesteps = apply_shift(linspace(1, 0, N+1)) * T``.

        The trailing 0 lets the rollout take the final ``(timesteps[N-1], 0)`` jump.
        """
        if num_inference_steps is None or num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}"
            )
        timesteps = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device,
        )
        timesteps = self.apply_shift(timesteps)
        self.timesteps = timesteps * self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # --- forward noising ---

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Linear forward noising: ``z_t = (1 - t/T) * x + (t/T) * noise``."""
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device, dtype=sample.dtype)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        timestep = timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        return timestep * noise + (1.0 - timestep) * sample

    # --- flow-map step ---

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        r_timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """One flow-map jump from ``z_t`` to ``z_r``; ``model_output`` is the average
        velocity over ``[r, t]``."""
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device, dtype=model_output.dtype)
        if not torch.is_tensor(r_timestep):
            r_timestep = torch.tensor(r_timestep, device=sample.device, dtype=model_output.dtype)
        timestep = timestep.to(device=model_output.device) / self.config.num_train_timesteps
        r_timestep = r_timestep.to(device=model_output.device) / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        r_timestep = r_timestep.view(*r_timestep.shape, *([1] * (model_output.ndim - r_timestep.ndim)))
        prev_sample = sample - (timestep - r_timestep) * model_output
        return prev_sample.to(model_output.dtype)

    # --- loss weighting over t ---

    def set_train_weight(self, weight_type: str) -> None:
        """Precompute loss weights on the T+1 training timesteps.

        Each curve is normalized to ``sum(weights) == T`` so the weight type does not
        rescale the loss.
        """
        T = self.config.num_train_timesteps
        # Separate from ``self.timesteps``, which set_timesteps() overwrites with
        # the short inference grid.
        raw = torch.linspace(1.0, 0.0, T + 1, dtype=torch.float64)
        self.train_timesteps = self.apply_shift(raw) * T
        x = self.train_timesteps
        if weight_type == "gaussian":
            y = torch.exp(-2.0 * ((x - T / 2.0) / T) ** 2)
            y_shifted = y - y.min()
            weights = y_shifted * (T / y_shifted.sum())
        elif weight_type == "beta08":
            t = x / T
            y = (t ** 1.0) * ((1.0 - t) ** 0.5)
            weights = y * (T / y.sum())
        elif weight_type == "uniform":
            weights = torch.ones_like(x) * (T / x.numel())
        else:
            raise ValueError(
                f"Invalid weight_type {weight_type!r}; must be one of "
                f"'gaussian', 'beta08', 'uniform'."
            )
        self.linear_timesteps_weights = weights.to(dtype=torch.float32)

    @torch.no_grad()
    def get_train_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Loss weight for raw timesteps (nearest training timestep, no interpolation)."""
        ref = self.train_timesteps.to(timesteps.device)
        # (num_train_timesteps + 1, num_query)
        diff = (ref.unsqueeze(1) - timesteps.flatten().unsqueeze(0).to(ref.dtype)).abs()
        idx = torch.argmin(diff, dim=0).reshape(timesteps.shape)
        return self.linear_timesteps_weights.to(timesteps.device)[idx]
