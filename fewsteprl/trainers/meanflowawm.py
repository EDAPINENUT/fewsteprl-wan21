"""MeanFlowAWM: advantage-weighted x0 matching on the flow map.

The rewarded sample ``x_0`` is re-noised with a fresh ``eps`` at a fresh
``(t, r)`` and the flow map is asked to denoise it back (latents are 5D
``[B, F, C, H, W]``)::

    x_t      = (1 - t) * x_0 + t * eps
    v_theta  = u_theta(x_t, t, r) + (t - r) * sg(dF/dt)
    x_0_pred = x_t - t * v_theta

The rollout only supplies ``x_0`` and its advantage (``fewstep_rl.reuse_x_t=false``,
``rollout_type=ode``); the inner loop is the base ``_rl_train_phase``. Converting
``u`` to the instantaneous velocity first makes this a MeanFlow objective:
``u_theta`` regresses onto ``v_tgt - (t - r) * dF/dt`` with ``v_tgt = eps - x_0``.

The PPO-style ratio against ``old_model`` (the EMA, which lags the policy) acts as a
trust region at synthetic query points, not as a true importance weight, since
``x_0`` was never drawn from ``pi_old(. | x_t)``.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class MeanFlowAWMTrainer(GeneralRLTrainer):
    """Advantage-weighted x0 matching on a noise-then-denoise transition.

    ``x_anchor_1 = None`` (no self-regression term); ``x_anchor_2`` is
    ``old_model``'s x0 prediction, used by :meth:`decoupled_weights` for the ratio.
    """

    TRAINER_KEY = "meanflowawm_rl"

    #: Extra logged keys. ``_run_rl_epoch`` silently drops keys not in an allowlist,
    #: and ``_reduce_metrics`` all-reduces per key, so the set must be rank-uniform
    #: (a class constant). ``clipfrac`` near 0 means the trust region never binds;
    #: near 1 means it binds everywhere.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "r_mean",           # mean flow-map target time
        "sigma_mean",       # post-floor ratio temperature
        "s_x0_abs_mean",    # |t * S|, the CD correction in x0 units
        "ratio_mean",       # E[rho]; 1.0 at zero drift
        "ratio_max_dev",    # max|rho - 1|; the exp-runaway detector
        "clipfrac",         # fraction of samples outside the trust region
        "approx_kl",        # 0.5*E[(log rho)^2], the PPO KL proxy
    )

    #: Floor on the ratio temperature ``sigma = t``. The gradient is t-independent
    #: (``t**2`` cancels), but the logged loss value carries ``omega**2`` with
    #: ``omega = adv / (2 * t**2)`` and would be heavy-tailed as ``t -> 0``. With
    #: Wan's ``shift=5.0`` the floor almost never binds.
    _SIGMA_MIN: float = 1e-2

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) * dF/dt`` in ``like``'s dtype, or None when S is exactly 0.

        Single source of S's sign and units for both directions of the MeanFlow
        identity.
        """
        if dF_dt is None:
            # Legal only when r == t for every sample (the CD is skipped there).
            if not bool(torch.equal(t_cd, r_cd)):
                raise RuntimeError(
                    "dF_dt is None but r != t for some samples: the shared central "
                    "difference was not computed. Check _pre_compute_v_hook."
                )
            return None

        wdt = like.dtype
        if wdt not in (torch.float32, torch.float64):
            raise TypeError(
                f"_s_term got a {wdt} operand; S must be folded in fp32/fp64. "
                "Folding it in the network's bf16 is what destroys the "
                "cancellation that makes dev S-free."
            )
        # (t - r) in raw timestep units, matching the CD denominator, so the product
        # is dimensionless.
        dt_cd = (t_cd - r_cd).view(
            like.shape[0], *([1] * (like.ndim - 1))
        ).to(wdt)
        return dt_cd * dF_dt.to(wdt)

    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt`` -- the identity solved for v.

        Applied to both policies. ``S`` is detached, so on ``u_theta`` it adds no
        gradient path; it only shifts the regression target to ``v_tgt - S``.
        """
        s = self._s_term(t_cd, r_cd, dF_dt, like=u_old)
        return u_old if s is None else u_old + s

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> Optional[torch.Tensor]:  # noqa: ARG002
        """None: no self-regression term, so the whole loss is the reward term."""
        return None

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_old`` -- pi_old's x0 prediction.

        Same op sequence as ``x_0_pred_v`` in :meth:`_rl_loss` with the same shared
        ``S``, so the two means differ by exactly ``-t * (u_theta - u_old)``.
        """
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"_instantaneous_velocity_from_u must return fp32/fp64, got "
                f"{v_inst.dtype}. Returning the network's bf16 dtype reintroduces "
                "the fold defect."
            )
        if v_inst.requires_grad:
            raise RuntimeError(
                "_instantaneous_velocity_from_u must be detached; x_anchor_2 is a "
                "stop-grad probe center."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    @torch.no_grad()
    def decoupled_weights(
        self,
        x_0_pred_v: torch.Tensor,
        x_anchor_1: Optional[torch.Tensor],  # noqa: ARG002 - always None here
        x_anchor_2: torch.Tensor,   # pi_old's x0 prediction at this query point
        x_0_rwd: torch.Tensor,      # the rewarded sample the rollout produced
        adv: torch.Tensor,          # occupies the anchor form's ``r_hat`` slot
        sigma: torch.Tensor,        # occupies the anchor form's ``beta`` slot
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AWM weights ``(ratio, ratio, omega)`` for the base's reward term::

            L       = gate * ratio * adv / (2 * sigma**2) * mse(x_0_pred_v, x_0_rwd)
            log rho = (l_old - l_pred) / (2 * sigma**2)

        ``sigma = t`` (per-sample, passed in the ``beta`` slot). Since
        ``x_0_pred_v - x_0_rwd == t * (v_tgt - v_theta)``, the ``t**2`` cancels and
        the gradient is t-independent; ``fewstep_rl.beta`` is unused on this trial.

        The mse is a spatial mean, so this is the per-dimension log-ratio (the
        joint ratio would overflow), which is why ``clip_range`` is small (1e-3).
        ``gate`` reproduces the gradient of the PPO clip
        ``max{-A*rho, -A*clip(rho, 1-eps, 1+eps)}`` as a 0/1 mask (ties go to the
        unclipped branch, as in ``torch.maximum``). ``reg_weight`` is returned equal
        to the ratio only so it is logged as ``affine_coeff_mean``; the
        self-regression term is off. ``omega`` is broadcast to the latent rank.
        """
        # sigma must be [B]; a broadcast temperature would silently reshape omega.
        if sigma.ndim != 1 or sigma.shape[0] != x_0_pred_v.shape[0]:
            raise ValueError(
                f"sigma must be [B] with B == {x_0_pred_v.shape[0]}, got shape "
                f"{tuple(sigma.shape)}. Pass the flattened per-sample "
                "temperature, not the broadcast timestep."
            )
        l_pred = self._per_sample_mse(x_0_pred_v, x_0_rwd, spatial)
        l_old = self._per_sample_mse(x_anchor_2, x_0_rwd, spatial)
        two_sigma2 = 2.0 * sigma**2
        # Old loss leads: log rho = log p_theta - log p_old. The reverse gives 1/rho.
        ratio = torch.exp((l_old - l_pred) / two_sigma2)

        # PPO gate: gradient routing of max{-A*rho, -A*clip(rho)} as a 0/1 mask.
        clip_range = float(self._rl_cfg().clip_range)
        unclipped = -adv * ratio
        clipped = -adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        inside = (ratio > 1.0 - clip_range) & (ratio < 1.0 + clip_range)
        gate = (inside | (unclipped >= clipped)).to(ratio.dtype)

        bcast = (-1, *([1] * (x_0_pred_v.ndim - 1)))
        omega = (gate * adv / two_sigma2).view(*bcast)
        return ratio, ratio, omega

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One AWM step on a freshly noised query point (see module docstring).

        The resulting gradient is an advantage- and ratio-weighted MeanFlow
        regression::

            dL/du_theta = ratio * adv * (u_theta - (v_tgt - S)) / N,
            v_tgt = eps - x_0,   S = (t - r) * sg(dF/dt).
        """
        cfg = self._rl_cfg()
        dtype = x_0.dtype
        batch_size = x_0.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)
        wdt = self._rl_work_dtype

        # 1. Forward noising. t_normed stays in the network dtype so x_t matches the
        # other trainers bit-for-bit.
        bcast = (batch_size, *([1] * (x_0.ndim - 1)))
        t_raw = t_raw.to(device=x_0.device)
        t_normed = t_raw.to(dtype) / T
        t_broadcast = t_normed.view(bcast)
        eps = self._draw_eps(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * eps
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = r_raw.to(dtype=torch.float32) if r_raw is not None else None
        # Without r, use r == t (S == 0). _pre_compute_v_hook still gets the raw
        # r_for_model, since None there means "skip the CD".
        r_cd = r_for_model if r_for_model is not None else t_for_model

        # Decided before the anchor block (it decides whether to keep dF/dt past the
        # ``finally``). Config-level, hence rank-uniform; the CD issues FSDP
        # collectives, so a rank that skips them deadlocks.
        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        adv_clip_max = float(cfg.adv_clip_max)
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        # 2. One shared central difference on old_model, then both velocities.
        self._pre_compute_v_hook(
            x_t=x_t, t_for_model=t_for_model, r_for_model=r_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, x_0=x_0,
        )
        try:
            u_theta = self._compute_u(
                self.generator, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=True,
            )
            u_old = self._compute_u(
                self.old_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                r_for_model, with_grad=False,
            ).detach()

            u_theta_f = u_theta.to(wdt)
            u_old_f = u_old.to(wdt)
            x_t_f = x_t.to(wdt)
            x_0_f = x_0.to(wdt)
            t_normed_b_f = t_broadcast.to(wdt)
            spatial = tuple(range(1, x_0.ndim))

            # 3. Denoise to x_0. Convert u to the instantaneous velocity before the
            # jump: x_t - t * u is only an x0 prediction when u is instantaneous.
            v_theta_f = self._instantaneous_velocity_from_u(
                u_theta_f, t_for_model, r_cd, self._shared_dF_dt,
            )
            x_0_pred_v = x_t_f - t_normed_b_f * v_theta_f          # carries grad

            # AnchorInputs rejects network-dtype operands and an unbroadcast [B] t.
            anchor_inputs = AnchorInputs(
                x_t=x_t_f,
                t_normed=t_normed_b_f,
                u_old=u_old_f,
                u_theta=u_theta_f.detach(),
                dF_dt=self._shared_dF_dt,
                t_cd=t_for_model,
                r_cd=r_cd,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)

            # |t * S| in x0 units, reduced to a float here so the CD can be freed.
            s_term = self._s_term(t_for_model, r_cd, self._shared_dF_dt, like=x_t_f)
            s_x0_abs_mean = (
                0.0 if s_term is None
                else float((t_normed_b_f * s_term).abs().mean())
            )

            # ``finally`` clears the attribute; keep a local reference only if the
            # v-mode KL needs it, so the CD can be freed otherwise.
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)

        # Ratio temperature, [B]. Taken from the same t_normed that built x_t so the
        # t**2 cancellation is exact.
        sigma = t_normed.to(wdt).reshape(batch_size).clamp(min=self._SIGMA_MIN)

        # Reward half of the decoupled loss only; the ``r_hat`` / ``beta`` slots
        # carry the clipped advantage and the temperature.
        unscaled, form_info = self.decoupled_rl_loss(
            x_0_pred_v, x_anchor_1, x_anchor_2, x_0_f, adv, sigma, spatial,
        )
        # Not rescaled by adv_clip_max: unlike the anchor trainers there is no
        # [0, 1] squash of the advantage to undo.
        policy_loss = unscaled.mean()
        loss = policy_loss

        with torch.no_grad():
            # Ratio diagnostics, recomputed since decoupled_weights' return
            # signature is fixed by the base.
            clip_range = float(cfg.clip_range)
            l_pred_d = self._per_sample_mse(x_0_pred_v.detach(), x_0_f, spatial)
            l_old_d = self._per_sample_mse(x_anchor_2, x_0_f, spatial)
            log_rho = (l_old_d - l_pred_d) / (2.0 * sigma**2)
            rho_d = torch.exp(log_rho)
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                "t_mean": float(t_raw.mean().item()),
                "r_mean": float(r_cd.mean().item()),
                "sigma_mean": float(sigma.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                "x0_norm_mean": float((x_0_f ** 2).mean().item()),
                # u-space drift between policy and old_model (S cancels).
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                "s_x0_abs_mean": s_x0_abs_mean,
                # Trust-region health: clipfrac near 1 means clip_range is too tight.
                "ratio_mean": float(rho_d.mean()),
                "ratio_max_dev": float((rho_d - 1.0).abs().max()),
                "clipfrac": float((rho_d - 1.0).abs().gt(clip_range).float().mean()),
                "approx_kl": float(0.5 * (log_rho**2).mean()),
                **form_info,
            }
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model,
            shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        # Merged unconditionally: the key set must be rank-uniform for
        # _reduce_metrics, and compute_kl_term pre-seeds inactive keys.
        info.update(kl_info)

        info["loss_total"] = float(loss.detach())
        return loss, info

    def reward_augmentation(
        self,
        x_pred: torch.Tensor,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,  # noqa: ARG002 - the center is x_pred, not x_neg
        omega: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Injects the gradient of ``weight * omega * mse(x_pred, x_pos)`` -> ``[B]``.

        Uses a stop-grad probe center ``x_target = x_pred.detach() + omega *
        (x_pos - x_pred.detach())``, so the gradient is exactly
        ``-weight * omega * (2/N) * (x_pos - x_pred)``: ``omega > 0`` pulls
        ``x_pred`` toward ``x_pos``. ``x_neg`` is unused but kept for the base's
        positional call. The loss value carries ``omega**2`` (always positive), so
        it is not comparable to ``ratio * adv * mse``; only the gradient is meaningful.
        """
        with torch.no_grad():
            norm_grad = (omega * (x_pos.double() - x_pred.double())).to(x_pred.dtype)
        x_target = (x_pred.detach() + norm_grad).to(x_pred.dtype)
        return weight * self._per_sample_mse(x_pred, x_target, spatial)
