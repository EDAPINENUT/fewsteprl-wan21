"""MeanFlowNFT: decoupled-loss RL with a self-induced instantaneous velocity.

``x_anchor_1 = x_t - t * u_old``
``x_anchor_2 = x_t - t * (u_old + (t - r) * sg(dF/dt))``

:class:`GeneralRLTrainer` splits the objective into
``self_regression + reward_augmentation``. This split is gradient-identical to a
single fused MSE, but the loss values differ by a cross term, so compare
gradients rather than loss values.

:class:`NFTTrainer` is the S-term ablation: ``v := u``, so both anchors are
``x_t - t * u_old`` and the central difference is never computed.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class MeanFlowNFTTrainer(GeneralRLTrainer):
    """MeanFlowNFT trainer for RL (x0 form, decoupled loss)."""

    TRAINER_KEY = "meanflownft_rl"

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) * dF/dt`` in ``like``'s dtype, or None when S is exactly 0.

        Single definition of S shared by both directions of the MeanFlow identity
        (``v = u + S`` and ``u = v - S``).
        """
        if dF_dt is None:
            # Only legal when r == t for every sample (central difference skipped).
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
        # (t - r) in raw timestep units, matching the central-difference
        # denominator, so the product is in velocity units.
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
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt`` -- the identity solved for v."""
        s = self._s_term(t_cd, r_cd, dF_dt, like=u_old)
        return u_old if s is None else u_old + s

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = x_t - t * u_old``."""
        return inputs.x_t - inputs.t_normed * inputs.u_old

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_inst``.

        The only place S enters the loss, inside a no-grad anchor.
        """
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        # Explicit raise, not assert: assert vanishes under `python -O`.
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"_instantaneous_velocity_from_u must return fp32/fp64, got "
                f"{v_inst.dtype}. Returning the network's bf16 dtype "
                "reintroduces the fold defect."
            )
        if v_inst.requires_grad:
            raise RuntimeError(
                "_instantaneous_velocity_from_u must be detached; x_anchor_2 is a "
                "stop-grad probe center."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One RL step. Forward noising and the r_hat mapping are shared across the
        decoupled-loss family; only the anchor algebra differs."""
        cfg = self._rl_cfg()
        dtype = x_0.dtype
        batch_size = x_0.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)

        # 1. Forward noising. t_normed is kept in the network dtype on purpose so
        # x_t matches the other trainers in the family bit for bit.
        t_raw = t_raw.to(device=x_0.device)
        t_normed = t_raw.to(dtype) / T
        t_broadcast = t_normed.view(batch_size, *([1] * (x_0.ndim - 1)))
        eps = self._draw_eps(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * eps
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = r_raw.to(dtype=torch.float32) if r_raw is not None else None

        kl_weight = float(cfg.kl_weight)

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        # 2. Average velocities only. S is never folded into these.
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

            wdt = self._rl_work_dtype
            u_theta_f = u_theta.to(wdt)
            u_old_f = u_old.to(wdt)
            x_t_f = x_t.to(wdt)
            x_0_f = x_0.to(wdt)
            t_normed_b_f = t_broadcast.to(wdt)
            spatial = tuple(range(1, x_0.ndim))

            beta = float(cfg.beta)
            adv_clip_max = float(cfg.adv_clip_max)
            if adv_clip_max <= 0:
                raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

            x_0_pred_u = x_t_f - t_normed_b_f * u_theta_f          # carries grad

            # AnchorInputs rejects network-dtype operands and unbroadcast [B]
            # timesteps.
            anchor_inputs = AnchorInputs(
                x_t=x_t_f,
                t_normed=t_normed_b_f,
                u_old=u_old_f,
                u_theta=u_theta_f.detach(),
                dF_dt=self._shared_dF_dt,
                t_cd=t_for_model,
                r_cd=r_for_model,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)
        finally:
            self._post_compute_v_hook()

        # 3. r_hat -- the advantage mapped into [0, 1].
        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        r_hat = ((adv / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)

        unscaled, form_info = self.decoupled_rl_loss(
            x_0_pred_u, x_anchor_1, x_anchor_2, x_0_f, r_hat, beta, spatial,
        )
        policy_loss = (unscaled * adv_clip_max).mean()
        loss = policy_loss

        # ``old_deviate`` is the u-form deviation (no S).
        with torch.no_grad():
            s_x0 = (x_anchor_1 - x_anchor_2)
            dev_abs = (x_0_pred_u.detach() - x_anchor_1).abs().mean()
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                "t_mean": float(t_raw.mean().item()),
                "r_hat_mean": float(r_hat.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                "x0_norm_mean": float((x_0_f ** 2).mean().item()),
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # Emitted unconditionally on every rank: _reduce_metrics
                # all-reduces per key, so a conditional key deadlocks.
                "s_x0_abs_mean": float(s_x0.abs().mean()),
                # |S| / |deviation|.
                "s_rel_median": float(
                    (s_x0.abs().mean() / dev_abs.clamp(min=1e-12))
                ),
                **form_info,
            }

        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model,
            None, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        # compute_kl_term always returns the same key set (KL on or off), keeping
        # the per-key all-reduce rank-uniform.
        info.update(kl_info)

        info["loss_total"] = float(loss.detach())
        return loss, info


class NFTTrainer(MeanFlowNFTTrainer):
    """S-term ablation of :class:`MeanFlowNFTTrainer`: the MeanFlow identity is dropped.

    The flow map's average velocity is used as the instantaneous velocity
    (``v := u`` and ``u := v``), i.e. ``S = (t - r) * dF/dt`` is taken to be 0
    for every sample. Both anchors then collapse to ``x_t - t * u_old`` and the
    loss is plain DiffusionNFT on ``x_0 = x_t - t * u``. The central difference
    is never computed, since nothing reads it.

    Everything else -- sampler, (t, r) draw, KL, optimizer loop -- is inherited
    unchanged, so the only difference from ``meanflow_nft`` is the S term.
    """

    TRAINER_KEY = "nft_rl"

    def _pre_compute_v_hook(self, **_kwargs) -> None:
        # Class-level, hence rank-uniform: no rank enters the CD's collectives.
        return None

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return None

    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``v(x_t, t) := u(x_t, r, t)`` -- no S term."""
        return u_old

