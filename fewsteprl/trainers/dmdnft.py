"""DMD-NFT: decoupled-loss RL with an auxiliary-network velocity.

``x_anchor_2 = x_t - t * v_fake(x_t, t)``

Variant B (MeanFlowNFT) reconstructs the instantaneous velocity from ``old_model``
(``u_old + (t - r) * sg(dF/dt)``, a probe of one trajectory). This variant reads it
from :attr:`aux_v_model`, a v-predictor trained online with a flow-matching loss on
the generator's rollouts, i.e. an amortized estimate of the marginal
``E[eps - x_0 | x_t, t]``. The two agree only in expectation.

Despite the name this is not full DMD: ``v_fake`` only supplies the probe center of
``decoupled_rl_loss``, so there is no real-score network and no backprop through
the rollout. The aux network is trained by :meth:`GeneralRLTrainer._run_aux_epoch`
before each policy epoch; its update ratio is set by ``max_aux_sample`` and
``renoise_per_epoch / aux_v_gradient_accum``.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class DMDNFTTrainer(GeneralRLTrainer):
    """DMD-NFT: MeanFlowNFT with the anchor velocity read off the aux v-predictor."""

    TRAINER_KEY = "dmdnft_rl"

    def setup_models(self) -> None:
        """Require ``aux_v_enabled`` (``aux_v_model`` supplies ``x_anchor_2``), then
        run the base setup. Checked before ``super()`` so it fails before any model
        loading or rollouts."""
        if not self._rl_cfg().aux_v_enabled:
            raise ValueError(
                f"{type(self).__name__} reads x_anchor_2 off aux_v_model, but "
                "fewstep_rl.aux_v_enabled=false, so setup_models would never build "
                "that model and the first policy step would fail -- after a full "
                "rollout and reward-scoring phase. Set fewstep_rl.aux_v_enabled="
                "true, or use the meanflow_nft trainer, which derives x_anchor_2 "
                "from old_model instead."
            )
        super().setup_models()

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
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt``. Used by the KL path only,
        not by ``x_anchor_2``."""
        s = self._s_term(t_cd, r_cd, dF_dt, like=u_old)
        return u_old if s is None else u_old + s

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = x_t - t * u_old`` -- identical to MeanFlowNFT."""
        return inputs.x_t - inputs.t_normed * inputs.u_old

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_fake(x_t, t)`` -- the probe center.

        ``v_fake`` comes from :attr:`aux_v_model`; no central difference enters and
        ``inputs.dF_dt`` is unused. The AnyFlow-Wan aux model takes an ``r`` input;
        with ``model_u_pred=False`` the base queries it at ``r == t``, where the
        average velocity equals the instantaneous one. ``model_u_pred=True`` would
        lift with ``dF_dt`` instead, which is MeanFlowNFT's construction.
        """
        aux = self.aux_v_model
        if aux is None:
            raise RuntimeError(
                "DMDNFTTrainer needs aux_v_model for x_anchor_2; setup_models only "
                "builds it when fewstep_rl.aux_v_enabled is set."
            )
        # Narrowing back to the network dtype is exact: AnchorInputs.x_t is a
        # widened copy of the network-dtype x_t.
        v_fake = self._predict_instantaneous_velocity(
            model=aux,
            noisy_latents=inputs.x_t.to(next(aux.parameters()).dtype),
            prompt_embeds=inputs.prompt_embeds,
            timesteps=inputs.t_cd,
            pooled_embeds=inputs.pooled_embeds,
            r_timesteps=inputs.r_cd,
            # CFG-free, matching _aux_fm_loss: aux_v_model is trained on the
            # conditional velocity only.
            guidance_scale=1.0,
            model_u_pred=False,
        )
        # A raw network output (nothing folded), so a bf16 result is fine; just
        # run the x0 algebra in the work dtype.
        v_fake = v_fake.to(inputs.x_t.dtype)
        # Explicit raise, not assert: assert vanishes under `python -O`.
        if v_fake.requires_grad:
            raise RuntimeError(
                "v_fake carries grad; x_anchor_2 is a stop-grad probe center, and "
                "aux_v_model has its own optimizer -- a gradient path from the "
                "policy loss into it would train it on the wrong objective."
            )
        return inputs.x_t - inputs.t_normed * v_fake

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

        # use_v_kl decides whether the central difference runs. It is config-level,
        # hence rank-uniform -- required because the central difference issues FSDP
        # collectives.
        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step leaked "
                "its central difference (missing _post_compute_v_hook)."
            )

        # Both anchors are S-free, so dF/dt is only needed for the v-KL target
        # ``v_ref - (t - r) * dF/dt``. use_v_kl is always False on Wan (no
        # ref_v_model), so in practice the central difference is skipped.
        if use_v_kl:
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
                # None even if the v-KL central difference was computed: it must
                # not reach the anchors (that would turn this into MeanFlowNFT).
                dF_dt=None,
                t_cd=t_for_model,
                # Ignored by the aux query, which evaluates at r == t.
                r_cd=r_for_model if r_for_model is not None else t_for_model,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)
            # Keep a local reference: ``finally`` clears the attribute before the KL.
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        # 4. r_hat -- the advantage mapped into [0, 1].
        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        r_hat = ((adv / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)

        unscaled, form_info = self.decoupled_rl_loss(
            x_0_pred_u, x_anchor_1, x_anchor_2, x_0_f, r_hat, beta, spatial,
        )
        policy_loss = (unscaled * adv_clip_max).mean()
        loss = policy_loss

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
                # u-form deviation (no S).
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # Emitted unconditionally on every rank: _reduce_metrics
                # all-reduces per key, so a conditional key deadlocks. Here the
                # anchor gap measures old_model's average velocity vs the aux
                # marginal, not ``t * S``.
                "s_x0_abs_mean": float(s_x0.abs().mean()),
                "s_rel_median": float(
                    (s_x0.abs().mean() / dev_abs.clamp(min=1e-12))
                ),
                **form_info,
            }

        # 5. KL against the frozen reference.
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model,
            shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        # compute_kl_term always returns the same key set (KL on or off), keeping
        # the per-key all-reduce rank-uniform.
        info.update(kl_info)

        info["loss_total"] = float(loss.detach())
        return loss, info
