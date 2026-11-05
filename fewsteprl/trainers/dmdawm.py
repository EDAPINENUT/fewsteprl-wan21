"""DMD-AWM: MeanFlowAWM with the aux v-model as the reward baseline, no central difference.

``aux_v_model`` is a v-predictor trained online by flow matching on the
generator's rollouts (:meth:`GeneralRLTrainer._run_aux_epoch`), so ``v_fake``
estimates the marginal ``E[eps - x_0 | x_t, t]`` over the rollout distribution.
The prediction is ``x_t - t * u_theta`` and the ratio anchor ``x_t - t * u_old``
(``old_model``, an EMA that lags the policy, which is what makes the clip a trust
region); ``v_fake`` only replaces the reward term's surrogate-gradient center
with ``x_t - t * v_fake``, giving::

    dL/du_theta = ratio * gate * adv * (v_fake - v_tgt) / N,   v_tgt = eps - x_0.

Since ``E[adv] = 0`` within a group, the aux marginal acts as a learned per-sample
baseline. Not DMD in the strict sense: there is no real-score model and no
backprop through the rollout.

The central difference is dropped (``dF/dt := 0``), which saves two ``old_model``
forwards per step. ``(t, r)`` is still drawn by the three-mode sampler; dropping S
leaves the gradient direction unchanged and only shifts the per-sample ratio
through ``log_rho`` by ``-<u_theta - u_old, S>`` (spatial mean).
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.meanflowawm import MeanFlowAWMTrainer
from fewsteprl.trainers.rl_general_base import AnchorInputs


class DMDAWMTrainer(MeanFlowAWMTrainer):
    """AWM with ``old_model`` as the ratio reference and aux_v as the reward baseline."""

    TRAINER_KEY = "dmdawm_rl"

    #: Whether the prediction and the anchor include the detached
    #: ``S = (t - r) dF/dt``. Both False (and no v-KL) => ``_pre_compute_v_hook``
    #: is skipped and ``_shared_dF_dt`` stays None. They must stay equal so S is
    #: absent from both prediction and anchor.
    _PRED_USES_S: bool = False
    _ANCHOR_USES_S: bool = False

    #: AWM's panel plus ``rwd_dir_abs_mean``; ``s_x0_abs_mean`` here measures
    #: ``|x_0_pred - aux marginal|`` instead of ``|t * S|``. A class constant so the
    #: key set is rank-uniform for ``_reduce_metrics``.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "r_mean",
        "sigma_mean",
        "s_x0_abs_mean",     # |x_0_pred - aux marginal|
        "rwd_dir_abs_mean",  # |x_0 - reward center|
        "ratio_mean",
        "ratio_max_dev",
        "clipfrac",
        "approx_kl",
    )

    #: Per-step stash for the reward term's probe center, set and cleared in
    #: :meth:`_rl_loss` and read by :meth:`reward_augmentation`.
    #: ``decoupled_rl_loss`` forwards one ``x_anchor_2`` to both the ratio and the
    #: reward term, but here they need different tensors.
    _rwd_center: Optional[torch.Tensor] = None

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) dF/dt``, with ``dF_dt=None`` meaning S is 0 by design.

        The parent raises on ``dF_dt=None`` with ``r != t``; here the central
        difference is skipped on purpose while ``r`` stays random.
        """
        if dF_dt is None:
            return None
        return super()._s_term(t_cd, r_cd, dF_dt, like)

    def setup_models(self) -> None:
        """Require ``aux_v_enabled`` before building any model, to fail fast."""
        if not self._rl_cfg().aux_v_enabled:
            raise ValueError(
                f"{type(self).__name__} reads the reward term's probe center off "
                "aux_v_model, but fewstep_rl.aux_v_enabled=false, so setup_models "
                "would never build that model and the first policy step would "
                "fail -- after a full rollout and reward-scoring phase. Set "
                "fewstep_rl.aux_v_enabled=true, or use the meanflowawm_rl trial, "
                "which centers the reward term on the policy's own prediction."
            )
        super().setup_models()

    @torch.no_grad()
    def _aux_v_x0(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_t - t * v_fake(x_t, t)`` -- the aux network's amortized x0 marginal.

        ``v_fake`` comes straight from :attr:`aux_v_model` (queried at ``r == t``,
        CFG-free, matching ``_aux_fm_loss``); ``inputs.dF_dt`` / ``inputs.u_old``
        are unused here.
        """
        aux = self.aux_v_model
        if aux is None:
            raise RuntimeError(
                "DMDAWMTrainer needs aux_v_model; setup_models only builds it "
                "when fewstep_rl.aux_v_enabled is set."
            )
        v_fake = self._predict_instantaneous_velocity(
            model=aux,
            noisy_latents=inputs.x_t.to(next(aux.parameters()).dtype),
            prompt_embeds=inputs.prompt_embeds,
            timesteps=inputs.t_cd,
            pooled_embeds=inputs.pooled_embeds,
            r_timesteps=inputs.r_cd,
            guidance_scale=1.0,
            model_u_pred=False,
        )
        # The x0 algebra runs in the work dtype, like x_anchor_2.
        v_fake = v_fake.to(inputs.x_t.dtype)
        if v_fake.requires_grad:
            raise RuntimeError(
                "v_fake carries grad; it is a stop-grad reference point and "
                "aux_v_model has its own optimizer -- a gradient path from the "
                "policy loss into it would train it on the wrong objective."
            )
        return inputs.x_t - inputs.t_normed * v_fake

    def reward_augmentation(
        self,
        x_pred: torch.Tensor,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,  # noqa: ARG002 - the center is the stash, not x_neg
        omega: torch.Tensor,
        weight: torch.Tensor,
        spatial: tuple[int, ...],
    ) -> torch.Tensor:
        """Injects the gradient of ``weight * omega * mse(x_pred, center)``.

        ``x_target = x_pred.detach() + omega * (x_pos - center)``, so
        ``dL/dx_pred = -weight * omega * (2/N) * (x_pos - center)`` and
        ``omega > 0`` pulls ``x_pred`` along ``+(x_pos - center)``.

        The center is the aux marginal ``x_t - t * v_fake`` from the
        :attr:`_rwd_center` stash (AWM uses ``x_pred.detach()``). An empty stash
        raises rather than silently falling back to plain AWM.
        """
        center = self._rwd_center
        if center is None:
            raise RuntimeError(
                f"{type(self).__name__} uses the aux marginal as the reward center, but the aux probe "
                "center stash is empty -- reward_augmentation ran outside the "
                "_rl_loss block that fills it. Silently falling back to "
                "x_pred.detach() here would turn the trial into plain AWM without "
                "changing one logged number."
            )
        if center.shape != x_pred.shape:
            raise ValueError(
                f"aux probe center has shape {tuple(center.shape)}, expected "
                f"{tuple(x_pred.shape)}."
            )
        with torch.no_grad():
            norm_grad = (omega * (x_pos.double() - center.double())).to(x_pred.dtype)
        x_target = (x_pred.detach() + norm_grad).to(x_pred.dtype)
        return weight * self._per_sample_mse(x_pred, x_target, spatial)

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One DMD-AWM step: AWM's step plus the stashed aux reward center.

        A full override because the stash must be filled while the shared
        ``dF_dt`` is alive and cleared within the same step.
        """
        cfg = self._rl_cfg()
        dtype = x_0.dtype
        batch_size = x_0.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)
        wdt = self._rl_work_dtype

        # 1. Forward noising, identical to MeanFlowAWMTrainer._rl_loss.
        bcast = (batch_size, *([1] * (x_0.ndim - 1)))
        t_raw = t_raw.to(device=x_0.device)
        t_normed = t_raw.to(dtype) / T
        t_broadcast = t_normed.view(bcast)
        eps = self._draw_eps(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * eps
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = r_raw.to(dtype=torch.float32) if r_raw is not None else None
        # Without r, use r == t (S == 0); _pre_compute_v_hook gets the raw r.
        r_cd = r_for_model if r_for_model is not None else t_for_model

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

        # 2. One shared central difference, so prediction and anchor use the same
        # S. The predicate is class-level, hence rank-uniform (the CD issues FSDP
        # collectives).
        need_cd = self._PRED_USES_S or self._ANCHOR_USES_S or use_v_kl
        if need_cd:
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

            # 3. Prediction, as in AWM (S is detached; u_theta carries the grad).
            if self._PRED_USES_S:
                v_theta_f = self._instantaneous_velocity_from_u(
                    u_theta_f, t_for_model, r_cd, self._shared_dF_dt,
                )
                x_0_pred_v = x_t_f - t_normed_b_f * v_theta_f     # carries grad
            else:
                x_0_pred_v = x_t_f - t_normed_b_f * u_theta_f      # carries grad

            anchor_inputs = AnchorInputs(
                x_t=x_t_f,
                t_normed=t_normed_b_f,
                u_old=u_old_f,
                u_theta=u_theta_f.detach(),
                # Same S as the prediction above.
                dF_dt=self._shared_dF_dt if self._ANCHOR_USES_S else None,
                t_cd=t_for_model,
                r_cd=r_cd,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
            )
            x_anchor_1 = self._compute_x_anchor_1(anchor_inputs)   # None on AWM
            x_anchor_2 = self._compute_x_anchor_2(anchor_inputs)   # old_model, for the ratio

            # Aux marginal as the reward center; cleared after the loss below.
            self._rwd_center = self._aux_v_x0(anchor_inputs)
            s_x0_abs_mean = float(
                (x_0_pred_v.detach() - self._rwd_center).abs().mean()
            )

            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        # Same t_normed that built x_t, so the t**2 cancellation is exact.
        sigma = t_normed.to(wdt).reshape(batch_size).clamp(min=self._SIGMA_MIN)

        # Magnitude of the reward direction, measured before the stash is cleared.
        with torch.no_grad():
            rwd_dir_abs_mean = float((x_0_f - self._rwd_center).abs().mean())

        try:
            unscaled, form_info = self.decoupled_rl_loss(
                x_0_pred_v, x_anchor_1, x_anchor_2, x_0_f, adv, sigma, spatial,
            )
        finally:
            # Never let the stash outlive this step.
            self._rwd_center = None
        # NOT rescaled by adv_clip_max -- see MeanFlowAWMTrainer._rl_loss.
        policy_loss = unscaled.mean()
        loss = policy_loss

        with torch.no_grad():
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
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # |x_0_pred - aux marginal| (|t * S| on MeanFlowAWM).
                "s_x0_abs_mean": s_x0_abs_mean,
                "rwd_dir_abs_mean": rwd_dir_abs_mean,
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
        # Merged unconditionally to keep the key set rank-uniform.
        info.update(kl_info)

        info["loss_total"] = float(loss.detach())
        return loss, info
