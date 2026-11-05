"""MeanFlowGRPO: flow-GRPO's clipped policy ratio on replayed SDE transitions.

Unlike the other trainers, which re-noise the rewarded sample at a fresh
``(t, r)``, this one replays the rollout's own ``(x_t, t) -> (x_r, r)``
transitions (from ``GeneralRLTrainer._rollout_samples_sde`` with
``fewstep_rl.reuse_x_t``) and scores them with flow-GRPO's Euler-Maruyama kernel.
It therefore requires ``rollout_type=sde``: a policy ratio needs nonzero
behaviour density at the replayed transition. The ratio is a true off-policy
importance weight, unlike MeanFlowAWM's, where it acts only as a trust region.

Approximation: the rollout is jump-and-renoise (kernel centre
``alpha * (x_t + (s - t) * u)``), while the scoring kernel is flow-GRPO's v-form
Euler-Maruyama; they coincide only when ``s == r``. The ratio is still exactly 1
at zero policy drift, since both means use the same expression; only the
per-transition weighting is approximate.
"""

from __future__ import annotations

from typing import Optional

import torch

from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer


class MeanFlowGRPOTrainer(GeneralRLTrainer):
    """MeanFlowGRPO trainer for RL."""

    TRAINER_KEY = "meanflowgrpo_rl"

    #: GRPO diagnostics, replacing the anchor-gap keys (undefined here since
    #: ``x_anchor_1`` is None). ``_run_rl_epoch`` silently drops keys not in an
    #: allowlist, so they must be declared. A class constant, hence rank-uniform,
    #: as required by the per-key all-reduce in ``_reduce_metrics``. ``clipfrac``
    #: near 0 means the trust region never binds; near 1 means it binds everywhere.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "s_mean",           # mean flow-map jump target (redrawn per transition)
        "r_mean",           # mean renoise endpoint
        "tau_mean",         # mean transition-kernel std
        "ratio_mean",       # E[rho]; 1.0 at zero drift
        "ratio_max_dev",    # max|rho - 1|; detects exp runaway
        "clipfrac",         # fraction of transitions outside the trust region
        "approx_kl",        # 0.5*E[(log rho)^2], the PPO KL proxy
    )

    #: ``sample_pack`` keys required from the rollout (None unless
    #: ``fewstep_rl.reuse_x_t`` is set). ``mid_timesteps`` is only presence- and
    #: shape-checked, since the training ``s`` is redrawn.
    _REPLAY_KEYS: tuple[str, ...] = (
        "x_t", "x_r", "start_timesteps", "mid_timesteps", "end_timesteps",
    )

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
        dF_dt: Optional[torch.Tensor]=None,
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
        """``x_anchor_2 = x_t - t * v_inst``."""
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        # Explicit raise, not assert: assert vanishes under `python -O`.
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"_instantaneous_velocity_from_u must return fp32/fp64, got {v_inst.dtype}. "
                "Returning the network's bf16 dtype reintroduces the fold defect."
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
        x_r_pred_u: torch.Tensor,
        x_anchor_1: Optional[torch.Tensor],  # noqa: ARG002 - always None here
        x_anchor_2: torch.Tensor,   # pi_old's mean for this transition
        x_r_rwd: torch.Tensor,      # the action pi_old actually sampled
        adv: torch.Tensor,          # occupies the anchor form's ``r_hat`` slot
        tau_r: torch.Tensor,        # occupies the anchor form's ``beta`` slot
        spatial: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GRPO weights ``(ratio, ratio, omega)`` for the base's reward term.

        The three slots encode the gradient of

            L = gate * ratio * adv / (2 * tau_r**2) * mse(x_r_pred_u, x_r_rwd)

        - ``ratio = exp((l_old - l_pred) / (2 tau_r^2))``: the Gaussian likelihood
          ratio of the sampled action under the current vs behaviour mean, since
          ``log p = -mse / (2 tau^2)``. The old loss leads; the reverse order gives
          ``1/rho``. ``tau_r`` is a per-sample ``[B]`` tensor.
        - ``omega = gate * adv / (2 tau_r^2)``: the ``1/(2 tau^2)`` is part of the
          per-transition gradient scale, not a global learning-rate factor.
        - ``gate``: PPO's ``max{-A*rho, -A*clip(rho, 1-eps, 1+eps)}`` as in
          flow-GRPO. The clipped branch is constant in ``rho``, so its gradient is
          either the unclipped one or zero, i.e. a 0/1 gate. ``>=`` matches
          ``torch.maximum`` taking its first argument on ties.

        The mse is a spatial mean, so this is the per-dimension log-ratio (the
        joint ratio would overflow), as in flow-GRPO; hence the small
        ``clip_range``. ``reg_weight`` is returned equal to ``ratio`` only so it is
        logged as ``affine_coeff_mean``; the self-regression term is disabled by
        ``x_anchor_1=None``. ``omega`` is broadcast to ``[B, 1, ...]``.
        """
        l_pred = self._per_sample_mse(x_r_pred_u, x_r_rwd, spatial)
        l_old = self._per_sample_mse(x_anchor_2, x_r_rwd, spatial)
        # tau_r must be [B]: a broadcast-shaped tau would silently produce a
        # [B, 1, ..., B] tensor here instead of failing.
        two_tau2 = 2.0 * tau_r**2
        # (l_old - l_pred), not the reverse.
        ratio = torch.exp((l_old - l_pred) / two_tau2)

        # PPO gate, elementwise over the batch.
        clip_range = float(self._rl_cfg().clip_range)
        unclipped = -adv * ratio
        clipped = -adv * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        inside = (ratio > 1.0 - clip_range) & (ratio < 1.0 + clip_range)
        gate = (inside | (unclipped >= clipped)).to(ratio.dtype)

        bcast = (-1, *([1] * (x_r_pred_u.ndim - 1)))
        omega = (gate * adv / two_tau2).view(*bcast)
        return ratio, ratio, omega

    def _rl_train_phase(self, sample_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replay the rollout's own transitions; redraw only the jump target s.

        The sampler produced ``K = sampling_num_steps`` transitions
        ``(x_t, t) -> (x_s, s) -> (x_r, r)`` per sample, so the inner loop indexes
        rollout steps instead of drawing ``(t, r)`` and re-noising ``x_0``
        (``_draw_training_tr`` and friends are unused here).

        ``s`` is redrawn per transition by
        :meth:`GeneralRLTrainer._draw_training_s_given_t` (``r``'s conditional law
        given ``t``), so ``u_theta(x_t, t, s)`` sees a continuum of jump lengths
        instead of a few grid values, while ``(x_t, x_r, t, r)`` stay as sampled.
        ``s`` is bounded by ``t``, not ``r``, so some draws have ``s > r``.

        The optimizer / grad-accum / EMA logic matches the base.
        """
        cfg = self._rl_cfg()
        device = torch.device("cuda")

        missing = [k for k in self._REPLAY_KEYS if sample_pack.get(k) is None]
        if missing:
            # Config-level, hence rank-uniform: raising cannot desync collectives.
            raise ValueError(
                f"{type(self).__name__} replays the rollout's own states, so it "
                f"requires fewstep_rl.reuse_x_t=true; sample_pack is missing "
                f"{missing}."
            )

        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        advantages = sample_pack["advantages"].to(device)
        # The trajectory's terminal latent, shared by all K steps of a sample.
        # Only the central difference reads it (see the caveat in _rl_loss).
        x_0_all = sample_pack["x_0"].to(device)
        # [n_local, K, ...]: x_t[:, k] enters step k and x_r[:, k] leaves it, so
        # x_t[:, k + 1] == x_r[:, k].
        x_t_all = sample_pack["x_t"].to(device)
        x_r_all = sample_pack["x_r"].to(device)
        # [n_local, K], raw timestep units. t -> s is the flow-map jump, s -> r
        # the renoise. The rollout's s is only shape-checked; training redraws s.
        t_all = sample_pack["start_timesteps"].to(device=device, dtype=torch.float32)
        s_all = sample_pack["mid_timesteps"].to(device=device, dtype=torch.float32)
        r_all = sample_pack["end_timesteps"].to(device=device, dtype=torch.float32)

        n_local, n_steps = t_all.shape
        if n_local == 0:
            return {}
        for name, tensor in (
            ("x_t", x_t_all), ("x_r", x_r_all),
            ("mid_timesteps", s_all), ("end_timesteps", r_all),
        ):
            if tensor.shape[:2] != (n_local, n_steps):
                raise ValueError(
                    f"sample_pack['{name}'] has leading shape "
                    f"{tuple(tensor.shape[:2])}, expected {(n_local, n_steps)}. "
                    "The rollout must stack samples on dim 0 and rollout steps "
                    "on dim 1."
                )
        # x_0 has no step axis.
        if x_0_all.shape[0] != n_local:
            raise ValueError(
                f"sample_pack['x_0'] has leading dim {x_0_all.shape[0]}, expected "
                f"{n_local}. It is indexed by the same row permutation as the "
                "per-step tensors, so a mismatch would pair a state with "
                "another sample's endpoint."
            )

        inner_bs = int(cfg.rl_inner_batch_size) if cfg.rl_inner_batch_size > 0 else n_local
        inner_bs = max(1, min(inner_bs, n_local))
        n_inner_epochs = max(1, int(cfg.num_inner_epochs))
        max_grad_norm = float(self.config.solver.generator.max_grad_norm)
        grad_accum = max(1, int(self.config.train.gradient_accumulation_steps))
        # One backward per rollout step, so the rollout length replaces
        # ``num_training_timesteps_per_sample`` in the accumulation window.
        effective_grad_accum_steps = grad_accum * n_steps

        self.generator.train()
        accumulator: dict[str, float] = {}
        n_backward = 0       # cumulative backwards across the whole train phase
        n_opt_steps = 0
        current_accumulated_steps = 0
        self.optimizers["generator"].zero_grad(set_to_none=True)
        last_grad_norm: float = 0.0

        # Used only by the s|t sampler, which draws in the network dtype to
        # consume the RNG the same way as the base's timestep draws.
        dtype = next(self.generator.parameters()).dtype

        # inner_epoch_idx keys the s|t RNG, so each inner epoch sees new s draws.
        for inner_epoch_idx in range(n_inner_epochs):
            # Shuffle samples only; the step axis must stay aligned with its
            # timesteps.
            perm = torch.randperm(n_local, device=device)
            sx0 = x_0_all[perm]
            sxt = x_t_all[perm]
            sxr = x_r_all[perm]
            st = t_all[perm]
            sr = r_all[perm]
            spe = prompt_embeds[perm]
            spp = pooled_embeds[perm]
            sadv = advantages[perm]

            for start in range(0, n_local, inner_bs):
                # Clamped: the s|t sampler sizes its draw from ``end``, which
                # must not exceed the rows of a ragged final chunk.
                end = min(start + inner_bs, n_local)
                pe_b = spe[start:end]
                pp_b = spp[start:end]
                adv_b = sadv[start:end]
                for k in range(n_steps):
                    t_b = st[start:end, k]
                    r_b = sr[start:end, k]
                    # Fresh jump target s ~ p(r | t); the rollout step k fills
                    # the base's ``j_idx`` RNG slot.
                    s_b = self._draw_training_s_given_t(
                        t_b,
                        state={},   # this arm carries no per-inner-epoch draw state
                        chunk_start=start,
                        chunk_end=end,
                        j_idx=k,
                        inner_epoch_idx=inner_epoch_idx,
                        device=device,
                        dtype=dtype,
                    )
                    loss, info = self._rl_loss(
                        x_0=sx0[start:end],
                        x_t=sxt[start:end, k],
                        x_r=sxr[start:end, k],
                        prompt_embeds=pe_b,
                        pooled_embeds=pp_b,
                        # One advantage per trajectory, shared by its K steps.
                        advantages=adv_b,
                        t_raw=t_b,
                        s_raw=s_b,
                        r_raw=r_b,
                    )
                    # Scale loss for grad accumulation; gradients accumulate
                    # across `effective_grad_accum_steps` backward calls.
                    (loss / float(effective_grad_accum_steps)).backward()
                    current_accumulated_steps += 1
                    n_backward += 1

                    for key, val in info.items():
                        accumulator[key] = accumulator.get(key, 0.0) + float(val)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        last_grad_norm = self._compute_grad_norm(self.generator)
                        if max_grad_norm > 0:
                            self._clip_grad_norm(self.generator, max_grad_norm)
                        self.optimizers["generator"].step()
                        self.schedulers["generator"].step()
                        self.optimizers["generator"].zero_grad(set_to_none=True)

                        # EMA per opt step, with the warmup-aware decay.
                        if (
                            self.generator_ema is not None
                            and self.global_step % self.config.train.ema_every == 0
                        ):
                            self.ema_update(
                                self.generator, self.generator_ema,
                                self._current_ema_decay(),
                            )

                        # ``global_step`` counts optimizer steps.
                        self.global_step += 1
                        n_opt_steps += 1
                        accumulator["grad_norm"] = (
                            accumulator.get("grad_norm", 0.0) + float(last_grad_norm)
                        )

        if n_backward == 0:
            return {}
        averaged = {k: v / n_backward for k, v in accumulator.items()}
        if n_opt_steps > 0:
            averaged["grad_norm"] = accumulator.get("grad_norm", 0.0) / n_opt_steps
        averaged["lr"] = self.schedulers["generator"].get_last_lr()[0]
        averaged["inner_backward_steps"] = float(n_backward)
        averaged["inner_opt_steps"] = float(n_opt_steps)
        # Backwards per sample per inner epoch (= rollout length here).
        averaged["num_train_timesteps"] = float(n_steps)
        averaged["effective_grad_accum_steps"] = float(effective_grad_accum_steps)
        return averaged


    def _rl_loss(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        x_r: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        s_raw: torch.Tensor,
        r_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One GRPO step over one replayed rollout transition.

        ``x_t`` / ``x_r`` are the visited states and ``(t, r)`` their grid
        endpoints; ``s`` is the freshly drawn jump target ``u_theta`` is queried
        over. ``s`` enters only through the MeanFlow lift
        ``v = u + (t - s) * dF/dt``; the transition is then scored with
        flow-GRPO's Euler-Maruyama kernel, which depends on ``(t, r)`` alone::

            std_dev_t = noise_level * sqrt(t / (1 - t))
            a         = 1 - std_dev_t^2 / (2 t) * h,     h = t - r
            b         = (1 + std_dev_t^2 (1 - t) / (2 t)) * h
            mu        = a * x_t - b * v
            tau       = std_dev_t * sqrt(h)

            L     = gate * ratio * adv / (2*tau**2) * mse(mu_theta, x_r)
            ratio = exp((mse(mu_old, x_r) - mse(mu_theta, x_r)) / (2*tau**2))

        The factors are built by :meth:`decoupled_weights`. ``fewstep_rl.beta`` is
        unused; ``clip_range`` and ``grpo_noise_level`` are the relevant knobs.

        ``x_0`` is the terminal latent, used only by ``_pre_compute_v_hook``. With
        ``cd_velocity_source="noise_minus_x0"`` it recovers ``eps`` assuming
        ``x_t`` lies on the straight path to ``x_0``, which the SDE renoise makes
        only approximately true; this biases the central-difference direction.
        ``cd_velocity_source="u_self"`` avoids it.
        """
        cfg = self._rl_cfg()
        batch_size = x_r.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)
        wdt = self._rl_work_dtype

        device = x_r.device
        t_for_model = t_raw.to(device=device, dtype=torch.float32)
        # The flow map is queried at (t, s), not (t, r): s is the jump target.
        s_for_model = s_raw.to(device=device, dtype=torch.float32)
        r_for_model = r_raw.to(device=device, dtype=torch.float32)

        # Normalised times in the work dtype, matching how _rollout_samples_sde
        # built x_r (the anchor-form trainers use the network dtype instead).
        bcast = (batch_size, *([1] * (x_r.ndim - 1)))
        t_normed_b_f = (t_for_model / T).view(bcast).to(wdt)
        r_normed_b_f = (r_for_model / T).view(bcast).to(wdt)

        # Decided before the velocity block, which must know whether to keep the
        # shared dF/dt past its ``finally``. Config-level, hence rank-uniform.
        # The KL uses ``s`` in the ``r`` slot (target ``v_ref - (t - s) * dF/dt``).
        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        # 2. Average velocities over [s, t]; S is not folded in. The ``r`` of the
        # MeanFlow identity is ``s`` here (the flow map never sees the renoise
        # endpoint r), so the central difference is taken at fixed s.
        self._pre_compute_v_hook(
            x_t=x_t, t_for_model=t_for_model, r_for_model=s_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds, x_0=x_0,
        )
        try:
            u_theta = self._compute_u(
                self.generator, x_t, t_for_model, prompt_embeds, pooled_embeds,
                s_for_model, with_grad=True,
            )
            u_old = self._compute_u(
                self.old_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                s_for_model, with_grad=False,
            ).detach()

            u_theta_f = u_theta.to(wdt)
            u_old_f = u_old.to(wdt)
            x_t_f = x_t.to(wdt)
            x_r_f = x_r.to(wdt)
            spatial = tuple(range(1, x_r.ndim))

            # 3. Lift both to instantaneous velocity with the same S:
            # ``v = u + (t - s) * dF/dt``. A shared S keeps the ratio exactly 1
            # at zero policy drift (``_pre_compute_v_hook`` requires
            # ``share_cd_with_old``). dF/dt is a no-grad constant, so v_theta_f
            # carries exactly u_theta_f's gradient.
            v_theta_f = self._instantaneous_velocity_from_u(
                u_theta_f, t_for_model, s_for_model, self._shared_dF_dt,
            )
            v_old_f = self._instantaneous_velocity_from_u(
                u_old_f, t_for_model, s_for_model, self._shared_dF_dt,
            )
            # Keep a local reference for the v-mode KL: ``finally`` clears the
            # attribute. Taken only when needed so the tensor can be freed.
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()


        adv_clip_max = float(cfg.adv_clip_max)
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

        # flow-GRPO's SDE transition coefficients, with sigma = t, h = t - r:
        #
        #     std_dev_t = noise_level * sqrt(sigma / (1 - sigma))
        #     a         = 1 - std_dev_t^2 / (2 sigma) * h
        #     b         = (1 + std_dev_t^2 (1 - sigma) / (2 sigma)) * h
        #     mean      = a * x_t - b * v
        #     tau       = std_dev_t * sqrt(h)
        #
        # The naive Euler mean ``x_t + (r - t) * v`` is only the noise_level -> 0
        # limit and would misplace the kernel centre.
        h = t_normed_b_f - r_normed_b_f                    # > 0, the step length
        # As in flow-GRPO, sigma == 1 is replaced by sigma_max in the denominator
        # of std_dev_t only (to avoid dividing by zero); the numerator and a / b
        # keep the raw sigma.
        sigma = t_normed_b_f
        sigma_max = float(self.flowmap_scheduler.timesteps[1]) / T
        denom = 1.0 - torch.where(
            sigma >= 1.0, torch.full_like(sigma, sigma_max), sigma,
        )
        std_dev_t = float(cfg.grpo_noise_level) * torch.sqrt(sigma / denom)
        a_coef = 1.0 - std_dev_t**2 / (2.0 * sigma) * h     # dt = -h
        b_coef = (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * h

        # Current and behaviour means of the transition kernel (see the module
        # docstring for the jump-and-renoise approximation).
        x_r_pred_u = a_coef * x_t_f - b_coef * v_theta_f    # carries grad
        x_r_old_u = a_coef * x_t_f - b_coef * v_old_f

        # Flattened to [B] to match the per-sample losses. No floor needed:
        # tau > 0 on every grid node.
        tau_r = (std_dev_t * torch.sqrt(h)).reshape(batch_size)

        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)

        # Reuse the decoupled loss with the reward half only: x_anchor_1=None
        # disables self-regression, x_anchor_2 carries pi_old's mean for the
        # ratio, and the r_hat / beta slots take adv and tau_r. The result has
        # the gradient of ``gate * ratio * adv / (2*tau_r**2) * mse(x_r_pred_u, x_r)``.
        x_anchor_1 = None
        x_anchor_2 = x_r_old_u
        unscaled, form_info = self.decoupled_rl_loss(
            x_r_pred_u, x_anchor_1, x_anchor_2, x_r_f, adv, tau_r, spatial,
        )
        # Not rescaled by adv_clip_max: the anchor-form trainers do that to undo
        # the r_hat squash into [0, 1], but omega here is already in raw units.
        policy_loss = unscaled.mean()
        loss = policy_loss

        # ``old_deviate`` is the u-space divergence between the policy and the
        # behaviour policy at the replayed query point.
        with torch.no_grad():
            # Standard GRPO health metrics, recomputed because decoupled_weights'
            # return signature is fixed by the base.
            clip_range = float(cfg.clip_range)
            l_pred_d = self._per_sample_mse(x_r_pred_u.detach(), x_r_f, spatial)
            l_old_d = self._per_sample_mse(x_r_old_u, x_r_f, spatial)
            log_rho = (l_old_d - l_pred_d) / (2.0 * tau_r**2)
            rho_d = torch.exp(log_rho)
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                # t and r follow the rollout grid; s follows the per-transition draw.
                "t_mean": float(t_raw.mean().item()),
                "s_mean": float(s_raw.mean().item()),
                "r_mean": float(r_raw.mean().item()),
                "tau_mean": float(tau_r.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                # Allowlisted key name; measures the replayed x_r here.
                "x0_norm_mean": float((x_r_f ** 2).mean().item()),
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                # ratio_mean drifting off 1 or clipfrac near 1 means clip_range is
                # too tight for the current policy drift.
                "ratio_mean": float(rho_d.mean()),
                "ratio_max_dev": float((rho_d - 1.0).abs().max()),
                "clipfrac": float((rho_d - 1.0).abs().gt(clip_range).float().mean()),
                # 0.5*E[(log rho)^2], the standard PPO KL proxy.
                "approx_kl": float(0.5 * (log_rho**2).mean()),
                **form_info,
            }
        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, s_for_model,
            shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        # compute_kl_term always returns the same key set (KL on or off), keeping
        # the per-key all-reduce rank-uniform.
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

        Unlike the base (constant direction ``x_pos - x_neg``), the direction is
        measured from ``x_pred``; ``x_neg`` is unused but kept for the positional
        call in :meth:`GeneralRLTrainer.decoupled_rl_loss`. With
        ``x_target = x_pred.detach() + omega * (x_pos - x_pred.detach())``::

            dL/dx_pred = -weight * omega * (2/N) * (x_pos - x_pred)
                       = d/dx_pred [ weight * omega * mean_sp((x_pred - x_pos)^2) ]

        exactly. The loss value carries ``omega`` squared (always positive), so
        compare gradients, not ``loss_policy``. ``omega > 0`` must pull ``x_pred``
        toward ``x_pos``, hence ``norm_grad`` is added to the detached center.
        """
        with torch.no_grad():
            norm_grad = (omega * (x_pos.double() - x_pred.double())).to(x_pred.dtype)
        x_target = (x_pred.detach() + norm_grad).to(x_pred.dtype)
        return weight * self._per_sample_mse(x_pred, x_target, spatial)
