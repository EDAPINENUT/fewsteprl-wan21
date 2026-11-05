"""ReFL as a corner of few-step MeanFlow-OPSD (trainer key ``meanflowrefl_rl``).

Take the decoupled three-term OPSD loss and impose (1) a single reward-gradient
step (``M_tgt = 1``), (2) no self-regression (``W_reg = 0``) and (3) the ascent
term only (no repulsion from ``xm``). What remains is one square whose gradient
is ReFL's, with ``lambda`` replaced by a detached positive constant and
``grad R`` read at the behaviour anchor instead of the live prediction; the two
coincide at ``theta = theta_old``.

Condition 2 is not reachable by config (``opsd_dual_neg=false`` still leaves the
proximal weight ``W = beta^2 * a_+ > 0``), hence a separate trainer. Rollout,
target builder, anchors, lift and fitting loop mirror ``meanflowopsd.py``. The
differences: :meth:`MeanFlowReFLTrainer.refl_loss` (one ``reward_augmentation``
call) replaces ``decoupled_opsd_loss``; ``omega`` defaults to 1 like official
ReFL (``fewstep_rl.refl_use_advantage`` restores the advantage weighting); the
trainer key differs; and the logged keys of the dropped terms are pinned at 0.0.

x0-space names:

    x_0_pred_v    live prediction, the only tensor carrying grad
    x_anchor_1    lifted anchor (self-regression target, unused in the loss)
    x_anchor_2    probe centre, where grad R is read
    xp / xm       reward ascent / descent targets (``xm`` unused in the loss)

``x_anchor_1`` and ``xm`` are still built and packed so the pipeline matches the
OPSD trainer; ``x_anchor_1`` also feeds the ``s_x0_abs_mean`` / ``s_rel_median``
diagnostics. ``grad R`` comes from
:class:`~fewsteprl.trainers.opsd_grad_mixin.OPSDGradMixin`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from fewsteprl.trainers.opsd_grad_mixin import OPSDGradMixin
from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer

_DIFFERENTIABLE_REWARDS = frozenset({'aesthetic', 'clipscore', 'hpsv2', 'pickscore'})


class MeanFlowReFLTrainer(OPSDGradMixin, GeneralRLTrainer):
    """ReFL corner of few-step MeanFlow-OPSD; inherits only the RL base."""

    #: Written into every checkpoint and checked on resume, so it must differ
    #: from the OPSD trainer's key.
    TRAINER_KEY = "meanflowrefl_rl"

    #: Must be rank-uniform: _reduce_metrics all-reduces per key and hangs on a
    #: divergent key set, so every key is emitted on every sigma mode (on
    #: "fixed" the sigma_q std/min/max read (0, sigma*, sigma*)).
    #: ``s_x0_abs_mean`` / ``s_rel_median`` monitor
    #: ``E = x_anchor_1 - x_anchor_2`` (zero for the v-form anchor pair).
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = ('s_x0_abs_mean', 's_rel_median', 'opsd_delta_res', 'opsd_sigma_q', 'opsd_r_mean', 'opsd_r_frac_zero', 'opsd_frame_shift', 'opsd_sigma_q_std', 'opsd_sigma_q_min', 'opsd_sigma_q_max', 'refl_lambda_eff')

    #: Same tuple as the OPSD trainer so logs stay comparable.
    #: ``self_regression_loss``, ``opsd_push_neg`` and ``opsd_omega_neg_mean``
    #: belong to terms this trainer drops and are logged as a constant 0.0.
    _EXTRA_LOSS_KEYS: tuple[str, ...] = ('self_regression_loss', 'reward_augmentation_loss', 'loss_kl_unweighted', 'kl_t_weight_mean', 'kl_rebalance_mean', 'opsd_pull_pos', 'opsd_push_neg', 'opsd_omega_pos_mean', 'opsd_omega_neg_mean')

    #: Per-sample tensors threaded from the sampling phase to the fitting
    #: phase (not defined on GeneralRLTrainer).
    _OPSD_KEYS: tuple[str, ...] = ('opsd_z_q', 'opsd_y_0', 'opsd_y_plus', 'opsd_y_minus')

    def __init__(self, config) -> None:
        super().__init__(config)
        # Resident differentiable scorer, distinct from the base's
        # ``_reward_scorer`` (see :meth:`_ensure_grad_scorer`).
        self._opsd_grad_scorer: Optional[Any] = None
        # Populated by :meth:`_rl_sampling_phase`, read by :meth:`_rl_train_phase`.
        self._opsd_last_stats: dict[str, float] = {}

    @staticmethod
    def _bcast(sigma_q: Any, like: torch.Tensor) -> torch.Tensor:
        """Broadcast a ``[B]`` (or scalar) noise level over ``like``'s rank.

        A bare ``[B]`` tensor would right-align onto the last axis instead of
        the batch axis. Python floats (``opsd_sigma_mode="fixed"``,
        ``_opsd_unpack_sigma``) are promoted to ``[B]`` first.
        """
        if not torch.is_tensor(sigma_q):
            sigma_q = torch.as_tensor(
                sigma_q, device=like.device, dtype=torch.float32,
            )
        if sigma_q.ndim == 0:
            sigma_q = sigma_q.expand(like.shape[0])
        if sigma_q.ndim != 1 or sigma_q.shape[0] != like.shape[0]:
            raise ValueError(
                f"sigma_q must be [B] with B == {like.shape[0]}, got "
                f"{tuple(sigma_q.shape)}."
            )
        return sigma_q.to(device=like.device, dtype=like.dtype).view(
            -1, *([1] * (like.ndim - 1))
        )

    @torch.no_grad()
    def _build_opsd_targets(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        prompts: list[str],
        z_q: torch.Tensor,
        sigma_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Build the probe anchor and reward targets with a per-sample ``sigma_q``.

        The chunk-loop count is a config constant times a rank-uniform sample
        count, so every rank runs the same number of scorer forwards.
        """
        cfg = self._rl_cfg()
        device = next(self.generator.parameters()).device
        n = x_0.shape[0]
        chunk = max(1, int(cfg.opsd_target_chunk))
        dir_mode = str(cfg.opsd_dir_mode)
        need_grad = dir_mode == "grad" and int(cfg.opsd_n_ascent) > 0
        if need_grad:
            self._ensure_grad_scorer()

        T = float(self.scheduler.config.num_train_timesteps)
        sigma_q = sigma_q.to(device=device, dtype=torch.float32)

        anchor_out = torch.empty_like(x_0, dtype=torch.float32)
        xp_out = torch.empty_like(x_0, dtype=torch.float32)
        xm_out = torch.empty_like(x_0, dtype=torch.float32)
        res_sum = 0.0

        net_dtype = next(self.generator.parameters()).dtype
        self.old_model.eval()

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            zq_c = z_q[start:end].to(device=device, dtype=net_dtype)
            x0_c = x_0[start:end].to(device=device, dtype=torch.float32)
            pe_c = prompt_embeds[start:end].to(device=device, dtype=net_dtype)
            pp_c = pooled_embeds[start:end].to(device=device)
            prm_c = list(prompts[start:end])
            s_c = sigma_q[start:end]

            t_c = (s_c * T).to(torch.float32)
            r_c = torch.zeros_like(t_c)

            # Routed through the anchor hook so targets are built in the same
            # (v-form) coordinates the loss fits in.
            x_anchor = self._opsd_build_anchor(
                zq_c, t_c, r_c, pe_c, pp_c, s_c, start, end, x0_c,
            )

            g0 = None
            if need_grad:
                # ``_reward_grad_chunked`` returns the gradient itself rather
                # than a graph: holding one graph over Wan's VAE decode does not
                # fit in memory (see opsd_grad_mixin).
                with torch.enable_grad():
                    _, g0 = self._reward_grad_chunked(x_anchor.detach(), prm_c)
                g0 = g0.detach()

            with torch.enable_grad():
                xp = self._opsd_tr_step(
                    x_anchor, prm_c, +1.0, x_end=x0_c, first_grad=g0,
                )
                xm = self._opsd_tr_step(
                    x_anchor, prm_c, -1.0, x_end=x0_c, first_grad=g0,
                )

            anchor_out[start:end] = x_anchor
            xp_out[start:end] = xp
            xm_out[start:end] = xm
            res_sum += float(
                (
                    (xp - x_anchor).flatten(1).norm(dim=1)
                    / (x_anchor.flatten(1).norm(dim=1) + 1e-8)
                ).sum()
            )
            del g0

        self._opsd_release_grad_memory()
        torch.cuda.empty_cache()

        return anchor_out, xp_out, xm_out, {
            "opsd_delta_res": res_sum / max(1, n),
            # Mean realised level; sigma varies per sample.
            "opsd_sigma_q": float(sigma_q.mean()),
        }

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = z_q - sigma_q * v_old`` -- the lifted (v-form) anchor.

        Must match the coordinates of ``x_0_pred_v`` (lifted in
        :meth:`_opsd_clean_output`); a u-form anchor would leave a constant
        ``-sigma_q * S`` residual at ``u_theta == u_old``. With both lifted,
        ``x_0_pred_v - x_anchor_1 = -sigma_q * (u_theta - u_old)``.
        """
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"the lift must stay in fp32/fp64, got {v_inst.dtype}."
            )
        if v_inst.requires_grad:
            raise RuntimeError(
                "_compute_x_anchor_1 must be detached; it is a stop-grad "
                "reference point."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    @torch.no_grad()
    def _compute_x_anchor_2(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_2 = x_t - t * v_inst`` -- the probe center."""
        v_inst = self._instantaneous_velocity_from_u(
            inputs.u_old, inputs.t_cd, inputs.r_cd, inputs.dF_dt,
        )
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"_instantaneous_velocity_from_u must return fp32/fp64, got "
                f"{v_inst.dtype}. Returning the network's bf16 dtype "
                "reintroduces the fold defect."
            )
        if v_inst.requires_grad:
            raise RuntimeError(
                "_instantaneous_velocity_from_u must be detached; x_anchor_2 is "
                "a stop-grad probe center."
            )
        return inputs.x_t - inputs.t_normed * v_inst

    def _instantaneous_velocity_from_u(
        self,
        u_old: torch.Tensor,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``v(x_t, t) = u(x_t, r, t) + (t - r) * dF/dt``."""
        s = self._s_term(t_cd, r_cd, dF_dt, like=u_old)
        return u_old if s is None else u_old + s

    @torch.no_grad()
    def _opsd_build_anchor(
        self,
        z_q: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        sigma_q: Any,
        chunk_start: int,
        chunk_end: int,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Return the v-form anchor and stash the shift to the u-form anchor.

            anchor_v = z_q - sigma * v_old(z_q, t)
            shift    = [z_q - sigma * u_old(z_q, r=0, t)] - anchor_v

        :meth:`_opsd_tr_step` uses ``shift`` to build the reward targets in the
        u-form frame and map the displacement back. ``u_old(z_q, 0, t)`` is
        evaluated by an extra forward rather than inferred from the MeanFlow
        identity, which a real network does not satisfy exactly.
        """
        net_dtype = next(self.generator.parameters()).dtype
        device = z_q.device

        r_draw = self._opsd_draw_r(
            t_raw,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            inner_epoch_idx=int(self._rl_epoch),
            device=device,
        )
        # Persist so the fitting phase queries the same (t, r) pair.
        self._opsd_stash_r(chunk_start, chunk_end, r_draw)

        u_old = self._compute_u(
            self.old_model, z_q, t_raw, prompt_embeds, pooled_embeds, r_draw,
            with_grad=False,
        ).detach().float()

        # Same central difference the fitting phase lifts with; computed here
        # because the target builder runs outside the per-step hook scope.
        self._pre_compute_v_hook(
            x_t=z_q.to(net_dtype), t_for_model=t_raw, r_for_model=r_draw,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
            x_0=x_0.to(net_dtype),
        )
        try:
            v_inst = self._instantaneous_velocity_from_u(
                u_old, t_raw, r_draw, self._shared_dF_dt,
            )
        finally:
            self._post_compute_v_hook()

        if v_inst.requires_grad:
            raise RuntimeError(
                "_opsd_build_anchor must stay stop-grad: anchor_v is the centre the "
                "reward targets are constructed around."
            )
        anchor_v = z_q.float() - self._opsd_sigma_scale_build(sigma_q, v_inst) * v_inst

        # The flow map's jump to the clean end (r = 0), i.e. where the few-step
        # sampler lands, so grad R is taken on an image the model can produce.
        u_r0 = self._compute_u(
            self.old_model, z_q, t_raw, prompt_embeds, pooled_embeds,
            torch.zeros_like(t_raw), with_grad=False,
        ).detach().float()
        anchor_u = z_q.float() - self._opsd_sigma_scale_build(sigma_q, u_r0) * u_r0

        shift = (anchor_u - anchor_v).detach()
        if shift.requires_grad:
            raise RuntimeError(
                "_opsd_build_anchor's frame shift must be stop-grad: it is a "
                "frozen reparametrisation of the target space, not a trainable "
                "quantity."
            )
        self._opsd_frame_shift = shift
        self._opsd_shift_stat = float(shift.abs().mean())
        return anchor_v

    def _opsd_clean_output(
        self,
        z_q_f: torch.Tensor,
        sigma_scale: Any,
        u_theta_f: torch.Tensor,
        *,
        t_for_model: torch.Tensor,
        r_for_model: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """v-form prediction ``x_0_pred_v = z_q - sigma_q * (u + (t - r) * sg(dF/dt))``.

        Uses the same lift as the anchors so prediction and references share one
        convention. ``dF_dt`` is detached, so the correction is a constant
        offset. With ``dF_dt is None`` (only legal when ``r == t``) this reduces
        to the u-form.
        """
        v_inst = self._instantaneous_velocity_from_u(
            u_theta_f, t_for_model, r_for_model, dF_dt,
        )
        if v_inst.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"the lift must stay in fp32/fp64, got {v_inst.dtype}: folding "
                "S in the network's bf16 is what destroys the cancellation "
                "this family depends on."
            )
        return z_q_f - sigma_scale * v_inst

    def _opsd_collect_r(self, n: int) -> Optional[torch.Tensor]:
        """Assemble the stashed per-chunk ``r`` into one ``[n]`` tensor."""
        buf = getattr(self, "_opsd_r_chunks", None)
        if not buf:
            return None
        out = torch.zeros(n, dtype=torch.float32)
        covered = 0
        for (s, e), v in buf.items():
            if e <= n and v.shape[0] == e - s:
                out[s:e] = v
                covered += e - s
        self._opsd_r_chunks = {}
        return out if covered == n else None

    def _opsd_draw_r(
        self,
        t_raw: torch.Tensor,
        *,
        chunk_start: int,
        chunk_end: int,
        inner_epoch_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Draw ``r ~ p(r | t)``, or replay the ``r`` the anchor was built at.

        In the sampling phase no replay buffer is set and ``r`` is drawn; in the
        fitting loop ``_opsd_r_replay`` holds the anchor's ``r``, so the anchor
        and the live prediction are queried at the same ``(t, r)``.
        ``opsd_r_mode="zero"`` always returns ``r = 0``.
        """
        if str(self._rl_cfg().opsd_r_mode) == "zero":
            return torch.zeros_like(t_raw)

        replay = getattr(self, "_opsd_r_replay", None)
        if replay is not None and replay.shape[0] >= chunk_end:
            return replay[chunk_start:chunk_end].to(
                device=device, dtype=torch.float32,
            )

        dtype = next(self.generator.parameters()).dtype
        return self._draw_training_s_given_t(
            t_raw,
            state={},          # no per-inner-epoch draw state
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            j_idx=0,           # one fitting query per sample, so no j axis
            inner_epoch_idx=inner_epoch_idx,
            device=device,
            dtype=dtype,
        )

    def _opsd_pack_sigma(
        self, sample_pack: dict[str, Any], sigma_q: torch.Tensor,
    ) -> None:
        """Store ``sigma_q`` in the pack as a ``[B]`` CPU tensor."""
        sample_pack["opsd_sigma_q"] = sigma_q.detach().cpu()

    def _opsd_permute_r(self, perm: torch.Tensor) -> None:
        """Keep the replayed ``r`` aligned with the shuffled rows."""
        r = getattr(self, "_opsd_r_replay", None)
        if r is not None:
            self._opsd_r_replay = r.to(perm.device)[perm]

    @torch.no_grad()
    def _opsd_query_state(
        self, x_0: torch.Tensor, sigma_q: torch.Tensor,
    ) -> torch.Tensor:
        """``z_q = (1 - s) * x_0 + s * eps`` with per-sample ``s``."""
        eps = self._draw_eps(x_0)
        s = self._bcast(sigma_q, x_0)
        return (1.0 - s) * x_0 + s * eps

    def _opsd_sigma_scale(
        self, sigma_q: Any, like: torch.Tensor,
    ) -> torch.Tensor:
        """The multiplier for ``z_q - sigma * u`` in the loss."""
        return self._bcast(sigma_q, like)

    @staticmethod
    def _opsd_sigma_scale_build(sigma_q: Any, like: torch.Tensor) -> Any:
        """``sigma_q`` (float or ``[B]``) as a multiplier inside the target builder."""
        if torch.is_tensor(sigma_q):
            return sigma_q.to(device=like.device, dtype=like.dtype).view(
                -1, *([1] * (like.ndim - 1))
            )
        return float(sigma_q)

    def _opsd_stash_r(self, start: int, end: int, r: torch.Tensor) -> None:
        """Record the ``r`` used to build this chunk's anchor."""
        buf = getattr(self, "_opsd_r_chunks", None)
        if buf is None:
            buf = {}
            self._opsd_r_chunks = buf
        buf[(int(start), int(end))] = r.detach().float().cpu()

    def _opsd_t_raw(
        self, sigma_q: Any, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Raw timesteps ``[B]`` for the model, from a ``[B]`` sigma."""
        T = float(self.scheduler.config.num_train_timesteps)
        return (
            sigma_q.to(device=device, dtype=torch.float32) * T
        ).to(torch.float32)

    def _opsd_tr_step(
        self,
        x_anchor: torch.Tensor,
        prompts: list[str],
        direction: float,
        *,
        x_end: Optional[torch.Tensor] = None,
        first_grad: Optional[torch.Tensor] = None,   # noqa: ARG002 - see below
    ) -> torch.Tensor:
        """Run the trust-region walk in the u-form frame (``x_anchor + shift``).

        Budget, projection centre and every ``grad R`` use ``anchor_u``; the
        result is shifted back, so the caller gets ``anchor_v + Delta`` (the
        displacement is kept, the target sits around the loss's own anchor).
        ``first_grad`` was computed at ``anchor_v`` and is therefore discarded,
        costing one extra reward backward per chunk. Without a stashed shift
        this falls through to the unshifted walk.
        """
        shift = getattr(self, "_opsd_frame_shift", None)
        if shift is None:
            return self._opsd_tr_step_base(
                x_anchor, prompts, direction, x_end=x_end, first_grad=first_grad,
            )
        if shift.shape != x_anchor.shape:
            raise RuntimeError(
                f"frame shift shape {tuple(shift.shape)} does not match the "
                f"anchor's {tuple(x_anchor.shape)}: the chunk loop and the anchor "
                "builder have gone out of step."
            )

        sh = shift.to(device=x_anchor.device, dtype=x_anchor.dtype)
        out = self._opsd_tr_step_base(
            x_anchor + sh,
            prompts,
            direction,
            # shifted too, so dir_mode="residual" keeps meaning the same vector
            x_end=None if x_end is None else x_end + sh,
            first_grad=None,
        )
        return (out - sh).detach()

    def _opsd_tr_step_base(
        self,
        x_anchor: torch.Tensor,
        prompts: list[str],
        direction: float,
        *,
        x_end: Optional[torch.Tensor] = None,
        first_grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One unshifted trust-region target: ``xp`` (dir=+1) or ``xm`` (dir=-1).

        ``M_tgt`` steps of length ``h = eta * rho * ||x_anchor|| / M_tgt`` along
        the unit-normalised reward gradient, then projection of the displacement
        into the ball of radius ``rho * ||x_anchor||`` (inactive for
        ``eta <= 1``; kept as a guard). ``first_grad`` is the gradient at
        ``x_anchor``, shareable between both directions. The result is detached,
        so the fitting phase never back-propagates through the reward.
        """
        cfg = self._rl_cfg()
        rho = float(cfg.opsd_rho)
        n_ascent = int(cfg.opsd_n_ascent)
        eta = float(cfg.opsd_eta_tgt)
        eps_g = float(cfg.opsd_eps_grad)
        dir_mode = str(cfg.opsd_dir_mode)

        x_start = x_anchor.detach().float()
        if dir_mode == "noop" or n_ascent == 0:
            # No displacement: xp == xm == x_anchor (reward-free control).
            return x_start

        budget = (rho * x_start.flatten(1).norm(dim=1)).view(-1, *([1] * (x_start.ndim - 1)))
        step_len = eta * budget / max(n_ascent, 1)

        fixed_dir = None
        if dir_mode == "rand":
            fixed_dir = torch.randn_like(x_start)
        elif dir_mode == "residual":
            if x_end is None:
                raise ValueError(
                    "opsd_dir_mode='residual' needs the rollout endpoint x_0."
                )
            fixed_dir = x_end.detach().float() - x_start

        x = x_start.clone()
        for i in range(n_ascent):
            if dir_mode == "grad":
                if i == 0 and first_grad is not None:
                    g = first_grad
                else:
                    _, g = self._reward_grad_chunked(x.detach(), prompts)
            else:
                g = fixed_dir
            gn = g.flatten(1).norm(dim=1).view(-1, *([1] * (x_start.ndim - 1))) + eps_g
            x = x.detach() + float(direction) * step_len * (g / gn)
            # Project the displacement (not the point) back into the ball.
            d = x - x_start
            dn = d.flatten(1).norm(dim=1).view(-1, *([1] * (x_start.ndim - 1)))
            x = (x_start + d * torch.clamp(budget / (dn + eps_g), max=1.0)).detach()
        return x.detach()

    def _opsd_unpack_sigma(
        self, sample_pack: dict[str, Any], device: torch.device,
    ) -> torch.Tensor:
        """Read ``sigma_q`` back as a ``[B]`` tensor on ``device``."""
        return sample_pack["opsd_sigma_q"].to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def _resolve_query(
        self, sample_pack: dict[str, Any], x_0: torch.Tensor, cfg: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a per-sample ``sigma_q`` and the query state ``z_q`` at it.

        Returns ``(sigma_q [B], z_q)``. ``opsd_sigma_mode`` is "fixed" (constant
        ``opsd_query_sigma``, forward-noised or taken off the rollout grid),
        "lognormal" (logit-normal draw) or the base sampler's ``t`` law.
        """
        if str(cfg.opsd_sigma_mode) == "fixed":
            sigma_star = float(cfg.opsd_query_sigma)
            mode = str(cfg.opsd_state_mode)

            if mode == "forward":
                sigma_f, z_q = sigma_star, self._opsd_query_state(x_0, sigma_star)
            else:
                # "rollout": take the state entering the step whose start-sigma
                # is nearest sigma*, straight off the recorded trajectory.
                x_t_all = sample_pack.get("x_t")
                start_ts = sample_pack.get("start_timesteps")
                if x_t_all is None or start_ts is None:
                    raise ValueError(
                        "opsd_state_mode='rollout' needs the per-step rollout tensors; "
                        "set fewstep_rl.reuse_x_t=true (the config validator normally "
                        "catches this)."
                    )
                T = float(self.scheduler.config.num_train_timesteps)
                # Row 0 is representative: every sample shares the inference grid.
                grid = start_ts[0].to(torch.float64) / T
                q = int(torch.argmin((grid - sigma_star).abs()).item())
                sigma_f = float(grid[q])
                z_q = x_t_all[:, q].to(device=x_0.device)
            n = x_0.shape[0]
            sigma_q = torch.full(
                (n,), float(sigma_f), device=x_0.device, dtype=torch.float32,
            )
            # Zero spread, recorded so the diagnostic keys exist on every mode.
            self._opsd_sigma_spread = (0.0, float(sigma_f), float(sigma_f))
            return sigma_q, z_q

        if str(cfg.opsd_state_mode) != "forward":
            # "rollout" snaps to an on-grid sigma, not the drawn one.
            raise ValueError(
                f"opsd_sigma_mode={str(cfg.opsd_sigma_mode)!r} requires "
                "opsd_state_mode='forward': a randomized sigma cannot be taken "
                "off the fixed rollout grid."
            )

        T = float(self.scheduler.config.num_train_timesteps)
        device = x_0.device
        dtype = next(self.generator.parameters()).dtype

        if str(cfg.opsd_sigma_mode) == "lognormal":
            # Logit-normal query sigma (one per trajectory), separate from
            # ``fewstep_rl.renoise_distribution``, which sets the training-t
            # law. Drawn from the global RNG, independently per rank; the
            # fitting phase reads the packed sigma_q, and r is replayed
            # separately.
            n = x_0.shape[0]
            logit_mean = float(cfg.opsd_logit_mean)
            logit_std = float(cfg.opsd_logit_std)

            # fp32 regardless of the network dtype: the anchors and lift key
            # off sigma_q.
            t01 = torch.sigmoid(
                logit_mean
                + logit_std * torch.randn(n, device=device, dtype=torch.float32)
            )
            sigma_q = self.flowmap_scheduler.apply_shift(t01).to(torch.float32)
            sigma_q = sigma_q.clamp(1.0 / T, 1.0 - 1.0 / T)

            self._opsd_sigma_spread = (
                float(sigma_q.std()) if n > 1 else 0.0,
                float(sigma_q.min()),
                float(sigma_q.max()),
            )
            return sigma_q, self._opsd_query_state(x_0, sigma_q)

        # The base sampler's (t, r) law; ``_rl_epoch`` keys the RNG so each
        # epoch draws a new t. r_raw is discarded (r comes from _opsd_draw_r).
        t_raw, _r_raw = self._sample_training_tr_three_mode(
            batch_size=x_0.shape[0],
            chunk_start=0,
            j_idx=0,
            inner_epoch_idx=int(self._rl_epoch),
            device=device,
            dtype=dtype,
        )
        sigma_q = (t_raw.to(torch.float32) / T).clamp(1.0 / T, 1.0 - 1.0 / T)
        self._opsd_sigma_spread = (
            float(sigma_q.std()) if sigma_q.numel() > 1 else 0.0,
            float(sigma_q.min()),
            float(sigma_q.max()),
        )
        return sigma_q, self._opsd_query_state(x_0, sigma_q)

    def _rl_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
        *,
        z_q: torch.Tensor,
        x_anchor_pre: torch.Tensor,   # noqa: ARG002 -- anchors are recomputed in-step
        xp_pre: torch.Tensor,
        xm_pre: torch.Tensor,
        sigma_q: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One ReFL fitting step at the query state (v-form).

        Same as the OPSD trainer's step except for the ``omega`` branch and the
        :meth:`refl_loss` call. Unlike MeanFlowNFT there is no forward noising
        (one query state per trajectory, built in the sampling phase) and the
        target is the reward-ascent point ``xp`` rather than ``x_0``
        (``xm_pre`` is passed through but unused).

        Expected logs: ``self_regression_loss`` and ``opsd_push_neg`` exactly
        0.0; ``s_x0_abs_mean`` 0 (``E = x_anchor_1 - x_anchor_2`` vanishes);
        ``opsd_frame_shift`` nonzero (0 means the shift was never measured);
        ``refl_lambda_eff`` is the detached effective ReFL ``lambda``.
        """
        cfg = self._rl_cfg()
        wdt = self._rl_work_dtype
        spatial = tuple(range(1, x_0.ndim))
        net_dtype = next(self.generator.parameters()).dtype

        beta = float(cfg.beta)
        adv_clip_max = float(cfg.adv_clip_max)
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

        _, use_v_kl = self._kl_usage()
        kl_weight = float(cfg.kl_weight)

        if self._shared_dF_dt is not None:
            raise RuntimeError(
                "stale _shared_dF_dt on entry to _rl_loss -- a previous step "
                "leaked its central difference (missing _post_compute_v_hook)."
            )

        x_t = z_q.to(net_dtype)
        t_for_model = t_raw.to(dtype=torch.float32)
        r_for_model = (
            r_raw.to(dtype=torch.float32) if r_raw is not None
            else torch.zeros_like(t_for_model)
        )
        t_broadcast = (
            t_for_model.to(net_dtype)
            / float(self.scheduler.config.num_train_timesteps)
        ).view(x_0.shape[0], *([1] * (x_0.ndim - 1)))

        # The central difference is a collective: it runs unconditionally and
        # is cleared in ``finally``.
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
            t_normed_b_f = t_broadcast.to(wdt)

            # Live v-form clean prediction z_q - sigma * (u + S); the only
            # tensor on the graph.
            x_0_pred_v = self._opsd_clean_output(
                x_t_f, t_normed_b_f, u_theta_f,
                t_for_model=t_for_model,
                r_for_model=r_for_model,
                dF_dt=self._shared_dF_dt,
            )

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
            shared_dF_dt = self._shared_dF_dt if use_v_kl else None
        finally:
            self._post_compute_v_hook()

        # Fitting weight omega. Official ReFL weights every sample equally
        # (omega = 1, the default); refl_use_advantage uses the OPSD weight
        # r_hat = clip(A, +-c_adv) rescaled to [0, 1]. Advantages are logged
        # either way.
        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        if bool(cfg.refl_use_advantage):
            omega = ((adv / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)
        else:
            omega = torch.ones_like(adv)

        unscaled, form_info = self.refl_loss(
            x_0_pred_v, x_anchor_1, x_anchor_2, xp_pre.to(wdt), xm_pre.to(wdt),
            omega, beta, spatial,
        )
        policy_loss = (unscaled * adv_clip_max).mean()
        loss = policy_loss

        with torch.no_grad():
            s_x0 = x_anchor_1 - x_anchor_2
            dev_abs = (x_0_pred_v.detach() - x_anchor_1).abs().mean()
            info = {
                "loss_policy": float(policy_loss.detach()),
                "loss_unscaled": float(unscaled.mean().detach()),
                # Mean of the per-sample query t.
                "t_mean": float(t_raw.float().mean().item()),
                "r_hat_mean": float(omega.mean().item()),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std().item()),
                "x0_norm_mean": float((x_0.to(wdt) ** 2).mean().item()),
                "old_deviate": float(((u_theta_f - u_old_f) ** 2).mean().item()),
                "s_x0_abs_mean": float(s_x0.abs().mean()),
                "s_rel_median": float(
                    (s_x0.abs().mean() / dev_abs.clamp(min=1e-12))
                ),
                # Trust-region displacement, carried from the sampling phase.
                "opsd_delta_res": float(
                    self._opsd_last_stats.get("opsd_delta_res", 0.0)
                ),
                "opsd_sigma_q": float(
                    sigma_q.mean() if torch.is_tensor(sigma_q) else sigma_q
                ),
                **form_info,
            }

        kl, kl_info = self.compute_kl_term(
            x_t, t_for_model, prompt_embeds, pooled_embeds, r_for_model,
            shared_dF_dt, u_theta_f, spatial,
        )
        loss = loss + kl_weight * kl
        info["loss_kl"] = float(kl.detach())
        info.update(kl_info)
        info["loss_total"] = float(loss.detach())

        # --- r diagnostics ---
        # Emitted on every rank (see _EXTRA_DIAGNOSTIC_KEYS).
        with torch.no_grad():
            if r_raw is None:
                info["opsd_r_mean"] = 0.0
                info["opsd_r_frac_zero"] = 1.0
            else:
                r_f = r_raw.float()
                info["opsd_r_mean"] = float(r_f.mean())
                info["opsd_r_frac_zero"] = float((r_f == 0).float().mean())

        # --- realised frame shift (0 before the first anchor build) ---
        info["opsd_frame_shift"] = float(getattr(self, "_opsd_shift_stat", 0.0))

        # --- realised query-sigma spread, emitted on every mode ---
        _std, _lo, _hi = getattr(self, "_opsd_sigma_spread", (0.0, 0.0, 0.0))
        info["opsd_sigma_q_std"] = _std
        info["opsd_sigma_q_min"] = _lo
        info["opsd_sigma_q_max"] = _hi
        return loss, info

    def _rl_sampling_phase(
        self, return_rewards: bool = True, return_intermediate: bool = False,
    ) -> tuple[dict[str, Any], Optional[tuple[torch.Tensor, list[str]]]]:
        """Base rollout and advantages, then the OPSD target stage.

        Adds ``z_q``, the anchor, the reward targets, ``sigma_q`` and the ``r``
        each anchor was built at (replayed by :meth:`_opsd_draw_r` during
        fitting) to the sample pack. Skipped when ``return_rewards=False`` or
        OPSD is disabled; both are rank-uniform, so the collectives inside the
        target stage cannot desync.
        """
        sample_pack, preview = super()._rl_sampling_phase(
            return_rewards=return_rewards,
            return_intermediate=return_intermediate,
        )

        cfg = self._rl_cfg()
        if not bool(cfg.opsd_enabled) or not return_rewards:
            return sample_pack, preview

        device = next(self.generator.parameters()).device
        x_0 = sample_pack["x_0"].to(device)
        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        prompts = sample_pack["prompts"]

        sigma_q, z_q = self._resolve_query(sample_pack, x_0, cfg)

        anchor_out, xp_out, xm_out, stats = self._build_opsd_targets(
            x_0=x_0,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            prompts=prompts,
            z_q=z_q,
            sigma_q=sigma_q,
        )

        # The pack holds CPU tensors; _rl_train_phase moves them per epoch.
        sample_pack["opsd_z_q"] = z_q.detach().cpu()
        sample_pack["opsd_y_0"] = anchor_out.detach().cpu()
        sample_pack["opsd_y_plus"] = xp_out.detach().cpu()
        sample_pack["opsd_y_minus"] = xm_out.detach().cpu()
        self._opsd_pack_sigma(sample_pack, sigma_q)
        self._opsd_last_stats = stats

        # --- stash r for replay in the fitting phase ---
        if sample_pack.get("opsd_y_0") is not None:
            r_all = self._opsd_collect_r(sample_pack["opsd_y_0"].shape[0])
            if r_all is not None:
                sample_pack["opsd_r"] = r_all
        return sample_pack, preview

    def _rl_train_phase(self, sample_pack: dict[str, Any]) -> dict[str, float]:
        """Fitting phase; the stashed ``r`` is published for replay and cleared
        in ``finally``."""
        self._opsd_r_replay = sample_pack.get("opsd_r")
        try:
            cfg = self._rl_cfg()
            device = next(self.generator.parameters()).device

            if not bool(cfg.opsd_enabled):
                raise ValueError(
                    f"{type(self).__name__} requires fewstep_rl.opsd_enabled=true. "
                    "With it off no reward-gradient targets are built and this trial "
                    "has nothing to fit; run --trainer meanflow_nft instead."
                )
            missing = [k for k in self._OPSD_KEYS if sample_pack.get(k) is None]
            if missing:
                # Config-level, hence rank-uniform: raising cannot desync.
                raise ValueError(
                    f"sample_pack is missing {missing}; the OPSD target stage did "
                    "not run. It is skipped when return_rewards=False."
                )

            x_0 = sample_pack["x_0"].to(device)
            prompt_embeds = sample_pack["prompt_embeds"].to(device)
            pooled_embeds = sample_pack["pooled_embeds"].to(device)
            advantages = sample_pack["advantages"].to(device)
            z_q_all = sample_pack["opsd_z_q"].to(device)
            anchor_all = sample_pack["opsd_y_0"].to(device)
            xp_all = sample_pack["opsd_y_plus"].to(device)
            xm_all = sample_pack["opsd_y_minus"].to(device)
            sigma_q = self._opsd_unpack_sigma(sample_pack, device)

            n_local = x_0.shape[0]
            if n_local == 0:
                return {}
            for name, tensor in (
                ("opsd_z_q", z_q_all), ("opsd_y_0", anchor_all),
                ("opsd_y_plus", xp_all), ("opsd_y_minus", xm_all),
            ):
                if tensor.shape[0] != n_local:
                    raise ValueError(
                        f"sample_pack['{name}'] has leading dim {tensor.shape[0]}, "
                        f"expected {n_local}. It is indexed by the same row "
                        "permutation as x_0, so a mismatch would pair a query "
                        "state with another sample's target."
                    )

            inner_bs = int(cfg.rl_inner_batch_size) if cfg.rl_inner_batch_size > 0 else n_local
            inner_bs = max(1, min(inner_bs, n_local))
            n_inner_epochs = max(1, int(cfg.num_inner_epochs))
            max_grad_norm = float(self.config.solver.generator.max_grad_norm)
            grad_accum = max(1, int(self.config.train.gradient_accumulation_steps))
            # One fitting query (one backward) per sample per inner epoch.
            effective_grad_accum_steps = grad_accum

            # A per-sample sigma_q must be permuted with the other per-sample
            # tensors; checked dynamically in case a scalar is handed back.
            sigma_is_per_sample = torch.is_tensor(sigma_q) and sigma_q.ndim == 1
            if sigma_is_per_sample and sigma_q.shape[0] != n_local:
                raise ValueError(
                    f"opsd_sigma_q has leading dim {sigma_q.shape[0]}, expected "
                    f"{n_local}: it is indexed by the same row permutation as x_0."
                )

            self.generator.train()
            accumulator: dict[str, float] = {}
            n_backward = 0
            n_opt_steps = 0
            current_accumulated_steps = 0
            self.optimizers["generator"].zero_grad(set_to_none=True)
            last_grad_norm: float = 0.0

            for inner_epoch_idx in range(n_inner_epochs):
                perm = torch.randperm(n_local, device=device)
                sx0 = x_0[perm]
                spe = prompt_embeds[perm]
                spp = pooled_embeds[perm]
                sadv = advantages[perm]
                szq = z_q_all[perm]
                s_anchor = anchor_all[perm]
                sxp = xp_all[perm]
                sxm = xm_all[perm]
                ssig = sigma_q[perm] if sigma_is_per_sample else sigma_q
                # The replayed per-sample r must follow the same permutation.
                self._opsd_permute_r(perm)

                for start in range(0, n_local, inner_bs):
                    end = min(start + inner_bs, n_local)
                    bsz = end - start
                    sig_b = ssig[start:end] if sigma_is_per_sample else ssig
                    t_b = self._opsd_t_raw(sig_b, bsz, device)
                    # Replays the r each anchor was built at, so the live
                    # prediction is queried at the same (t, r); r = 0 only
                    # under opsd_r_mode=zero.
                    r_b = self._opsd_draw_r(
                        t_b,
                        chunk_start=start,
                        chunk_end=end,
                        inner_epoch_idx=inner_epoch_idx,
                        device=device,
                    )

                    loss, info = self._rl_loss(
                        x_0=sx0[start:end],
                        prompt_embeds=spe[start:end],
                        pooled_embeds=spp[start:end],
                        advantages=sadv[start:end],
                        t_raw=t_b,
                        r_raw=r_b,
                        z_q=szq[start:end],
                        x_anchor_pre=s_anchor[start:end],
                        xp_pre=sxp[start:end],
                        xm_pre=sxm[start:end],
                        sigma_q=sig_b,
                    )
                    (loss / float(effective_grad_accum_steps)).backward()
                    current_accumulated_steps += 1
                    n_backward += 1

                    for k, v in info.items():
                        accumulator[k] = accumulator.get(k, 0.0) + float(v)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        last_grad_norm = self._compute_grad_norm(self.generator)
                        if max_grad_norm > 0:
                            self._clip_grad_norm(self.generator, max_grad_norm)
                        self.optimizers["generator"].step()
                        self.schedulers["generator"].step()
                        self.optimizers["generator"].zero_grad(set_to_none=True)

                        if (
                            self.generator_ema is not None
                            and self.global_step % self.config.train.ema_every == 0
                        ):
                            self.ema_update(
                                self.generator, self.generator_ema,
                                self._current_ema_decay(),
                            )

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
            # Backwards per sample per inner epoch.
            averaged["num_train_timesteps"] = 1.0
            averaged["effective_grad_accum_steps"] = float(effective_grad_accum_steps)
            return averaged
        finally:
            self._opsd_r_replay = None

    def _s_term(
        self,
        t_cd: torch.Tensor,
        r_cd: torch.Tensor,
        dF_dt: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``S = (t - r) * dF/dt`` in ``like``'s dtype, or None when S is exactly 0."""
        if dF_dt is None:
            # With ``disable_cd`` the central difference is skipped on purpose
            # while r != t, and S is defined to be zero (checked before the
            # r == t guard; config-level, hence rank-uniform). This changes the
            # objective slightly: S cancels in the differentiable channel, but
            # the stop-grad coefficients are built from anchor positions that
            # shift by -sigma*S. See FewstepRLTrainConfig.disable_cd.
            if bool(getattr(self._rl_cfg(), "disable_cd", False)):
                return None
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
        dt_cd = (t_cd - r_cd).view(
            like.shape[0], *([1] * (like.ndim - 1))
        ).to(wdt)
        return dt_cd * dF_dt.to(wdt)

    def refl_loss(
        self,
        x_0_pred_v: torch.Tensor,  # fp32, the only tensor carrying grad
        x_anchor_1: torch.Tensor,  # fp32 detached: lifted anchor (weights only)
        x_anchor_2: torch.Tensor,  # fp32 detached: probe centre, where grad R was read
        xp: torch.Tensor,          # fp32 detached: reward-ascent target
        xm: torch.Tensor,          # fp32 detached: reward-descent target (unused)
        omega: torch.Tensor,       # [B] in [0,1]; == 1 unless refl_use_advantage
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """ReFL as the one-term corner of the decoupled OPSD loss.

        Of the OPSD terms (proximal self-regression, attraction to ``xp``,
        repulsion from ``xm``) only the attraction is kept. With
        ``opsd_n_ascent=1``, ``xp - x_anchor_2 = h * ghat`` and

            dL/dx_0_pred_v = -(2*W*omega_p*h/N) * ghat,

        ReFL's gradient with a detached positive ``lambda`` and ``grad R`` read
        at the behaviour anchor. Argument order matters: ``reward_augmentation``
        pushes toward its first target argument, and swapping ``xp`` and
        ``x_anchor_2`` would leave the logged loss unchanged while training away
        from the reward. ``x_anchor_1`` only enters the weights. The loss value
        ``W * mean[norm_grad**2]`` is theta-independent; do not monitor it.
        """
        with torch.no_grad():
            # gamma_- and omega_m are computed by the shared kernel and discarded.
            W, omega_p, _omega_m = self.opsd_weights(
                x_0_pred_v, x_anchor_1, x_anchor_2, xp, xm, omega, beta, spatial,
            )

        # Reward attraction toward xp, the only force.
        l_pos = self.reward_augmentation(x_0_pred_v, xp, x_anchor_2, omega_p, W, spatial)

        # Effective detached lambda: W * omega_p == a_+ * beta multiplies the
        # reward step.
        lam_eff = float((W * omega_p.reshape(-1)).mean().item())

        return l_pos, {
            # Key names shared with the OPSD trainer.
            "affine_coeff_mean": float(W.mean().item()),
            "affine_adv_coeff_mean": float(omega_p.mean().item()),
            # Dropped terms, logged as 0.0 (see _EXTRA_LOSS_KEYS).
            "self_regression_loss": 0.0,
            "opsd_push_neg": 0.0,
            "opsd_omega_neg_mean": 0.0,
            # With a single reward term these two keys coincide.
            "reward_augmentation_loss": float(l_pos.detach().mean()),
            "opsd_pull_pos": float(l_pos.detach().mean()),
            "opsd_omega_pos_mean": float(omega_p.mean().item()),
            "refl_lambda_eff": lam_eff,
        }

    @torch.no_grad()
    def opsd_weights(
        self,
        x_0_pred_v: torch.Tensor,
        x_anchor_1: torch.Tensor,
        x_anchor_2: torch.Tensor,
        xp: torch.Tensor,
        xm: torch.Tensor,
        omega: torch.Tensor,
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frozen per-sample scalars ``(W, omega_p, omega_m)``.

        With fitting weight ``omega`` in [0, 1] and
        ``probe_pm = x_anchor_2 pm beta * (x_0_pred_v - x_anchor_1)``::

            gamma_+ = max(mean|probe_pos - xp|, eps_gamma)
            gamma_- = max(mean|probe_neg - xm|, eps_gamma)
            a_+ = omega / gamma_+          a_- = (1 - omega) / gamma_-
            W = beta**2 * (a_+ + a_-)      W * omega_pm == a_pm * beta

        ``beta**2`` (not ``beta``): one factor comes from differentiating the
        probe branch, one from the outer MSE coefficient. Computed in fp64 since
        the ``gamma`` reciprocals can be large.
        """
        eps_gamma = float(self._rl_cfg().opsd_eps_gamma)
        bc = (-1, *([1] * (x_0_pred_v.ndim - 1)))

        # Probe direction from x_anchor_1, centre at x_anchor_2.
        dev = x_0_pred_v.detach() - x_anchor_1
        probe_pos = x_anchor_2 + beta * dev
        probe_neg = x_anchor_2 - beta * dev

        gamma_p = (
            (probe_pos.double() - xp.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=eps_gamma)
        )
        gamma_m = (
            (probe_neg.double() - xm.double())
            .abs().mean(dim=spatial, keepdim=True).clip(min=eps_gamma)
        )

        om_b = omega.view(bc).double()
        a_p = om_b / gamma_p
        a_m = (1.0 - om_b) / gamma_m
        if not bool(self._rl_cfg().opsd_dual_neg):
            # Positive-branch-only ablation: a_- = 0.
            a_m = torch.zeros_like(a_m)
        ab = a_p + a_m

        W = (beta * beta * ab).reshape(-1)          # beta**2, see docstring
        omega_p = a_p / (beta * ab)                 # W * omega_p == a_p * beta
        omega_m = a_m / (beta * ab)                 # W * omega_m == a_m * beta
        return W, omega_p, omega_m
