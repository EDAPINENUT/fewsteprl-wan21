"""MeanFlow-OPSD (v-form): DiffusionOPSD-style reward-gradient targets on a flow map.

:class:`MeanFlowOPSDTrainer` (key ``meanflowopsd_rl``) extends the RL base with an
OPSD target stage and fitting loss:

* Sampling phase: the base rollout and advantages, then a per-sample query level
  ``sigma_q`` (``t = sigma_q * T``) and ``r ~ r | t``, the query state
  ``z_q = (1 - sigma_q) * x_0 + sigma_q * eps``, the v-form anchor
  ``anchor_v = z_q - sigma_q * v_old`` and two trust-region targets ``xp`` / ``xm``
  from normalized reward-gradient ascent / descent. The walk runs around the
  flow map's own clean-end jump ``anchor_u = z_q - sigma_q * u_old(z_q, r=0, t)``
  and its displacement is mapped back onto ``anchor_v``.
* Fitting phase: at the same ``(t, r)``, the live prediction
  ``x_0_pred_v = z_q - sigma_q * (u_theta + (t - r) * sg(dF/dt))`` is trained with
  a proximal term toward the anchor plus attraction to ``xp`` and repulsion from
  ``xm``, weighted by the advantage-derived ``omega`` in [0, 1].

With ``fewstep_rl.disable_cd`` the central difference is skipped and
``S = (t - r) * dF/dt`` is defined to be zero, so the prediction and anchors fall
back to the u-form while ``r`` stays random.

The differentiable video-reward gradient (``grad R``) comes from
:class:`~fewsteprl.trainers.opsd_grad_mixin.OPSDGradMixin`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from fewsteprl.trainers.opsd_grad_mixin import OPSDGradMixin
from fewsteprl.trainers.rl_general_base import AnchorInputs, GeneralRLTrainer

_DIFFERENTIABLE_REWARDS = frozenset({'aesthetic', 'clipscore', 'hpsv2', 'pickscore'})


class MeanFlowOPSDTrainer(OPSDGradMixin, GeneralRLTrainer):
    """MeanFlow-OPSD trainer (v-form, random ``t`` and ``r``)."""

    TRAINER_KEY = "meanflowopsd_rl"

    #: Must be identical on every rank: _reduce_metrics all-reduces per key and
    #: hangs on a divergent key set, so every key is emitted on every sigma mode.
    _EXTRA_DIAGNOSTIC_KEYS: tuple[str, ...] = ('s_x0_abs_mean', 's_rel_median', 'opsd_delta_res', 'opsd_sigma_q', 'opsd_r_mean', 'opsd_r_frac_zero', 'opsd_frame_shift', 'opsd_sigma_q_std', 'opsd_sigma_q_min', 'opsd_sigma_q_max')

    _EXTRA_LOSS_KEYS: tuple[str, ...] = ('self_regression_loss', 'reward_augmentation_loss', 'loss_kl_unweighted', 'kl_t_weight_mean', 'kl_rebalance_mean', 'opsd_pull_pos', 'opsd_push_neg', 'opsd_omega_pos_mean', 'opsd_omega_neg_mean')

    #: Per-sample tensors threaded from the sampling phase to the fitting phase.
    _OPSD_KEYS: tuple[str, ...] = ('opsd_z_q', 'opsd_y_0', 'opsd_y_plus', 'opsd_y_minus')

    def __init__(self, config) -> None:
        super().__init__(config)
        # Differentiable scorer, separate from the base's ``_reward_scorer``.
        self._opsd_grad_scorer: Optional[Any] = None
        # Populated by :meth:`_rl_sampling_phase`, read by :meth:`_rl_train_phase`.
        self._opsd_last_stats: dict[str, float] = {}

    @staticmethod
    def _bcast(sigma_q: Any, like: torch.Tensor) -> torch.Tensor:
        """Broadcast a ``[B]`` (or scalar) noise level to ``[B, 1, ...]`` over ``like``.

        A bare ``[B]`` tensor would silently right-align onto the last axis.
        Python floats and 0-d tensors are promoted to ``[B]`` first.
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
        """Build the anchor and the ``xp`` / ``xm`` targets for a ``[B]`` ``sigma_q``.

        Returns ``(anchor, xp, xm, stats)``, all fp32 and detached. The chunk loop
        count is a config constant times a rank-uniform sample count, so every
        rank runs the same number of forwards (collective safety).
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

            # The anchor must be in the same (v-form) coordinate as the loss's
            # prediction; _opsd_build_anchor owns that choice.
            x_anchor = self._opsd_build_anchor(
                zq_c, t_c, r_c, pe_c, pp_c, s_c, start, end, x0_c,
            )

            g0 = None
            if need_grad:
                # ``_reward_grad_chunked`` returns the gradient itself rather
                # than a graph: one graph over the full Wan VAE decode does not
                # fit in host memory.
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
            # Mean of the per-sample levels.
            "opsd_sigma_q": float(sigma_q.mean()),
        }

    @torch.no_grad()
    def _compute_x_anchor_1(self, inputs: AnchorInputs) -> torch.Tensor:
        """``x_anchor_1 = z_q - sigma_q * v_old``: the lifted (v-form) proximal anchor.

        It must be in the same coordinate as ``x_0_pred_v``; then
        ``x_0_pred_v - x_anchor_1 = -sigma_q * (u_theta - u_old)`` exactly, a pure
        policy deviation that vanishes at initialisation (a u-form anchor would
        leave a constant ``-sigma_q * S`` residual).
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
        """Return the v-form anchor ``anchor_v = z_q - sigma_q * v_old`` and stash the shift.

        Draws ``r`` (stashed so the fitting phase replays the same ``(t, r)``),
        lifts ``u_old(z_q, r, t)`` to ``v_old`` with the central difference, and
        records ``shift = anchor_u - anchor_v`` with
        ``anchor_u = z_q - sigma_q * u_old(z_q, r=0, t)``, which
        :meth:`_opsd_tr_step` uses to run the target walk around ``anchor_u``.
        ``u_old(z_q, r=0, t)`` is evaluated with an extra forward rather than
        inferred from the MeanFlow identity, which a real network only
        approximately satisfies. ``r_raw`` is unused.
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
        self._opsd_stash_r(chunk_start, chunk_end, r_draw)

        u_old = self._compute_u(
            self.old_model, z_q, t_raw, prompt_embeds, pooled_embeds, r_draw,
            with_grad=False,
        ).detach().float()

        # The same central difference the fitting phase lifts with; the hook is
        # normally scoped to one fitting step, so it is opened explicitly here.
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

        # The flow map's own jump to the clean end (r = 0), i.e. the kind of point
        # the few-step sampler lands on, so grad R is taken at a sample the
        # model can actually produce.
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

        Uses the same lift as the anchors so the S convention cannot drift between
        them. ``dF_dt`` is detached, so S is a constant offset. Degenerates to the
        u-form when ``dF_dt is None`` (``r == t`` everywhere, or ``disable_cd``).
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
        """Draw ``r ~ r | t`` (sampling phase) or replay the anchor's ``r`` (fitting).

        The replay (``_opsd_r_replay``) makes the anchor and the live prediction
        share the identical ``(t, r)``; otherwise they would differ by the model's
        MeanFlow-identity residual, which grows with S. ``opsd_r_mode="zero"``
        always returns ``r = 0``.
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
        """Forward-noise ``z_q = (1 - s) * x_0 + s * eps`` with per-sample ``s = sigma_q``."""
        eps = self._draw_eps(x_0)
        s = self._bcast(sigma_q, x_0)
        return (1.0 - s) * x_0 + s * eps

    def _opsd_sigma_scale(
        self, sigma_q: Any, like: torch.Tensor,
    ) -> torch.Tensor:
        """``sigma_q`` broadcast as the multiplier in ``z_q - sigma * v`` in the loss.

        Must equal the fp32 value the target builder used, so the loss's probe
        centre matches the centre the targets were built around.
        """
        return self._bcast(sigma_q, like)

    @staticmethod
    def _opsd_sigma_scale_build(sigma_q: Any, like: torch.Tensor) -> Any:
        """``sigma_q`` as a multiplier inside the target builder (tensor or float)."""
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
        """Run the trust-region walk around ``anchor_u = x_anchor + shift``.

        The budget, the ball projection and every ``grad R`` are taken in the
        u-form frame; the result is mapped back by ``- shift``, so the caller gets
        ``anchor_v + Delta``: the displacement is kept and the target sits around
        the loss's own anchor (returning the absolute u-form point would
        reintroduce a ``sigma * S`` offset).

        ``first_grad`` (computed at ``anchor_v``) is discarded and each direction
        recomputes it at ``anchor_u``, costing one extra reward backward per chunk.
        Without a stashed shift this is the plain walk.
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
        """One trust-region target: ``xp`` (direction=+1) or ``xm`` (direction=-1).

        ``M = opsd_n_ascent`` steps of length ``eta * rho * ||x_anchor|| / M`` along
        the unit-normalized reward gradient, each followed by projecting the
        displacement into the ball of radius ``rho * ||x_anchor||``.
        ``first_grad``, if given, replaces the first gradient evaluation.
        Returns a detached tensor: the fitting phase never backprops through
        the reward.
        """
        cfg = self._rl_cfg()
        rho = float(cfg.opsd_rho)
        n_ascent = int(cfg.opsd_n_ascent)
        eta = float(cfg.opsd_eta_tgt)
        eps_g = float(cfg.opsd_eps_grad)
        dir_mode = str(cfg.opsd_dir_mode)

        x_start = x_anchor.detach().float()
        if dir_mode == "noop" or n_ascent == 0:
            # xp == xm == x_anchor: the reward forces cancel and only the
            # proximal term remains (reward-free control).
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
                    # Returns the gradient directly, not a graph.
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
        """Choose the per-sample query level ``sigma_q`` and the query state ``z_q``.

        Returns ``(sigma_q [B] fp32, z_q)``. ``opsd_sigma_mode``: ``"fixed"`` uses
        ``opsd_query_sigma`` (forward-noised, or snapped to the rollout grid with
        ``opsd_state_mode="rollout"``); ``"lognormal"`` draws a shifted logit-normal
        level; otherwise ``t`` follows the trainer's training-time sampler.
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
                # Every sample shares the inference grid, so row 0 suffices.
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
            # "rollout" snaps to an on-grid sigma, which would not be the drawn one.
            raise ValueError(
                f"opsd_sigma_mode={str(cfg.opsd_sigma_mode)!r} requires "
                "opsd_state_mode='forward': a randomized sigma cannot be taken "
                "off the fixed rollout grid."
            )

        T = float(self.scheduler.config.num_train_timesteps)
        device = x_0.device
        dtype = next(self.generator.parameters()).dtype

        if str(cfg.opsd_sigma_mode) == "lognormal":
            n = x_0.shape[0]
            logit_mean = float(cfg.opsd_logit_mean)
            logit_std = float(cfg.opsd_logit_std)

            # fp32 regardless of the network dtype: the anchors and the lift
            # all key off sigma_q.
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

        # The trainer's own training-time t law. ``_rl_epoch`` keys the RNG so
        # each epoch draws a fresh t; r is drawn later by _opsd_draw_r.
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
        x_anchor_pre: torch.Tensor,   # noqa: ARG002 -- anchors are recomputed here
        xp_pre: torch.Tensor,
        xm_pre: torch.Tensor,
        sigma_q: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One OPSD fitting step at the stored query state ``z_q`` (v-form).

        Unlike MeanFlowNFT there is no fresh forward noising: each trajectory is
        fitted at the single ``(z_q, t, r)`` built in the sampling phase, against
        the ``(xp, xm)`` target pair. ``opsd_frame_shift`` logging 0 means the
        u-frame shift was never measured.
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
        # Use the packed fp32 sigma, not t_for_model.to(net_dtype) / T: rounding
        # through bf16 would put x_anchor_2 off the centre the targets were built
        # around (anchor_v).
        t_broadcast = self._opsd_sigma_scale(sigma_q, x_0.to(wdt))

        # The central difference is a collective: it runs unconditionally and is
        # cleared in ``finally``.
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

            # Live prediction z_q - sigma * (u_theta + S); the only tensor on the graph.
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

        # Fitting weight omega in [0, 1], same as MeanFlowNFT's r_hat: clip the
        # advantage to +-adv_clip_max and rescale. adv_clip_max also scales the loss.
        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        omega = ((adv / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)

        unscaled, form_info = self.decoupled_opsd_loss(
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
                # Trust-region displacement from the sampling phase.
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

        # The diagnostics below are emitted unconditionally on every rank (see
        # _EXTRA_DIAGNOSTIC_KEYS).
        with torch.no_grad():
            if r_raw is None:
                info["opsd_r_mean"] = 0.0
                info["opsd_r_frac_zero"] = 1.0
            else:
                r_f = r_raw.float()
                info["opsd_r_mean"] = float(r_f.mean())
                info["opsd_r_frac_zero"] = float((r_f == 0).float().mean())

        # Zero before the first anchor build.
        info["opsd_frame_shift"] = float(getattr(self, "_opsd_shift_stat", 0.0))

        # Realized query-sigma spread; (0, sigma*, sigma*) on "fixed".
        _std, _lo, _hi = getattr(self, "_opsd_sigma_spread", (0.0, 0.0, 0.0))
        info["opsd_sigma_q_std"] = _std
        info["opsd_sigma_q_min"] = _lo
        info["opsd_sigma_q_max"] = _hi
        return loss, info

    def _rl_sampling_phase(
        self, return_rewards: bool = True, return_intermediate: bool = False,
    ) -> tuple[dict[str, Any], Optional[tuple[torch.Tensor, list[str]]]]:
        """Base rollout and advantages, then the OPSD target stage.

        ``super()`` owns rollout, reward scoring and advantages (identical to
        MeanFlowNFT's). This adds ``z_q``, the anchor, ``xp`` / ``xm``, ``sigma_q``
        and the per-sample ``r`` (``opsd_r``) that the fitting phase replays.
        Skipped when ``return_rewards=False`` or OPSD is disabled; both are
        rank-uniform, so no rank can desync the collectives in the target stage.
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

        # The pack holds CPU tensors; _rl_train_phase moves them to the device.
        sample_pack["opsd_z_q"] = z_q.detach().cpu()
        sample_pack["opsd_y_0"] = anchor_out.detach().cpu()
        sample_pack["opsd_y_plus"] = xp_out.detach().cpu()
        sample_pack["opsd_y_minus"] = xm_out.detach().cpu()
        self._opsd_pack_sigma(sample_pack, sigma_q)
        self._opsd_last_stats = stats

        if sample_pack.get("opsd_y_0") is not None:
            r_all = self._opsd_collect_r(sample_pack["opsd_y_0"].shape[0])
            if r_all is not None:
                sample_pack["opsd_r"] = r_all
        return sample_pack, preview

    def _rl_train_phase(self, sample_pack: dict[str, Any]) -> dict[str, float]:
        """OPSD fitting phase: inner epochs over shuffled rows, one query per sample.

        ``opsd_r`` is published as ``_opsd_r_replay`` for :meth:`_opsd_draw_r`
        for the duration of the phase and cleared in ``finally``.
        """
        self._opsd_r_replay = sample_pack.get("opsd_r")
        try:
            cfg = self._rl_cfg()
            device = next(self.generator.parameters()).device

            if not bool(cfg.opsd_enabled):
                raise ValueError(
                    f"{type(self).__name__} requires fewstep_rl.opsd_enabled=true. "
                    "With it off no reward-gradient targets are built and this arm "
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
            # One backward per sample per inner epoch.
            effective_grad_accum_steps = grad_accum

            # A per-sample sigma_q must be permuted with the other per-sample
            # tensors, or a query state would get another sample's noise level.
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
                # The replayed r must follow the same row permutation.
                self._opsd_permute_r(perm)

                for start in range(0, n_local, inner_bs):
                    end = min(start + inner_bs, n_local)
                    bsz = end - start
                    sig_b = ssig[start:end] if sigma_is_per_sample else ssig
                    t_b = self._opsd_t_raw(sig_b, bsz, device)
                    # Replays the r each anchor was built at, so the live
                    # prediction is queried at the identical (t, r).
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
            # One fitting query per sample per inner epoch.
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
            # With ``disable_cd`` the central difference is skipped on purpose and
            # S is defined to be zero even when r != t, so this must precede the
            # r == t guard. Config-level, hence rank-uniform.
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

    def decoupled_opsd_loss(
        self,
        x_0_pred_v: torch.Tensor,  # fp32, the only tensor carrying grad
        x_anchor_1: torch.Tensor,  # fp32 detached: anchor / self-regression target
        x_anchor_2: torch.Tensor,  # fp32 detached: probe center
        xp: torch.Tensor,          # fp32 detached: reward-ascent target
        xm: torch.Tensor,          # fp32 detached: reward-descent target
        omega: torch.Tensor,       # [B] in [0,1], the advantage fitting weight
        beta: float,
        spatial: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Three-term OPSD loss built from the base's two loss primitives::

            proximal  self_regression(x_0_pred_v, x_anchor_1, W)
            attract   reward_augmentation(x_0_pred_v, xp, x_anchor_2, omega_p, W)
            repel     reward_augmentation(x_0_pred_v, x_anchor_2, xm, omega_m, W)

        ``reward_augmentation(x_pos, x_neg)`` pushes the prediction along
        ``x_pos - x_neg``, so the repel term's arguments are deliberately swapped
        (pushing away from ``xm``). The sign does not show in the loss value, only
        in the gradient. Gradients match the fused OPSD kernel exactly; loss
        values do not (cross terms), so never compare on the loss.
        """
        with torch.no_grad():
            W, omega_p, omega_m = self.opsd_weights(
                x_0_pred_v, x_anchor_1, x_anchor_2, xp, xm, omega, beta, spatial,
            )

        # Proximal anchor term (implicit trust region).
        l_reg = self.self_regression(x_0_pred_v, x_anchor_1, W, spatial)
        # Attraction toward xp.
        l_pos = self.reward_augmentation(x_0_pred_v, xp, x_anchor_2, omega_p, W, spatial)
        # Repulsion away from xm (arguments swapped, see docstring).
        l_neg = self.reward_augmentation(x_0_pred_v, x_anchor_2, xm, omega_m, W, spatial)

        return l_reg + l_pos + l_neg, {
            # Key names shared with the other trainers' logs.
            "affine_coeff_mean": float(W.mean().item()),
            "affine_adv_coeff_mean": float((omega_p - omega_m).mean().item()),
            "self_regression_loss": float(l_reg.detach().mean()),
            # Sum of the two reward terms.
            "reward_augmentation_loss": float((l_pos + l_neg).detach().mean()),
            "opsd_pull_pos": float(l_pos.detach().mean()),
            "opsd_push_neg": float(l_neg.detach().mean()),
            "opsd_omega_pos_mean": float(omega_p.mean().item()),
            "opsd_omega_neg_mean": float(omega_m.mean().item()),
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
        """Frozen per-sample loss weights ``(W, omega_p, omega_m)``.

        With probes ``probe_pm = x_anchor_2 +- beta * (x_0_pred_v - x_anchor_1)``::

            gamma_pm = max(mean|probe_pm - x_pm|, eps_gamma)
            a_+ = omega / gamma_+          a_- = (1 - omega) / gamma_-
            W = beta**2 * (a_+ + a_-),     W * omega_pm = beta * a_pm

        ``beta**2`` (not ``beta``) matches the DiffusionOPSD gradient, where the
        probe branch and the outer MSE each contribute one factor of beta.
        Computed in fp64: these are reciprocals of possibly small residuals.
        """
        eps_gamma = float(self._rl_cfg().opsd_eps_gamma)
        bc = (-1, *([1] * (x_0_pred_v.ndim - 1)))

        # Probe direction from the anchor, centre from the probe centre (same
        # split as the base's ``decoupled_weights``).
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
            # Positive-branch-only ablation: zeroing a_- removes both the
            # repulsion from xm and its share of the proximal weight.
            a_m = torch.zeros_like(a_m)
        ab = a_p + a_m

        W = (beta * beta * ab).reshape(-1)          # beta**2, see docstring
        omega_p = a_p / (beta * ab)                 # W * omega_p == a_p * beta
        omega_m = a_m / (beta * ab)                 # W * omega_m == a_m * beta
        return W, omega_p, omega_m
