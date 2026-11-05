"""VBench T2V evaluation for the Wan trainers (port of AnyFlow's protocol).

1. Read the augmented VBench info JSON and expand each prompt to
   ``num_samples_per_prompt`` samples, seeded by global index.
2. Generate each video from ``aug_prompt_en`` with the CFG-free flow-map rollout
   and save it as ``{prompt_en}-{idx}.mp4`` (VBench matches by ``prompt_en``).
3. Score the 16 VBench dimensions with the ``vbench`` package and aggregate
   quality / semantic / overall scores with AnyFlow's normalization.

Gated by ``eval.vbench.enabled``. Generation and scoring are sharded across ranks.
If ``vbench`` is unavailable, scoring is skipped and the videos are kept for
offline scoring.
"""

from __future__ import annotations

import json
import logging
import os

import torch

from fewsteprl.parallel.utils import barrier, get_rank, get_world_size, is_main_process

logger = logging.getLogger(__name__)


# Per-dimension normalization ranges (from AnyFlow); unlisted dimensions use [0, 1].
_NORM_RANGES = {
    "subject_consistency": [0.1462, 1.0],
    "motion_smoothness": [0.706, 0.9975],
    "temporal_flickering": [0.6293, 1.0],
    "background_consistency": [0.2615, 1.0],
    "scene": [0.0, 0.8222],
    "appearance_style": [0.0009, 0.2855],
    "temporal_style": [0.0, 0.364],
    "overall_consistency": [0.0, 0.364],
}

_DIMENSIONS = [
    "subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "color", "spatial_relationship",
    "scene", "temporal_style", "overall_consistency", "human_action",
    "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style",
]


def _norm(metric: float, key: str) -> float:
    rng = _NORM_RANGES.get(key, [0.0, 1.0])
    metric = max(metric, rng[0])
    metric = min(metric, rng[1])
    return (metric - rng[0]) / (rng[1] - rng[0])


def _aggregate(info: dict) -> dict:
    """Add AnyFlow's quality / semantic / overall aggregate scores."""
    out = dict(info)
    try:
        out["quality_score"] = (
            _norm(info["subject_consistency"], "subject_consistency")
            + _norm(info["background_consistency"], "background_consistency")
            + _norm(info["motion_smoothness"], "motion_smoothness")
            + _norm(info["temporal_flickering"], "temporal_flickering")
            + _norm(info["dynamic_degree"], "dynamic_degree") * 0.5
            + _norm(info["aesthetic_quality"], "aesthetic_quality")
            + _norm(info["imaging_quality"], "imaging_quality")
        ) / 6.5
        out["semantic_score"] = (
            _norm(info["object_class"], "object_class")
            + _norm(info["multiple_objects"], "multiple_objects")
            + _norm(info["human_action"], "human_action")
            + _norm(info["color"], "color")
            + _norm(info["spatial_relationship"], "spatial_relationship")
            + _norm(info["scene"], "scene")
            + _norm(info["appearance_style"], "appearance_style")
            + _norm(info["temporal_style"], "temporal_style")
            + _norm(info["overall_consistency"], "overall_consistency")
        ) / 9.0
        out["overall_score"] = 0.2 * out["semantic_score"] + 0.8 * out["quality_score"]
    except KeyError as e:
        logger.warning(f"[VBench] missing dimension {e}; skipping aggregate scores.")
    return out


def _save_video_mp4(frames: torch.Tensor, path: str, fps: int) -> None:
    """Save a ``[F, 3, H, W]`` float[0,1] clip to mp4 (libx264)."""
    import imageio

    arr = (frames.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    imageio.mimsave(path, list(arr), fps=fps, codec="libx264", format="FFMPEG")


@torch.no_grad()
def _generate_one(trainer, model, prompt: str, num_steps: int, seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    embeds, dummy = trainer._encode_prompts([prompt])
    x0 = trainer._rollout_samples(
        embeds, dummy, num_steps=int(num_steps), guidance_scale=1.0, model=model,
        # Eval clip length (video.eval_num_frames; <=0 falls back to training length).
        latent_frames=getattr(trainer, "eval_latent_frames", None),
    )
    return trainer._decode_latents_to_tensor(x0)  # [1, F, 3, H, W]


@torch.no_grad()
def run_vbench_eval(trainer, gen_model=None) -> dict:
    """Run the full VBench protocol. Must be called on all ranks.

    Args:
        trainer: a :class:`~fewsteprl.trainers.rl_general_base.GeneralRLTrainer` instance.
        gen_model: generator to use (defaults to EMA, then the generator).

    Returns:
        Aggregated VBench info dict (rank 0; empty on other ranks / if skipped).
    """
    cfg = trainer.config
    vb = cfg.eval.vbench
    if not vb.enabled:
        return {}
    if not vb.aug_info_json or not os.path.exists(vb.aug_info_json):
        logger.warning(
            "[VBench] eval.vbench.aug_info_json not found (%s); skipping.", vb.aug_info_json,
        )
        return {}
    if vb.cache_dir:
        os.environ["VBENCH_CACHE_DIR"] = vb.cache_dir

    gen_model = gen_model or trainer.models.get("generator_ema") or trainer.models.get("generator")

    # Build the sample list (prompt x num_samples_per_prompt, seeded by index).
    with open(vb.aug_info_json, "r") as f:
        meta = json.load(f)
    samples = []
    for item in meta:
        prompt_en = item["prompt_en"]
        aug = item.get("aug_prompt_en", prompt_en)
        for idx in range(int(vb.num_samples_per_prompt)):
            samples.append({
                "prompt_en": prompt_en,
                "aug_prompt_en": aug,
                "video_path": f"{prompt_en}-{idx}.mp4",
                "seed": len(samples),
            })

    save_root = os.path.join(cfg.output_dir, vb.output_subdir, f"iter_{trainer._nft_epoch}")
    samples_dir = os.path.join(save_root, "samples")
    if is_main_process():
        os.makedirs(samples_dir, exist_ok=True)
    barrier()

    rank, ws = get_rank(), get_world_size()
    my_samples = samples[rank::ws]
    max_count = (len(samples) + ws - 1) // ws  # symmetric iteration count across ranks

    was_training = gen_model.training
    gen_model.eval()
    fps = int(cfg.video.fps)
    num_steps = int(vb.num_inference_steps)

    if is_main_process():
        logger.info(
            "[VBench] generating %d videos (%d prompts x %d) across %d ranks @ %d-step",
            len(samples), len(meta), int(vb.num_samples_per_prompt), ws, num_steps,
        )

    # Every rank must run the same number of rollouts: each one issues FSDP
    # collectives. Do not skip already-existing videos here (a per-rank,
    # data-dependent check desynchronizes ranks and deadlocks); resume is handled
    # per scoring dimension via the ``{dim}_eval_results.json`` caches below.
    for i in range(max_count):
        if i < len(my_samples):
            s = my_samples[i]
            out_path = os.path.join(samples_dir, s["video_path"])
            video = _generate_one(trainer, gen_model, s["aug_prompt_en"], num_steps, s["seed"])
            try:
                _save_video_mp4(video[0], out_path, fps)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[VBench] failed to save {out_path}: {e}")
        else:
            # Symmetric padding forward so all ranks issue equal FSDP collectives.
            _ = _generate_one(trainer, gen_model, samples[0]["aug_prompt_en"], num_steps, 0)

    if was_training:
        gen_model.train()
    barrier()

    # --- scoring: dimensions are distributed across ranks ---
    # Each rank scores a disjoint subset of dimensions and writes
    # ``{dim}_eval_results.json`` into the shared ``info_dir``; rank 0 aggregates.
    # Scoring everything on rank 0 would leave the other ranks waiting in a barrier
    # long enough to hit the NCCL timeout.
    info_dir = os.path.join(save_root, "vbench_info")
    if is_main_process():
        os.makedirs(info_dir, exist_ok=True)
    barrier()  # ensure info_dir exists before any rank writes into it

    my_dims = _DIMENSIONS[rank::ws]
    if my_dims and vb.full_info_json and os.path.exists(vb.full_info_json):
        try:
            from vbench import VBench
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[VBench] `vbench` package unavailable (%s); videos at %s, skipping "
                "scoring (install vbench + pretrained models, then rescore).",
                e, samples_dir,
            )
        else:
            evaluator = VBench(torch.device("cuda"), vb.full_info_json, info_dir)
            for dim in my_dims:
                cached = os.path.join(info_dir, f"{dim}_eval_results.json")
                if os.path.exists(cached):  # resume: keep prior result
                    continue
                try:
                    evaluator.evaluate(
                        videos_path=samples_dir, name=dim, dimension_list=[dim], local=True,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[VBench] dimension '{dim}' failed on rank {rank}: {e}")
    elif my_dims and is_main_process():
        logger.warning(
            "[VBench] eval.vbench.full_info_json not found (%s); skipping scoring.",
            vb.full_info_json,
        )
    barrier()  # all ranks finished scoring their assigned dimensions

    # --- aggregation (rank 0) ---
    eval_info: dict = {}
    if is_main_process():
        for dim in _DIMENSIONS:
            cached = os.path.join(info_dir, f"{dim}_eval_results.json")
            if not os.path.exists(cached):
                logger.warning("[VBench] missing result for '%s'; excluded from aggregate.", dim)
                continue
            with open(cached, "r") as f:
                raw = json.load(f)[dim]
            # VBench stores ``[avg_score, [per_video_scores...]]``; keep the average.
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            eval_info[dim] = float(raw)
        if eval_info:
            eval_info = _aggregate(eval_info)
            # Logs every dimension plus the three aggregates as ``vbench/<key>``.
            if trainer.wandb_logger:
                scalar = {k: float(v) for k, v in eval_info.items() if isinstance(v, (int, float))}
                trainer.wandb_logger.log(scalar, step=trainer.global_step, section="vbench")
            logger.info(
                "[VBench] epoch=%d overall=%.4f quality=%.4f semantic=%.4f",
                trainer._nft_epoch,
                eval_info.get("overall_score", float("nan")),
                eval_info.get("quality_score", float("nan")),
                eval_info.get("semantic_score", float("nan")),
            )
    barrier()
    return eval_info
