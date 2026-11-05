"""VideoMultiScorer: weighted multi-reward evaluation for video models.

Self-contained port of the GenRL video rewards:

- ``hpsv3_general``: HPSv3 per-frame quality (prompt "A high-quality image"),
  mean over frames.
- ``hpsv3_percentile``: HPSv3 per-frame text alignment, mean of the top-30% frames.
- ``videoalign_mq``: VideoAlign motion quality on a grayscale mp4.
- ``videoalign_ta``: VideoAlign text-video alignment on a color mp4.

The reward packages (HPSv3, VideoAlign) are imported lazily, so this module imports
before they are installed. Usage::

    scorer = VideoMultiScorer(device, {"hpsv3_general": 1.0, ...})
    score_details, _ = scorer(videos[B,F,3,H,W], prompts)
    # score_details = {"hpsv3_general": [...], ..., "mean": [...]}  (weighted sum)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from fewsteprl.utils.fast_init import fast_init
from fewsteprl.utils.logging import setup_logging

from .video_utils import (
    cleanup_temp_files,
    prepare_video_images,
    preserve_accelerate_state,
    save_frame_to_temp_png,
    save_video_to_temp_mp4,
    to_grayscale,
)

logger = logging.getLogger(__name__)

# Default reward repo/checkpoint locations; override via VideoMultiScorer kwargs
# (hpsv3_root=, videoalign_dir=, videoalign_ckpt=, ...).
_REWARDS_DIR = Path(__file__).resolve().parent
_DEFAULT_HPSV3_ROOT = _REWARDS_DIR / "HPSv3"
_DEFAULT_VIDEOALIGN_DIR = _REWARDS_DIR / "VideoAlign"

# Each VLM is loaded once and shared: HPSv3 across general/percentile, VideoAlign
# across mq/ta.
_HPSV3_INFERENCER_CACHE: dict = {}
_VIDEOALIGN_INFERENCER_CACHE: dict = {}


def _transformers_renamed_qwen2vl() -> bool:
    """True if transformers>=4.52, whose Qwen2-VL layout moves the vision tower and
    text model under ``model`` (text model renamed ``language_model``). The reward
    checkpoints were saved with the older layout."""
    try:
        import transformers

        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return (major, minor) >= (4, 52)
    except Exception:  # noqa: BLE001
        return True


def convert_qwen2vl_old_to_new_keys(state_dict: dict) -> dict:
    """Remap pre-4.52 Qwen2-VL ``state_dict`` keys to the transformers>=4.52 layout.

    Applies the same mapping as transformers' ``_checkpoint_conversion_mapping``::

        ^visual                              -> model.visual
        ^model(?!\\.(language_model|visual))  -> model.language_model

    which is needed because the reward models load their checkpoints with a manual
    ``load_state_dict`` that bypasses the automatic conversion. A leading PEFT
    ``base_model.model.`` prefix is preserved; keys already in the new layout are
    unchanged.
    """
    import re

    out: dict = {}
    for k, v in state_dict.items():
        prefix, rest = "", k
        if rest.startswith("base_model.model."):
            prefix, rest = "base_model.model.", rest[len("base_model.model.") :]
        new = re.sub(r"^visual", "model.visual", rest)
        if new == rest:  # the two rules are mutually exclusive
            new = re.sub(r"^model(?!\.(language_model|visual))", "model.language_model", rest)
        out[prefix + new] = v
    return out


def _alias_qwen2vl_embed_tokens(model) -> None:
    """Re-expose ``self.model.embed_tokens`` for transformers>=4.52.

    The reward models' ``forward`` calls ``self.model.embed_tokens``, which 4.52
    moved to ``self.model.language_model.embed_tokens``; the rest of their forward
    is unaffected (``self.visual`` is kept as a compatibility property). Aliasing
    the module on every ``Qwen2VLModel`` submodule covers both the bare (HPSv3) and
    PEFT-wrapped (VideoAlign) models. No-op on older transformers.
    """
    if not _transformers_renamed_qwen2vl():
        return
    for mod in model.modules():
        if type(mod).__name__ == "Qwen2VLModel" and "embed_tokens" not in mod._modules:
            lm = getattr(mod, "language_model", None)
            if lm is not None and hasattr(lm, "embed_tokens"):
                mod.embed_tokens = lm.embed_tokens


def _as_torch_device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _get_hpsv3_inferencer(
    device: torch.device,
    hpsv3_root: str | None = None,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
):
    """Build or fetch the cached ``HPSv3RewardInferencer``.

    With ``config_path`` and ``checkpoint_path`` both None, hpsv3 downloads
    ``MizzenAI/HPSv3`` and uses its packaged ``HPSv3_7B.yaml``.
    """
    import sys as _sys
    root = Path(hpsv3_root) if hpsv3_root else _DEFAULT_HPSV3_ROOT
    if root.exists() and str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    # transformers>=4.49 moved ``VideoInput`` to ``video_utils``; hpsv3 still imports
    # it from ``image_utils``, so alias it back.
    try:
        import transformers.image_utils as _iu
        if not hasattr(_iu, "VideoInput"):
            from transformers.video_utils import VideoInput as _VI
            _iu.VideoInput = _VI
    except Exception:  # noqa: BLE001
        pass
    from hpsv3 import HPSv3RewardInferencer  # lazy

    key = (str(device), config_path, checkpoint_path)
    if key not in _HPSV3_INFERENCER_CACHE:
        # hpsv3 loads ``HPSv3.safetensors`` via ``load_file`` + strict
        # ``load_state_dict``, which fails on transformers>=4.52's renamed keys.
        # Temporarily patch ``load_file`` to remap only that file (the base
        # Qwen2-VL shards go through ``from_pretrained`` and are converted there).
        import safetensors.torch as _stt

        _orig_load_file = _stt.load_file
        _do_remap = _transformers_renamed_qwen2vl()

        def _load_file_remap(filename, *args, **kwargs):
            sd = _orig_load_file(filename, *args, **kwargs)
            try:
                if _do_remap and os.path.basename(str(filename)) == "HPSv3.safetensors":
                    sd = convert_qwen2vl_old_to_new_keys(sd)
            except Exception:  # noqa: BLE001
                pass
            return sd

        _stt.load_file = _load_file_remap
        try:
            with preserve_accelerate_state(), fast_init(device, init_weights=False):
                _HPSV3_INFERENCER_CACHE[key] = HPSv3RewardInferencer(
                    config_path=config_path, checkpoint_path=checkpoint_path, device=device,
                )
        finally:
            _stt.load_file = _orig_load_file
        # transformers>=4.52: hpsv3's forward needs ``self.model.embed_tokens``.
        _alias_qwen2vl_embed_tokens(_HPSV3_INFERENCER_CACHE[key].model)
    return _HPSV3_INFERENCER_CACHE[key]


def _get_videoalign_inferencer(device: torch.device, checkpoint_path: str, videoalign_dir: str | None):
    """Build or fetch the cached ``VideoVLMRewardInference`` (shared by mq/ta)."""
    import importlib.util
    import sys as _sys

    va_dir = Path(videoalign_dir) if videoalign_dir else _DEFAULT_VIDEOALIGN_DIR
    inference_path = va_dir / "inference.py"
    if not inference_path.is_file():
        raise FileNotFoundError(
            f"VideoAlign inference.py not found: {inference_path}"
        )
    module_name = "_fewsteprl_videoalign_inference"
    module = _sys.modules.get(module_name)
    if module is None:
        # ``inference.py`` does bare sibling imports (``import utils``, ``import
        # data``, ...), so its directory is put on sys.path only while it loads.
        # Restoring sys.path is not enough: those siblings stay cached in
        # sys.modules under generic names, and e.g. VideoAlign's ``utils`` would
        # shadow the ``utils`` that VBench's vendored DINO imports later in the
        # same process. So the sibling modules loaded here are also evicted from
        # sys.modules afterwards; ``module`` keeps working because it holds direct
        # references to what it imported.
        _inserted = False
        if va_dir.exists() and str(va_dir) not in _sys.path:
            _sys.path.insert(0, str(va_dir))
            _inserted = True
        _before = set(_sys.modules)
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                inference_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load VideoAlign from {inference_path}")
            module = importlib.util.module_from_spec(spec)
            _sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            if _inserted:
                try:
                    _sys.path.remove(str(va_dir))
                except ValueError:
                    pass
            # Evict only modules that appeared during this load and come from va_dir.
            va_prefix = str(va_dir)
            for name in set(_sys.modules) - _before:
                if name == module_name:
                    continue
                mod = _sys.modules.get(name)
                origin = getattr(getattr(mod, "__spec__", None), "origin", None) \
                    or getattr(mod, "__file__", None)
                if origin and str(origin).startswith(va_prefix):
                    _sys.modules.pop(name, None)
        original_loader = module.load_model_from_checkpoint

        def _compatible_checkpoint_loader(model, checkpoint_dir, checkpoint_step):
            try:
                return original_loader(model, checkpoint_dir, checkpoint_step)
            except RuntimeError:
                import glob

                checkpoint_paths = glob.glob(
                    os.path.join(checkpoint_dir, "checkpoint-*")
                )
                checkpoint_paths.sort(
                    key=lambda item: int(item.rsplit("-", 1)[-1]),
                    reverse=True,
                )
                if not checkpoint_paths:
                    raise
                if checkpoint_step not in (None, -1):
                    requested = os.path.join(
                        checkpoint_dir,
                        f"checkpoint-{checkpoint_step}",
                    )
                    selected = (
                        requested
                        if requested in checkpoint_paths
                        else checkpoint_paths[0]
                    )
                else:
                    selected = checkpoint_paths[0]
                state = torch.load(
                    os.path.join(selected, "model.pth"),
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(
                    convert_qwen2vl_old_to_new_keys(state),
                    strict=True,
                )
                return model, selected.rsplit("checkpoint-", 1)[-1]

        module.load_model_from_checkpoint = _compatible_checkpoint_loader
    VideoVLMRewardInference = module.VideoVLMRewardInference

    key = (str(va_dir.resolve()), os.path.abspath(checkpoint_path), str(device))
    if key not in _VIDEOALIGN_INFERENCER_CACHE:
        with preserve_accelerate_state(), fast_init(device, init_weights=False):
            _VIDEOALIGN_INFERENCER_CACHE[key] = VideoVLMRewardInference(
                load_from_pretrained=os.path.abspath(checkpoint_path),
                device=str(device), dtype=torch.bfloat16,
            )
        # transformers>=4.52: VideoAlign's forward needs ``self.model.embed_tokens``.
        _alias_qwen2vl_embed_tokens(_VIDEOALIGN_INFERENCER_CACHE[key].model)
    return _VIDEOALIGN_INFERENCER_CACHE[key]


def _extract_reward_scalar(reward) -> float:
    if isinstance(reward, (list, tuple)) and len(reward) > 0:
        return float(reward[0])
    if isinstance(reward, torch.Tensor):
        return 0.0 if reward.numel() == 0 else float(reward.flatten()[0].item())
    if hasattr(reward, "item"):
        return float(reward.item())
    return float(reward)


# --- HPSv3 per-frame scorer (general / percentile) ---

class HPSv3VideoScorer:
    """HPSv3 per-frame scorer. ``mode`` is ``"general"`` or ``"percentile"``."""

    GENERAL_PROMPT = "A high-quality image"

    def __init__(self, device, mode: str, hpsv3_root: str | None = None,
                 config_path: str | None = None, checkpoint_path: str | None = None):
        assert mode in ("general", "percentile"), mode
        self.mode = mode
        self.device = _as_torch_device(device)
        # Shared across general/percentile (loaded once).
        self.inferencer = _get_hpsv3_inferencer(
            self.device, hpsv3_root, config_path=config_path, checkpoint_path=checkpoint_path,
        )

    def to(self, device):
        device = _as_torch_device(device)
        if getattr(self.inferencer, "device", None) != device:
            self.inferencer.device = device
            self.inferencer.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def __call__(self, videos, prompts, metadata=None) -> np.ndarray:
        images_np, is_video = prepare_video_images(videos)
        dev_type = self.device.type
        rewards: list[float] = []
        tmp: list[str] = []
        try:
            for i in range(images_np.shape[0]):
                frames = images_np[i] if is_video else images_np[i : i + 1]
                paths = [save_frame_to_temp_png(f) for f in frames]
                tmp.extend(paths)
                prompt = self.GENERAL_PROMPT if self.mode == "general" else str(prompts[i])
                frame_prompts = [prompt] * len(paths)
                with torch.amp.autocast(
                    device_type=dev_type,
                    dtype=torch.bfloat16,
                    enabled=dev_type == "cuda",
                ):
                    raw = self.inferencer.reward(image_paths=paths, prompts=frame_prompts)
                scores = [_extract_reward_scalar(r) for r in raw]
                if not scores:
                    rewards.append(0.0)
                elif self.mode == "general":
                    rewards.append(float(np.mean(scores)))
                else:
                    scores_sorted = sorted(scores, reverse=True)
                    k = max(1, int(len(scores_sorted) * 0.3))
                    rewards.append(float(np.mean(scores_sorted[:k])))
        finally:
            cleanup_temp_files(tmp)
        return np.asarray(rewards, dtype=np.float32)


# --- VideoAlign whole-video scorer (MQ / TA) ---

class VideoAlignScorer:
    """VideoAlign scorer. ``mode`` is ``"mq"`` (grayscale) or ``"ta"`` (color)."""

    def __init__(self, device, mode: str, checkpoint_path: str | None = None,
                 videoalign_dir: str | None = None, fps: float = 8.0):
        assert mode in ("mq", "ta"), mode
        self.mode = mode
        self.key = "MQ" if mode == "mq" else "TA"
        self.fps = fps
        self.device = _as_torch_device(device)
        va_dir = Path(videoalign_dir) if videoalign_dir else _DEFAULT_VIDEOALIGN_DIR
        ckpt = checkpoint_path or str(va_dir / "checkpoints")
        # Shared across mq/ta (loaded once per checkpoint/device).
        self.inferencer = _get_videoalign_inferencer(self.device, ckpt, videoalign_dir)

    def to(self, device):
        device = _as_torch_device(device)
        if str(getattr(self.inferencer, "device", "")) != str(device):
            self.inferencer.device = str(device)
            self.inferencer.model.to(str(device))
        self.device = device
        return self

    @torch.no_grad()
    def __call__(self, videos, prompts, metadata=None) -> np.ndarray:
        images_np, is_video = prepare_video_images(videos)
        if self.mode == "mq":
            images_np = to_grayscale(images_np)
        dev_type = self.device.type
        rewards: list[float] = []
        tmp: list[str] = []
        try:
            for i in range(images_np.shape[0]):
                frames = images_np[i] if is_video else images_np[i : i + 1]
                path = save_video_to_temp_mp4(frames, fps=self.fps)
                tmp.append(path)
                with torch.amp.autocast(
                    device_type=dev_type,
                    dtype=torch.bfloat16,
                    enabled=dev_type == "cuda",
                ):
                    out = self.inferencer.reward(
                        video_paths=[path], prompts=[str(prompts[i])], use_norm=True,
                    )
                rewards.append(float(out[0][self.key]))
        finally:
            cleanup_temp_files(tmp)
        return np.asarray(rewards, dtype=np.float32)


_VIDEO_SCORER_FACTORIES = {
    "hpsv3_general": lambda device, **kw: HPSv3VideoScorer(
        device, "general", hpsv3_root=kw.get("hpsv3_root"),
        config_path=kw.get("hpsv3_config_path"), checkpoint_path=kw.get("hpsv3_checkpoint_path"),
    ),
    "hpsv3_percentile": lambda device, **kw: HPSv3VideoScorer(
        device, "percentile", hpsv3_root=kw.get("hpsv3_root"),
        config_path=kw.get("hpsv3_config_path"), checkpoint_path=kw.get("hpsv3_checkpoint_path"),
    ),
    "videoalign_mq": lambda device, **kw: VideoAlignScorer(
        device, "mq", checkpoint_path=kw.get("videoalign_ckpt"), videoalign_dir=kw.get("videoalign_dir"),
    ),
    "videoalign_ta": lambda device, **kw: VideoAlignScorer(
        device, "ta", checkpoint_path=kw.get("videoalign_ckpt"), videoalign_dir=kw.get("videoalign_dir"),
    ),
}

VIDEO_REWARD_NAMES = tuple(_VIDEO_SCORER_FACTORIES.keys())


class VideoMultiScorer:
    """Weighted multi-reward wrapper for video models."""

    def __init__(self, device, score_dict, allow_unavailable: bool = False, **scorer_kwargs):
        self.device = _as_torch_device(device)
        self.requested_score_dict = dict(score_dict)
        self.score_dict: dict[str, float] = {}
        self.score_fns: dict[str, object] = {}
        self._unavailable: dict[str, str] = {}

        for name, weight in score_dict.items():
            if name not in _VIDEO_SCORER_FACTORIES:
                raise KeyError(
                    f"Unsupported video reward {name!r}. "
                    f"Supported: {list(_VIDEO_SCORER_FACTORIES.keys())}"
                )
            try:
                with preserve_accelerate_state():
                    fn = _VIDEO_SCORER_FACTORIES[name](self.device, **scorer_kwargs)
            except Exception as e:  # noqa: BLE001
                if not allow_unavailable:
                    raise
                self._unavailable[name] = f"{type(e).__name__}: {e}"
                warnings.warn(
                    f"[VideoMultiScorer] Skip unavailable reward {name!r}: {type(e).__name__}: {e}",
                    stacklevel=2,
                )
                continue
            finally:
                # Reward backends can mutate root logger on import; restore.
                setup_logging()
            self.score_dict[name] = weight
            self.score_fns[name] = fn

        self.active_reward_names = list(self.score_dict.keys())
        self.unavailable_rewards = dict(self._unavailable)

    def __call__(self, videos, prompts, metadata=None, only_strict: bool = True):
        del metadata, only_strict
        score_details: dict[str, object] = {}
        total: np.ndarray | None = None
        for name, weight in self.score_dict.items():
            scores = self.score_fns[name](videos, prompts)
            scores = np.asarray(scores, dtype=np.float32)
            score_details[name] = scores
            weighted = float(weight) * scores
            total = weighted if total is None else total + weighted
        if total is None:
            total = np.zeros(len(prompts), dtype=np.float32)
        # "mean" is the weighted sum of active rewards.
        score_details["mean"] = total
        return score_details, {}

    def to(self, target_device):
        for fn in self.score_fns.values():
            if hasattr(fn, "to"):
                fn.to(target_device)
        self.device = _as_torch_device(target_device)
        return self
