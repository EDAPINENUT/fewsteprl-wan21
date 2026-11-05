"""Logging utilities: section-prefixed wandb logging (``WandbLogger``), a CUDA
stage timer (``Timer``) and root-logger setup.

Metrics are logged as ``<section>/<name>`` against wandb's ``_step`` (= global
step). New section prefixes should be added to the ``define_metric`` loop in
``WandbLogger.__init__``.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch

from fewsteprl.parallel.utils import is_main_process

logger = logging.getLogger(__name__)


class WandbLogger:
    """wandb logger with section-prefixed metrics (e.g. ``profile/gpu_mem_gb``).

    Only logs on the main process and when ``enabled``; wandb files go to
    ``output_dir``.
    """

    def __init__(
        self,
        project: str = "fewsteprl-wan21",
        run_name: str = "",
        entity: str = "",
        config: dict | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
        output_dir: str = "",
    ):
        self.enabled = enabled and is_main_process()
        self._run = None

        if self.enabled:
            import wandb
            wandb_dir = output_dir or None
            if wandb_dir:
                os.makedirs(wandb_dir, exist_ok=True)
            self._run = wandb.init(
                project=project,
                name=run_name or None,
                entity=entity or None,
                config=config,
                tags=tags,
                resume="allow",
                dir=wandb_dir,
            )
            # Plot every section against _step (= global_step). Prefixes are listed
            # explicitly because a bare "*" glob would also match _step itself.
            for prefix in ["dmd", "fake_score", "nft", "nft_buffer",
                           "eval", "teacher_eval", "media", "profile", "debug"]:
                wandb.define_metric(f"{prefix}/*", step_metric="_step")
            logger.info(f"wandb initialized: {wandb.run.url}")

    def log(self, metrics: dict[str, Any], step: int, section: str = "") -> None:
        """Log ``metrics`` at ``step``, as ``section/name`` if ``section`` is given."""
        if not self.enabled:
            return

        if section:
            metrics = {f"{section}/{k}": v for k, v in metrics.items()}

        self._run.log(metrics, step=step)

    def log_multi_section(self, sections: dict[str, dict[str, Any]], step: int) -> None:
        """Log ``{section: {name: value}}`` in a single wandb call."""
        if not self.enabled:
            return

        flat = {}
        for section, metrics in sections.items():
            for k, v in metrics.items():
                flat[f"{section}/{k}"] = v
        self._run.log(flat, step=step)

    def log_images(
        self,
        images: list,
        captions: list[str],
        step: int,
        key: str = "media/eval",
    ) -> None:
        """Log PIL images with one caption each under ``key``."""
        if not self.enabled:
            return
        import wandb
        wandb_images = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(images, captions)
        ]
        self._run.log({key: wandb_images}, step=step)

    def log_videos(
        self,
        videos: list,
        captions: list[str],
        step: int,
        key: str = "media/eval_video",
        fps: int = 16,
    ) -> None:
        """Log ``[F, 3, H, W]`` float clips in ``[0, 1]`` as mp4 videos.

        Each clip is encoded to a temporary mp4 (libx264, needs ``imageio-ffmpeg``),
        which is deleted after the log call (wandb copies it at log time). ``key``
        must start with a registered section, e.g. ``media/``.
        """
        if not self.enabled:
            return
        import os as _os
        import tempfile

        import imageio
        import numpy as np
        import wandb

        tmp_paths: list[str] = []
        try:
            wandb_videos = []
            for vid, cap in zip(videos, captions):
                # [F, 3, H, W] float[0,1] -> [F, H, W, 3] uint8 for imageio.
                arr = (vid.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                arr = np.transpose(arr, (0, 2, 3, 1))
                f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                f.close()
                imageio.mimsave(f.name, list(arr), fps=int(fps), codec="libx264", format="FFMPEG")
                tmp_paths.append(f.name)
                wandb_videos.append(wandb.Video(f.name, caption=cap, format="mp4"))
            self._run.log({key: wandb_videos}, step=step)
        finally:
            for p in tmp_paths:
                try:
                    _os.remove(p)
                except Exception:  # noqa: BLE001 - best-effort temp cleanup
                    pass

    def finish(self) -> None:
        """Finish the wandb run."""
        if self.enabled and self._run is not None:
            self._run.finish()


class Timer:
    """Wall-clock timer for training stages (``with timer.measure("name"): ...``)."""

    def __init__(self):
        self._times: dict[str, list[float]] = defaultdict(list)
        self._start: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        """Time a code block (synchronizes CUDA before and after)."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self._times[name].append(elapsed)

    def get_last(self, name: str) -> float:
        """Get the most recent measurement for a named operation."""
        if name not in self._times or len(self._times[name]) == 0:
            return 0.0
        return self._times[name][-1]

    def get_average(self, name: str) -> float:
        """Get the average time across all measurements for a named operation."""
        if name not in self._times or len(self._times[name]) == 0:
            return 0.0
        return sum(self._times[name]) / len(self._times[name])

    def get_metrics(self) -> dict[str, float]:
        """Get average times for all tracked operations, suitable for logging."""
        return {f"time_{k}_s": self.get_average(k) for k in self._times}

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._times.clear()


def get_gpu_memory_gb() -> float:
    """Peak allocated GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def setup_logging(level: int = logging.INFO) -> None:
    """(Re)configure the root logger; non-main ranks log at WARNING only."""
    fmt = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    target_level = level if is_main_process() else logging.WARNING

    # Replace existing handlers (basicConfig would be a no-op if any exist).
    root = logging.getLogger()
    root.setLevel(target_level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setLevel(target_level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(handler)

    # Silence httpx per-request INFO logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
