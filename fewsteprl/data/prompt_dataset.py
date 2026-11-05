"""Text prompt dataset and dataloader.

Supported files: JSON (list of strings, or dicts with a "prompt" / "text" key),
JSONL (one string or dict per line) and TXT (one prompt per line). A dataset
directory may be given instead; its prompt file is found by name
(``_DATASET_PROMPT_FILES``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


# Prompt file names searched in a dataset directory, in priority order.
_DATASET_PROMPT_FILES = [
    "train.txt",
    "train.jsonl",
    "train_metadata.jsonl",
    "prompts.json",
    "prompts.txt",
]


class PromptDataset(Dataset):
    """Dataset of text prompts from a prompt file or dataset directory.

    Args:
        repeat: number of times to repeat the prompt list.
        return_metadata: yield ``{"prompt", "metadata"}`` dicts instead of strings.
    """

    def __init__(self, path: str, repeat: int = 1, return_metadata: bool = False):
        self.return_metadata = bool(return_metadata)

        if os.path.isdir(path):
            path = self._find_prompt_file(path, prefer_metadata=self.return_metadata)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            self.prompts, self.metadatas = self._load_json(
                path, return_metadata=self.return_metadata
            )
        elif ext == ".jsonl":
            self.prompts, self.metadatas = self._load_jsonl(
                path, return_metadata=self.return_metadata
            )
        elif ext in (".txt", ".text"):
            self.prompts, self.metadatas = self._load_txt(
                path, return_metadata=self.return_metadata
            )
        else:
            raise ValueError(
                f"Unsupported prompt file format: {ext}. Use .json, .jsonl, or .txt"
            )

        if repeat > 1:
            self.prompts = self.prompts * repeat
            if self.metadatas is not None:
                self.metadatas = self.metadatas * repeat

        logger.info(f"Loaded {len(self.prompts)} prompts from {path}")

    @staticmethod
    def _find_prompt_file(directory: str, prefer_metadata: bool = False) -> str:
        """Search a dataset directory for a standard prompt file."""
        ordered_files = list(_DATASET_PROMPT_FILES)
        if prefer_metadata:
            ordered_files = ["train_metadata.jsonl"] + [
                f for f in ordered_files if f != "train_metadata.jsonl"
            ]

        for fname in ordered_files:
            fpath = os.path.join(directory, fname)
            if os.path.exists(fpath):
                return fpath
        raise FileNotFoundError(
            f"No prompt file found in dataset directory: {directory}. "
            f"Expected one of: {_DATASET_PROMPT_FILES}"
        )

    @staticmethod
    def _load_json(path: str, return_metadata: bool = False) -> tuple[list[str], Optional[list[dict]]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"JSON file must contain a non-empty list: {path}")
        if isinstance(data[0], str):
            prompts = data
            metadatas = [{"prompt": p} for p in prompts] if return_metadata else None
            return prompts, metadatas
        elif isinstance(data[0], dict) and ("prompt" in data[0] or "text" in data[0]):
            prompts = [item.get("prompt", item.get("text", "")) for item in data]
            metadatas = data if return_metadata else None
            return prompts, metadatas
        else:
            raise ValueError(
                f"JSON entries must be strings or dicts with a 'prompt' key: {path}"
            )

    @staticmethod
    def _load_jsonl(path: str, return_metadata: bool = False) -> tuple[list[str], Optional[list[dict]]]:
        """Load prompts from a JSONL file; dict lines are kept as metadata."""
        prompts = []
        metadatas = [] if return_metadata else None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        prompts.append(item.get("prompt", item.get("text", "")))
                        if metadatas is not None:
                            metadatas.append(item)
                    elif isinstance(item, str):
                        prompts.append(item)
                        if metadatas is not None:
                            metadatas.append({"prompt": item})
                    else:
                        raise ValueError(
                            f"JSONL entries must be dict or string in file: {path}"
                        )
        return prompts, metadatas

    @staticmethod
    def _load_txt(path: str, return_metadata: bool = False) -> tuple[list[str], Optional[list[dict]]]:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        metadatas = [{"prompt": p} for p in lines] if return_metadata else None
        return lines, metadatas

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str | dict:
        if self.return_metadata:
            metadata = (
                self.metadatas[idx]
                if self.metadatas is not None
                else {"prompt": self.prompts[idx]}
            )
            return {"prompt": self.prompts[idx], "metadata": metadata}
        return self.prompts[idx]


def create_prompt_dataloader(
    prompt_path: str,
    batch_size: int,
    distributed: bool = True,
    seed: int = 42,
    repeat: int = 1,
    return_metadata: bool = False,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    """Create a prompt DataLoader (per-GPU ``batch_size``, ``drop_last=True``).

    Returns ``(loader, sampler)``; ``sampler`` is a ``DistributedSampler`` when
    ``distributed``, else None.
    """
    dataset = PromptDataset(prompt_path, repeat=repeat, return_metadata=return_metadata)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        drop_last=True,
        collate_fn=_collate_prompts,
    )
    return loader, sampler


def _collate_prompts(batch: list[str] | list[dict]) -> list[str] | dict[str, list]:
    """Collate prompts, preserving optional metadata when provided."""
    if batch and isinstance(batch[0], dict):
        prompts = []
        metadatas = []
        for item in batch:
            prompts.append(item.get("prompt", item.get("text", "")))
            metadatas.append(item.get("metadata", {}))
        return {"prompts": prompts, "metadata": metadatas}
    return batch
