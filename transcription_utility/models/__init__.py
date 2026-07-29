"""Base class for all ASR model backends."""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod


def free_gpu_memory() -> None:
    """
    Best-effort VRAM reclaim after dropping a model.

    Dropping the last Python reference is not enough — the allocator holds the
    blocks until the cache is emptied, so an idle unload would free nothing
    visible to other processes.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends."""

    # Chunk length this backend is happiest with, in seconds. The pipeline uses
    # it when the caller does not pass an explicit --chunk-duration.
    default_chunk_duration: int = 120

    # How many chunks to send per transcribe() call. The pipeline flushes
    # output after every batch, so this also controls crash-recovery
    # granularity.
    default_batch_size: int = 8

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def transcribe(self, audio_paths: list[str], language: str) -> list[str]:
        """
        Transcribe a list of audio file paths.

        Args:
            audio_paths: List of paths to WAV audio chunks.
            language: Language hint (format depends on backend).

        Returns:
            List of transcription strings, one per input file.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        ...

    def __enter__(self) -> "BaseTranscriber":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
