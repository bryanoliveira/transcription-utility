"""Base class for all ASR model backends."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends."""

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
