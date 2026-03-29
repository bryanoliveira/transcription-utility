"""Omnilingual ASR backend."""
from __future__ import annotations
from . import BaseTranscriber

LANGUAGE_MAP = {
    "pt": "por_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
}


class OmnilingualTranscriber(BaseTranscriber):
    """Transcription backend using Omnilingual ASR pipeline."""

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self._pipeline = None

    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        self._pipeline = ASRInferencePipeline()

    def transcribe(self, audio_paths: list[str], language: str = "pt") -> list[str]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        lang = LANGUAGE_MAP.get(language.lower(), language)
        transcriptions = self._pipeline.transcribe(
            audio_paths,
            lang=[lang] * len(audio_paths),
            batch_size=self.batch_size,
        )
        return [t if isinstance(t, str) else str(t) for t in transcriptions]

    def unload(self) -> None:
        del self._pipeline
        self._pipeline = None
