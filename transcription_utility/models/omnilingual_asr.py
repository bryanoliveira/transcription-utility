"""Omnilingual ASR backend."""
from __future__ import annotations
from . import BaseTranscriber, free_gpu_memory

DEFAULT_MODEL_CARD = "omniASR_LLM_Unlimited_7B_v2"

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

    # The Unlimited model card is long-context; chunking it at 120s throws that
    # away, so default to 2h chunks as the original omni script did.
    default_chunk_duration = 60 * 60 * 2

    def __init__(self, model_card: str = DEFAULT_MODEL_CARD):
        self.model_card = model_card
        self._pipeline = None

    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        self._pipeline = ASRInferencePipeline(model_card=self.model_card)

    def transcribe(self, audio_paths: list[str], language: str = "pt") -> list[str]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        lang = LANGUAGE_MAP.get(language.lower(), language)
        # The pipeline already hands us exactly one batch.
        transcriptions = self._pipeline.transcribe(
            audio_paths,
            lang=[lang] * len(audio_paths),
            batch_size=len(audio_paths),
        )
        return [t if isinstance(t, str) else str(t) for t in transcriptions]

    def unload(self) -> None:
        self._pipeline = None
        free_gpu_memory()
