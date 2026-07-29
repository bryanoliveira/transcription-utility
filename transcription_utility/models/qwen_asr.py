"""Qwen3-ASR backend (vLLM-based)."""
from __future__ import annotations
from . import BaseTranscriber, free_gpu_memory

MODEL_IDS = {
    "qwen3-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-ASR-1.7B",
}

LANGUAGE_MAP = {
    "pt": "Portuguese",
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
}


class QwenTranscriber(BaseTranscriber):
    """Transcription backend using Qwen3-ASR via vLLM."""

    def __init__(
        self,
        model_name: str = "qwen3-1.7b",
        gpu_memory_utilization: float = 0.9,
        max_new_tokens: int = 32000,
    ):
        self.model_name = model_name
        self.model_id = MODEL_IDS.get(model_name, model_name)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_new_tokens = max_new_tokens
        self._model = None

    def load(self) -> None:
        from qwen_asr import Qwen3ASRModel
        self._model = Qwen3ASRModel.LLM(
            model=self.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_inference_batch_size=-1,
            max_new_tokens=self.max_new_tokens,
        )

    def transcribe(self, audio_paths: list[str], language: str = "pt") -> list[str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # Normalize language: accept ISO code or full name
        lang = LANGUAGE_MAP.get(language.lower(), language)
        results = self._model.transcribe(
            audio=audio_paths,
            language=[lang] * len(audio_paths),
        )
        return [r.text if hasattr(r, "text") else str(r) for r in results]

    def unload(self) -> None:
        self._model = None
        free_gpu_memory()
