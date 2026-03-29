"""OpenAI Whisper backend."""
from __future__ import annotations
from . import BaseTranscriber

MODEL_IDS = {
    "whisper-tiny": "tiny",
    "whisper-base": "base",
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large": "large",
    "whisper-large-v2": "large-v2",
    "whisper-large-v3": "large-v3",
    "whisper-large-v3-turbo": "large-v3-turbo",
}


class WhisperTranscriber(BaseTranscriber):
    """Transcription backend using OpenAI Whisper."""

    def __init__(self, model_name: str = "whisper-large-v3", device: str = "cuda"):
        self.model_name = model_name
        self.model_id = MODEL_IDS.get(model_name, model_name.replace("whisper-", ""))
        self.device = device
        self._model = None

    def load(self) -> None:
        import whisper
        self._model = whisper.load_model(self.model_id, device=self.device)

    def transcribe(self, audio_paths: list[str], language: str = "pt") -> list[str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        results = []
        for path in audio_paths:
            result = self._model.transcribe(path, language=language)
            results.append(result["text"].strip())
        return results

    def unload(self) -> None:
        del self._model
        self._model = None
