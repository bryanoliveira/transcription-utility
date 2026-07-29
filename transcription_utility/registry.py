"""Model registry — resolve a model name to a transcriber instance."""
from __future__ import annotations

from .models import BaseTranscriber

MODEL_NAMES = (
    "qwen3-0.6b", "qwen3-1.7b",
    "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
    "whisper-large", "whisper-large-v2", "whisper-large-v3", "whisper-large-v3-turbo",
    "omnilingual",
)


def get_transcriber(
    model_name: str,
    gpu_memory_utilization: float = 0.9,
    model_card: str | None = None,
) -> BaseTranscriber:
    """
    Instantiate the transcriber for the given model name (see MODEL_NAMES).

    Args:
        model_card: Omnilingual only. Overrides the default model card.
    """
    name = model_name.lower()

    # Backends are imported lazily so one missing extra doesn't break the rest.
    if name.startswith("qwen"):
        from .models.qwen_asr import QwenTranscriber
        return QwenTranscriber(
            model_name=name, gpu_memory_utilization=gpu_memory_utilization
        )
    if name.startswith("whisper"):
        from .models.whisper_asr import WhisperTranscriber
        return WhisperTranscriber(model_name=name)
    if name == "omnilingual":
        from .models.omnilingual_asr import DEFAULT_MODEL_CARD, OmnilingualTranscriber
        return OmnilingualTranscriber(model_card=model_card or DEFAULT_MODEL_CARD)

    raise ValueError(f"unknown model {model_name!r}; expected one of {', '.join(MODEL_NAMES)}")
