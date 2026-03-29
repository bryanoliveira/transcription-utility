"""Model registry — resolve model name to transcriber instance."""
from __future__ import annotations
from .models import BaseTranscriber


def get_transcriber(
    model_name: str,
    gpu_memory_utilization: float = 0.9,
    batch_size: int | None = None,
) -> BaseTranscriber:
    """
    Instantiate the appropriate transcriber for the given model name.

    Supported model names:
        qwen3-0.6b, qwen3-1.7b
        whisper-tiny, whisper-base, whisper-small, whisper-medium,
        whisper-large, whisper-large-v2, whisper-large-v3, whisper-large-v3-turbo
        omnilingual
    """
    name = model_name.lower()

    if name.startswith("qwen"):
        from .models.qwen_asr import QwenTranscriber
        return QwenTranscriber(
            model_name=name,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    elif name.startswith("whisper"):
        from .models.whisper_asr import WhisperTranscriber
        return WhisperTranscriber(model_name=name)
    elif name == "omnilingual":
        from .models.omnilingual_asr import OmnilingualTranscriber
        return OmnilingualTranscriber(batch_size=batch_size or 8)
    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            "Supported: qwen3-0.6b, qwen3-1.7b, whisper-*, omnilingual"
        )
