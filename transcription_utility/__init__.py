"""
transcription_utility — audio in, transcription out. CLI, batch, or HTTP service.
"""
from .engine import TranscriptionEngine
from .pipeline import BatchPipeline, transcribe_audio
from .registry import get_transcriber

__all__ = [
    "TranscriptionEngine",
    "BatchPipeline",
    "transcribe_audio",
    "get_transcriber",
]
__version__ = "0.3.0"
