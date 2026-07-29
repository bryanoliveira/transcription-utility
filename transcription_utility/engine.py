"""The single shared model instance. All transcription routes through here."""
from __future__ import annotations

import threading
import time

from .models import BaseTranscriber


class TranscriptionEngine:
    """
    Owns one loaded model, serializes access to it, and unloads it when idle.

    Batch jobs and HTTP requests in the same process share one engine, so
    whichever arrives second waits for the model rather than loading a second
    copy onto the GPU. The lock is held per inference call — one batch of
    chunks — so a long batch job never blocks a request for more than a chunk.

    Loading is lazy: the weights arrive on the first request that needs them,
    and leave again after `keep_alive_minutes` with no work. 0 pins the model
    in memory once loaded.
    """

    def __init__(self, transcriber: BaseTranscriber, keep_alive_minutes: float = 5.0):
        self._transcriber = transcriber
        self._lock = threading.RLock()
        self._loaded = False
        self._last_used = 0.0
        self.keep_alive = max(0.0, keep_alive_minutes) * 60  # seconds; 0 = pinned
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None

    @property
    def model_name(self) -> str:
        return type(self._transcriber).__name__

    @property
    def chunk_duration(self) -> int:
        return self._transcriber.default_chunk_duration

    @property
    def batch_size(self) -> int:
        return self._transcriber.default_batch_size

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used if self._loaded else 0.0

    def load(self) -> None:
        """Load the model now. Optional — transcribe() loads on first use."""
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if not self._loaded:
            print(f"  [engine] loading {self.model_name}...", flush=True)
            self._transcriber.load()
            self._loaded = True
        self._last_used = time.monotonic()

    def transcribe(self, audio_paths: list[str], language: str) -> list[str]:
        """Transcribe one batch of chunks. Loads the model on first call."""
        with self._lock:
            self._load_locked()
            texts = self._transcriber.transcribe(audio_paths, language=language)
            # Refresh after the work, so a long job doesn't age out mid-file.
            self._last_used = time.monotonic()

        # A backend returning the wrong count would silently drop audio.
        if len(texts) != len(audio_paths):
            raise RuntimeError(
                f"{self.model_name} returned {len(texts)} transcript(s) "
                f"for {len(audio_paths)} chunk(s)"
            )
        return texts

    def unload(self) -> str | None:
        """Drop the model from memory. Returns its name if one was loaded."""
        with self._lock:
            if not self._loaded:
                return None
            name = self.model_name
            self._transcriber.unload()
            self._loaded = False
            print(f"  [engine] unloaded {name}", flush=True)
            return name

    def start_idle_reaper(self) -> None:
        """Unload the model after keep_alive seconds with no work. No-op if pinned."""
        if self.keep_alive == 0 or self._reaper is not None:
            return

        def reap():
            # Poll often enough to be punctual without busy-waiting.
            interval = min(30.0, max(1.0, self.keep_alive / 4))
            while not self._stop.wait(interval):
                with self._lock:
                    if self._loaded and self.idle_seconds >= self.keep_alive:
                        self.unload()

        self._reaper = threading.Thread(target=reap, daemon=True)
        self._reaper.start()

    def close(self) -> None:
        self._stop.set()
        if self._reaper is not None:
            self._reaper.join(timeout=5)
            self._reaper = None
        self.unload()
