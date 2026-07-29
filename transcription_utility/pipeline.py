"""Transcription of one file (shared by both entry points) and batch orchestration."""
from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .audio import (
    AUDIO_EXTENSIONS,
    cleanup_chunks,
    format_timestamp,
    get_duration,
    make_chunk_dir,
    split_into_chunks,
    sweep_stale_chunk_dirs,
    try_recover,
)
from .engine import TranscriptionEngine

MIN_DURATION = 5.0  # seconds — shorter clips aren't worth a model pass
STALE_LOCK_AGE = 4 * 60 * 60  # seconds — locks older than this are assumed dead


class AudioTooShort(Exception):
    """Audio is below MIN_DURATION. A skip, not a failure."""


class UnreadableAudio(Exception):
    """ffprobe could not read the file and recovery failed."""


def transcribe_audio(
    engine: TranscriptionEngine,
    audio_path: Path,
    language: str = "pt",
    chunk_duration: int | None = None,
    overlap: int = 2,
    batch_size: int | None = None,
    on_progress: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """
    Transcribe a single audio file into timestamped chunks.

    The one place audio becomes text — the CLI and the HTTP server both come
    through here, so they cannot drift apart.

    Args:
        on_progress: called with all chunks so far after each inference batch.
            The batch path uses it to keep a crash-recoverable partial file.

    Returns:
        List of {"start", "end", "text"} dicts.

    Raises:
        AudioTooShort, UnreadableAudio, or whatever the backend raises.
    """
    chunk_duration = chunk_duration or engine.chunk_duration
    batch_size = max(1, batch_size or engine.batch_size)

    duration = get_duration(audio_path)
    effective_path = audio_path
    recovered: Path | None = None

    if duration is None:
        recovered = try_recover(audio_path)
        if recovered is None:
            raise UnreadableAudio(f"could not read {audio_path.name}, recovery failed")
        duration = get_duration(recovered)
        effective_path = recovered

    if duration < MIN_DURATION:
        raise AudioTooShort(f"too short ({duration:.1f}s < {MIN_DURATION}s)")

    chunk_dir = make_chunk_dir()
    chunks: list[tuple[str, float]] = []
    try:
        chunks = split_into_chunks(
            effective_path, chunk_dir, duration,
            chunk_duration=chunk_duration, overlap=overlap,
        )
        if not chunks:
            raise RuntimeError("no chunks produced (ffmpeg failed?)")

        chunks_data: list[dict] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = engine.transcribe([cp for cp, _ in batch], language=language)
            for (_, start), text in zip(batch, texts):
                chunks_data.append({
                    "start": start,
                    "end": min(start + chunk_duration, duration),
                    "text": text.strip(),
                })
            if on_progress:
                on_progress(chunks_data)
        return chunks_data
    finally:
        cleanup_chunks(chunks, chunk_dir)
        if recovered and recovered.exists():
            recovered.unlink()


@dataclass
class TranscriptionResult:
    audio_path: Path
    output_path: Path
    success: bool
    error: str | None = None
    chunks: list[dict] = field(default_factory=list)
    # True when the file was deliberately passed over (e.g. too short) rather
    # than failing. Callers should not treat these as errors.
    skipped: bool = False


class BatchPipeline:
    """
    Transcribes a directory (or a single file) to disk.

    Idempotent: a `.processing` lock keeps concurrent runs off the same file,
    and output lands via `.partial` → atomic rename so a half-written
    transcript is never mistaken for a finished one.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        language: str = "pt",
        output_dir: Path | str | None = None,
        output_format: str = "txt",
        chunk_duration: int | None = None,
        overlap: int = 2,
        batch_size: int | None = None,
        skip_existing: bool = True,
        dry_run: bool = False,
    ):
        self.engine = engine
        self.language = language
        self.output_dir = Path(output_dir) if output_dir else None
        self.output_format = output_format
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.batch_size = batch_size
        self.skip_existing = skip_existing
        self.dry_run = dry_run

    def _output_path(self, audio_path: Path) -> Path:
        out_dir = self.output_dir or audio_path.parent
        return out_dir / (audio_path.stem + "." + self.output_format)

    def _lock_path(self, audio_path: Path) -> Path:
        return audio_path.parent / (audio_path.name + ".processing")

    def _skip_reason(self, audio_path: Path, output_path: Path) -> str | None:
        """Return why this file should be passed over, or None if it's pending."""
        if self.skip_existing and output_path.exists():
            return "transcript already exists"
        lock = self._lock_path(audio_path)
        if lock.exists():
            age = time.time() - lock.stat().st_mtime
            if age < STALE_LOCK_AGE:
                return f"already being processed (lock age {age:.0f}s)"
            print(f"  [warn] {audio_path.name} — stale lock ({age:.0f}s), removing")
            lock.unlink()
        return None

    def _write_output(self, path: Path, chunks_data: list[dict]) -> None:
        """Write the full transcript so far. Called after every batch."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if self.output_format == "json":
                json.dump(
                    {
                        "model": self.engine.model_name,
                        "language": self.language,
                        "chunks": chunks_data,
                    },
                    f, ensure_ascii=False, indent=2,
                )
            else:
                for chunk in chunks_data:
                    f.write(f"[{format_timestamp(chunk['start'])}]\n{chunk['text']}\n\n")

    def run(self, input_path: Path | str) -> list[TranscriptionResult]:
        """Transcribe the file or directory at input_path."""
        sweep_stale_chunk_dirs()
        input_path = Path(input_path)

        if input_path.is_file():
            audio_files = [input_path]
        else:
            audio_files = [
                p for p in sorted(input_path.iterdir())
                if p.suffix.lower() in AUDIO_EXTENSIONS
            ]

        pending: list[tuple[Path, Path]] = []
        for af in audio_files:
            out = self._output_path(af)
            reason = self._skip_reason(af, out)
            if reason is None:
                pending.append((af, out))
            else:
                print(f"  [skip] {af.name} — {reason}")

        if not pending:
            print("No files to process.")
            return []

        if self.dry_run:
            for af, out in pending:
                print(f"  [dry-run] {af.name} → {out}")
            return []

        print(f"Processing {len(pending)} file(s)...")
        results = []
        for audio_path, output_path in pending:
            result = self._process_file(audio_path, output_path)
            results.append(result)
            if result.success:
                print(f"  ✓ {audio_path.name} → {output_path}")
            elif result.skipped:
                print(f"  [skip] {audio_path.name} — {result.error}")
            else:
                print(f"  ✗ {audio_path.name}: {result.error}")
        return results

    def _process_file(self, audio_path: Path, output_path: Path) -> TranscriptionResult:
        lock = self._lock_path(audio_path)
        lock.write_text(
            f"host={socket.gethostname()}\n"
            f"pid={os.getpid()}\n"
            f"started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"model={self.engine.model_name}\n"
        )
        partial = output_path.parent / (output_path.name + ".partial")

        try:
            print(f"\n  Transcribing: {audio_path.name}")
            try:
                # Rewrite after every batch so a hard kill keeps the finished
                # chunks. Transcripts are kilobytes; rewriting is free.
                chunks = transcribe_audio(
                    self.engine, audio_path,
                    language=self.language,
                    chunk_duration=self.chunk_duration,
                    overlap=self.overlap,
                    batch_size=self.batch_size,
                    on_progress=lambda data: self._write_output(partial, data),
                )
                partial.rename(output_path)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise

            return TranscriptionResult(audio_path, output_path, True, chunks=chunks)

        except AudioTooShort as e:
            return TranscriptionResult(
                audio_path, output_path, False, error=str(e), skipped=True
            )
        except Exception as e:
            return TranscriptionResult(audio_path, output_path, False, error=str(e))
        finally:
            lock.unlink(missing_ok=True)
