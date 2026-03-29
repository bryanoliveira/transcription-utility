"""Core transcription pipeline: file discovery, locking, atomic writes, batching."""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio import (
    AUDIO_EXTENSIONS,
    cleanup_chunks,
    format_timestamp,
    get_duration,
    split_into_chunks,
    try_recover,
)
from .models import BaseTranscriber

MIN_DURATION = 5.0  # seconds — skip very short files


@dataclass
class TranscriptionResult:
    audio_path: Path
    output_path: Path
    success: bool
    error: str | None = None
    chunks: list[dict] = field(default_factory=list)


class TranscriptionPipeline:
    """
    Orchestrates audio transcription for one or more files.

    Handles:
    - File discovery in directories
    - Idempotency (.processing lock + .txt.partial → .txt atomic rename)
    - Chunking and audio preprocessing
    - Batched model inference
    - Output in txt or json format
    """

    def __init__(
        self,
        transcriber: BaseTranscriber,
        language: str = "pt",
        output_dir: Path | str | None = None,
        output_format: str = "txt",
        chunk_duration: int = 120,
        overlap: int = 2,
        skip_existing: bool = True,
        dry_run: bool = False,
    ):
        self.transcriber = transcriber
        self.language = language
        self.output_dir = Path(output_dir) if output_dir else None
        self.output_format = output_format
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.skip_existing = skip_existing
        self.dry_run = dry_run

    def _find_audio_files(self, input_path: Path) -> list[Path]:
        """Return list of audio files to process."""
        if input_path.is_file():
            return [input_path]
        files = []
        for p in sorted(input_path.iterdir()):
            if p.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(p)
        return files

    def _output_path(self, audio_path: Path) -> Path:
        out_dir = self.output_dir or audio_path.parent
        ext = "." + self.output_format
        return out_dir / (audio_path.stem + ext)

    def _lock_path(self, audio_path: Path) -> Path:
        return audio_path.parent / (audio_path.name + ".processing")

    def _partial_path(self, output_path: Path) -> Path:
        return output_path.parent / (output_path.name + ".partial")

    def _is_pending(self, audio_path: Path, output_path: Path) -> bool:
        if self.skip_existing and output_path.exists():
            return False
        if self._lock_path(audio_path).exists():
            # Check if lock is stale (>4h)
            lock = self._lock_path(audio_path)
            age = time.time() - lock.stat().st_mtime
            if age < 14400:
                print(f"  [skip] {audio_path.name} — already being processed (lock age {age:.0f}s)")
                return False
            print(f"  [warn] {audio_path.name} — stale lock ({age:.0f}s), removing")
            lock.unlink()
        return True

    def _write_lock(self, audio_path: Path, model_name: str) -> Path:
        lock = self._lock_path(audio_path)
        lock.write_text(
            f"host={socket.gethostname()}\n"
            f"pid={os.getpid()}\n"
            f"started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"model={model_name}\n"
        )
        return lock

    def _write_output(
        self,
        output_path: Path,
        chunks_data: list[dict],
        model_name: str,
    ) -> None:
        partial = self._partial_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.output_format == "json":
                with open(partial, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": model_name,
                            "language": self.language,
                            "chunks": chunks_data,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            else:
                with open(partial, "w", encoding="utf-8") as f:
                    for chunk in chunks_data:
                        ts = format_timestamp(chunk["start"])
                        f.write(f"[{ts}]\n{chunk['text']}\n\n")
            partial.rename(output_path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    def run(self, input_path: Path | str) -> list[TranscriptionResult]:
        """
        Transcribe audio file(s) at input_path.

        Args:
            input_path: Path to audio file or directory.

        Returns:
            List of TranscriptionResult per file.
        """
        input_path = Path(input_path)
        audio_files = self._find_audio_files(input_path)
        results: list[TranscriptionResult] = []

        # Filter to pending files
        pending: list[tuple[Path, Path]] = []
        for af in audio_files:
            out = self._output_path(af)
            if self._is_pending(af, out):
                pending.append((af, out))
            else:
                print(f"  [skip] {af.name}")

        if not pending:
            print("No files to process.")
            return results

        if self.dry_run:
            for af, out in pending:
                print(f"  [dry-run] {af.name} → {out}")
            return results

        print(f"Processing {len(pending)} file(s)...")

        # Load model once for all files
        model_name = type(self.transcriber).__name__
        self.transcriber.load()
        try:
            for audio_path, output_path in pending:
                result = self._process_file(audio_path, output_path, model_name)
                results.append(result)
                if result.success:
                    print(f"  ✓ {audio_path.name} → {output_path}")
                else:
                    print(f"  ✗ {audio_path.name}: {result.error}")
        finally:
            self.transcriber.unload()

        return results

    def _process_file(
        self, audio_path: Path, output_path: Path, model_name: str
    ) -> TranscriptionResult:
        lock = self._write_lock(audio_path, model_name)
        recovered: Path | None = None
        temp_dir: str | None = None

        try:
            duration = get_duration(audio_path)
            effective_path = audio_path

            if duration is None:
                print(f"  [warn] {audio_path.name}: could not read duration, attempting recovery...")
                recovered = try_recover(audio_path)
                if recovered is None:
                    return TranscriptionResult(
                        audio_path=audio_path,
                        output_path=output_path,
                        success=False,
                        error="Could not read duration and recovery failed",
                    )
                duration = get_duration(recovered)
                effective_path = recovered
                print(f"  [info] recovered, duration={duration:.1f}s")

            if duration < MIN_DURATION:
                return TranscriptionResult(
                    audio_path=audio_path,
                    output_path=output_path,
                    success=False,
                    error=f"Too short ({duration:.1f}s < {MIN_DURATION}s)",
                )

            print(f"\n  Transcribing: {audio_path.name} ({duration/3600:.2f}h)")
            temp_dir = tempfile.mkdtemp()
            chunks = split_into_chunks(
                effective_path,
                temp_dir,
                duration,
                chunk_duration=self.chunk_duration,
                overlap=self.overlap,
            )

            if not chunks:
                return TranscriptionResult(
                    audio_path=audio_path,
                    output_path=output_path,
                    success=False,
                    error="No chunks produced (ffmpeg failed?)",
                )

            print(f"  {len(chunks)} chunk(s), transcribing...")
            chunk_paths = [cp for cp, _ in chunks]
            texts = self.transcriber.transcribe(chunk_paths, language=self.language)

            chunks_data = []
            for (chunk_path, start), text in zip(chunks, texts):
                end = min(start + self.chunk_duration, duration)
                chunks_data.append({"start": start, "end": end, "text": text.strip()})
                print(f"    [{format_timestamp(start)}] {text[:80]}...")

            self._write_output(output_path, chunks_data, model_name)

            return TranscriptionResult(
                audio_path=audio_path,
                output_path=output_path,
                success=True,
                chunks=chunks_data,
            )

        except Exception as e:
            return TranscriptionResult(
                audio_path=audio_path,
                output_path=output_path,
                success=False,
                error=str(e),
            )
        finally:
            lock.unlink(missing_ok=True)
            if temp_dir:
                cleanup_chunks([], temp_dir)
            if recovered and recovered.exists():
                recovered.unlink()
