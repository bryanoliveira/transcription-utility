"""Audio preprocessing utilities: duration probing, stream detection, chunking."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


AUDIO_EXTENSIONS: set[str] = {
    ".m4a", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".wma", ".aac", ".mp4", ".mov", ".mkv"
}

CHUNK_DIR_PREFIX = "transcription-utility-"
STALE_CHUNK_DIR_AGE = 4 * 60 * 60  # seconds


def make_chunk_dir() -> str:
    """Create a temp directory for audio chunks, tagged so it can be swept."""
    return tempfile.mkdtemp(prefix=CHUNK_DIR_PREFIX)


def sweep_stale_chunk_dirs(max_age: float = STALE_CHUNK_DIR_AGE) -> int:
    """
    Delete chunk directories abandoned by a previous run.

    Nothing runs on SIGKILL, so an OOM kill or SLURM preemption always leaks
    the chunks of the file in flight. Sweeping at startup is the only way to
    reclaim them; the age check keeps it safe when runs overlap.
    """
    removed = 0
    for path in Path(tempfile.gettempdir()).glob(f"{CHUNK_DIR_PREFIX}*"):
        try:
            if path.is_dir() and time.time() - path.stat().st_mtime > max_age:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    return removed


def get_duration(audio_path: Path | str) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def count_audio_streams(audio_path: Path | str) -> int:
    """Return the number of audio streams in the file."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    return len(result.stdout.strip().splitlines())


def try_recover(audio_path: Path) -> Path | None:
    """
    Re-mux audio ignoring errors to salvage truncated/corrupt containers.
    Returns path to recovered file, or None if recovery failed.
    """
    recovered = audio_path.with_stem(audio_path.stem + "_recovered")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-i", str(audio_path),
            "-c", "copy",
            str(recovered),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and recovered.exists() and recovered.stat().st_size > 0:
        duration = get_duration(recovered)
        if duration is not None and duration > 0:
            return recovered
    if recovered.exists():
        recovered.unlink()
    return None


def split_into_chunks(
    audio_path: Path | str,
    temp_dir: str,
    total_duration: float,
    chunk_duration: int = 120,
    overlap: int = 2,
    sample_rate: int = 16000,
) -> list[tuple[str, float]]:
    """
    Split audio into WAV chunks.

    Handles multi-stream files by mixing all audio streams with amix.
    Outputs mono PCM at the specified sample rate.

    Returns:
        List of (chunk_file_path, start_time_seconds) tuples.
    """
    audio_path = Path(audio_path)
    step = chunk_duration - overlap
    chunks: list[tuple[str, float]] = []
    start = 0.0
    idx = 0

    num_streams = count_audio_streams(audio_path)
    if num_streams > 1:
        audio_filter = [
            "-filter_complex",
            f"{''.join(f'[0:a:{i}]' for i in range(num_streams))}"
            f"amix=inputs={num_streams}:duration=longest",
        ]
    else:
        audio_filter = ["-ac", "1"]

    while start < total_duration:
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(chunk_duration),
                *audio_filter,
                "-acodec", "pcm_s16le",
                "-ar", str(sample_rate),
                chunk_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.isfile(chunk_path):
            print(f"  [warn] ffmpeg failed for chunk {idx} (start={start:.1f}s)")
            if result.stderr:
                print(f"    ffmpeg stderr: {result.stderr.strip()[-200:]}")
        else:
            chunks.append((chunk_path, start))
        start += step
        idx += 1

    return chunks


def cleanup_chunks(chunks: list[tuple[str, float]], temp_dir: str | None = None) -> None:
    """Remove chunk files and optionally the temp directory."""
    for chunk_path, _ in chunks:
        try:
            os.remove(chunk_path)
        except FileNotFoundError:
            pass
    if temp_dir:
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
