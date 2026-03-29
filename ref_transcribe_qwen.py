from pathlib import Path
import argparse
import os
import socket
import subprocess
import tempfile
import time

from qwen_asr import Qwen3ASRModel


AUDIO_EXTENSIONS = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".wma", ".aac"}
TRANSCRIPTION_EXTENSION = ".txt"
PARTIAL_SUFFIX = ".partial"      # <name>.txt.partial — in-progress write
PROCESSING_SUFFIX = ".processing" # <name>.m4a.processing — job lock

CHUNK_DURATION = 120  # seconds
OVERLAP = 2  # seconds
LANGUAGE = "Portuguese"
MIN_DURATION = 5  # seconds — skip files shorter than this


def get_audio_duration(audio_path: Path) -> float | None:
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


def try_recover_audio(audio_path: Path) -> Path | None:
    """Re-mux with ffmpeg ignoring errors — can salvage truncated/corrupt containers."""
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
        duration = get_audio_duration(recovered)
        if duration is not None and duration > 0:
            return recovered
    if recovered.exists():
        recovered.unlink()
    return None


def count_audio_streams(audio_path: Path) -> int:
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


def split_into_chunks(audio_path: Path, temp_dir: str, total_duration: float) -> list[tuple[str, float]]:
    step = CHUNK_DURATION - OVERLAP
    chunks = []
    start = 0.0
    idx = 0

    num_streams = count_audio_streams(audio_path)
    if num_streams > 1:
        audio_filter = [
            "-filter_complex",
            f"{''.join(f'[0:a:{i}]' for i in range(num_streams))}amix=inputs={num_streams}:duration=longest",
        ]
    else:
        audio_filter = ["-ac", "1"]

    while start < total_duration:
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(CHUNK_DURATION),
                *audio_filter,
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                chunk_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.isfile(chunk_path):
            print(f"  Warning: ffmpeg failed for chunk {idx} (start={start:.1f}s)")
            if result.stderr:
                print(f"    ffmpeg stderr: {result.stderr.strip()[-200:]}")
        else:
            chunks.append((chunk_path, start))
        start += step
        idx += 1

    return chunks


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_and_save(model, chunks: list[tuple[str, float]], output_path: Path) -> None:
    """Write incrementally to a .partial file, then rename atomically to final path."""
    partial_path = output_path.parent / (output_path.name + PARTIAL_SUFFIX)
    try:
        with open(partial_path, "w", encoding="utf-8") as f:
            for chunk_path, start_time in chunks:
                results = model.transcribe(
                    audio=[chunk_path],
                    language=[LANGUAGE],
                )
                text = results[0].text if results else ""
                timestamp = format_timestamp(start_time)
                f.write(f"[{timestamp}]\n{text}\n\n")
                f.flush()
                print(f"  [{timestamp}] {text[:100]}...")
        # Atomic rename — only appears as .txt when fully written
        partial_path.rename(output_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink(missing_ok=True)
        raise


def lock_path(audio_path: Path) -> Path:
    return audio_path.parent / (audio_path.name + PROCESSING_SUFFIX)


def find_pending_audio_files(input_dir: Path) -> list[Path]:
    """Return audio files that have no .txt and no .processing lock."""
    pending = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            txt = path.with_suffix(TRANSCRIPTION_EXTENSION)
            lock = lock_path(path)
            if not txt.exists() and not lock.exists():
                pending.append(path)
    return pending


def cleanup_chunks(chunks: list[tuple[str, float]], temp_dir: str) -> None:
    for chunk_path, _ in chunks:
        try:
            os.remove(chunk_path)
        except FileNotFoundError:
            pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files.")
    parser.add_argument("input_dir", nargs="?", default="meetings", help="Folder containing audio files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return

    pending = find_pending_audio_files(input_dir)
    if not pending:
        print("No new audio files to transcribe.")
        return

    print(f"Found {len(pending)} file(s) to transcribe:")
    for p in pending:
        print(f"  - {p.name}")

    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.9,
        max_inference_batch_size=-1,
        max_new_tokens=32000,
    )

    failed = []

    for audio_path in pending:
        output_path = audio_path.with_suffix(TRANSCRIPTION_EXTENSION)
        lock = lock_path(audio_path)

        print(f"\nTranscribing: {audio_path.name}")

        # Write lock with metadata for stale-lock detection
        lock.write_text(
            f"host={socket.gethostname()}\n"
            f"pid={os.getpid()}\n"
            f"started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"slurm_job={os.environ.get('SLURM_JOB_ID', 'none')}\n"
        )

        duration = get_audio_duration(audio_path)
        effective_path = audio_path
        recovered_path = None

        try:
            if duration is None:
                print("  Warning: could not read duration (file may be corrupt)")
                print("  Attempting recovery...")
                recovered_path = try_recover_audio(audio_path)
                if recovered_path is not None:
                    duration = get_audio_duration(recovered_path)
                    effective_path = recovered_path
                    print(f"  Recovered! Duration: {format_timestamp(duration)}")
                else:
                    print("  Recovery failed — skipping")
                    failed.append(audio_path.name)
                    continue
            else:
                print(f"  Duration: {format_timestamp(duration)}")

            if duration < MIN_DURATION:
                print(f"  Skipping (shorter than {MIN_DURATION}s)")
                continue

            temp_dir = tempfile.mkdtemp()
            chunks = split_into_chunks(effective_path, temp_dir, duration)
            print(f"  Chunks: {len(chunks)}")

            if not chunks:
                print("  Skipping (no chunks produced — ffmpeg may have failed)")
                failed.append(audio_path.name)
                os.rmdir(temp_dir)
                continue

            try:
                transcribe_and_save(model, chunks, output_path)
            except Exception as e:
                print(f"  Error during transcription: {e}")
                failed.append(audio_path.name)
            else:
                print(f"  Saved: {output_path}")
            finally:
                cleanup_chunks(chunks, temp_dir)

        finally:
            # Always release the lock (crash leaves it for stale detection)
            lock.unlink(missing_ok=True)
            if recovered_path and recovered_path.exists():
                recovered_path.unlink()

    del model

    if failed:
        print(f"\nFailed to process {len(failed)} file(s):")
        for name in failed:
            print(f"  - {name}")
    else:
        print("\nAll files processed successfully.")

    print("Done.")


if __name__ == "__main__":
    main()
