from pathlib import Path
import argparse
import os
import subprocess
import tempfile

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


AUDIO_EXTENSIONS = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".wma", ".aac"}
TRANSCRIPTION_EXTENSION = ".txt"

CHUNK_DURATION = 60*60*2  # seconds / 2h
OVERLAP = 2  # seconds
LANGUAGE = "por_Latn"


def get_audio_duration(audio_path: Path) -> float:
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
    return float(result.stdout.strip())


def split_into_chunks(audio_path: Path, temp_dir: str, total_duration: float) -> list[tuple[str, float]]:
    step = CHUNK_DURATION - OVERLAP
    chunks = []
    start = 0.0
    idx = 0

    while start < total_duration:
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(CHUNK_DURATION),
                "-filter_complex", "[0:a:0][0:a:1]amix=inputs=2:duration=longest",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                chunk_path,
            ],
            capture_output=True,
        )
        chunks.append((chunk_path, start))
        start += step
        idx += 1

    return chunks


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_and_save(pipeline: ASRInferencePipeline, chunks: list[tuple[str, float]], output_path: Path, batch_size: int) -> None:
    chunk_paths = [chunk_path for chunk_path, _ in chunks]
    start_times = [start_time for _, start_time in chunks]
    languages = [LANGUAGE] * len(chunks)

    transcriptions = pipeline.transcribe(chunk_paths, lang=languages, batch_size=batch_size)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, transcription in enumerate(transcriptions):
            text = transcription if isinstance(transcription, str) else str(transcription)
            timestamp = format_timestamp(start_times[i])
            f.write(f"[{timestamp}]\n{text}\n\n")
            print(f"  [{timestamp}] {text[:100]}...")


def find_pending_audio_files(input_dir: Path) -> list[Path]:
    pending = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            transcription_path = path.with_suffix(TRANSCRIPTION_EXTENSION)
            if not transcription_path.exists():
                pending.append(path)
    return pending


def cleanup_chunks(chunks: list[tuple[str, float]], temp_dir: str) -> None:
    for chunk_path, _ in chunks:
        os.remove(chunk_path)
    os.rmdir(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files.")
    parser.add_argument("input_dir", nargs="?", default="meetings", help="Folder containing audio files")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for parallel transcription")
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

    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_Unlimited_7B_v2")

    for audio_path in pending:
        output_path = audio_path.with_suffix(TRANSCRIPTION_EXTENSION)
        print(f"\nTranscribing: {audio_path.name}")

        duration = get_audio_duration(audio_path)
        print(f"  Duration: {format_timestamp(duration)}")

        temp_dir = tempfile.mkdtemp()
        chunks = split_into_chunks(audio_path, temp_dir, duration)
        print(f"  Chunks: {len(chunks)}")

        transcribe_and_save(pipeline, chunks, output_path, args.batch_size)
        cleanup_chunks(chunks, temp_dir)

        print(f"  Saved: {output_path}")

    del pipeline
    print("\nDone.")


if __name__ == "__main__":
    main()
