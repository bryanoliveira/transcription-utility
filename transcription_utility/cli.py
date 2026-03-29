"""CLI entry point for transcription-utility."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from .pipeline import TranscriptionPipeline
from .registry import get_transcriber


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model", "-m",
    default="qwen3-1.7b",
    show_default=True,
    help=(
        "Model to use. Options: qwen3-0.6b, qwen3-1.7b, "
        "whisper-tiny, whisper-base, whisper-small, whisper-medium, "
        "whisper-large, whisper-large-v2, whisper-large-v3, whisper-large-v3-turbo, "
        "omnilingual"
    ),
)
@click.option(
    "--language", "-l",
    default="pt",
    show_default=True,
    help="Language ISO code (e.g. pt, en, es) or full name for qwen (e.g. Portuguese).",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory to write transcripts. Defaults to same directory as input.",
)
@click.option(
    "--output-format",
    default="txt",
    show_default=True,
    type=click.Choice(["txt", "json"]),
    help="Output format. txt: timestamped text. json: structured with metadata.",
)
@click.option(
    "--chunk-duration",
    default=120,
    show_default=True,
    type=int,
    help="Duration of each audio chunk in seconds.",
)
@click.option(
    "--overlap",
    default=2,
    show_default=True,
    type=int,
    help="Overlap between consecutive chunks in seconds.",
)
@click.option(
    "--gpu-memory-utilization",
    default=0.9,
    show_default=True,
    type=float,
    help="GPU memory utilization fraction for vLLM-based models.",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help="Batch size (for models that support batching).",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip files that already have a completed transcript.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print files that would be processed without running inference.",
)
def main(
    input_path: Path,
    model: str,
    language: str,
    output_dir: Path | None,
    output_format: str,
    chunk_duration: int,
    overlap: int,
    gpu_memory_utilization: float,
    batch_size: int | None,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    """Transcribe audio file(s) using a configurable ASR model.

    INPUT_PATH can be a single audio file or a directory of audio files.
    """
    try:
        transcriber = get_transcriber(
            model,
            gpu_memory_utilization=gpu_memory_utilization,
            batch_size=batch_size,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    pipeline = TranscriptionPipeline(
        transcriber=transcriber,
        language=language,
        output_dir=output_dir,
        output_format=output_format,
        chunk_duration=chunk_duration,
        overlap=overlap,
        skip_existing=skip_existing,
        dry_run=dry_run,
    )

    results = pipeline.run(input_path)

    failed = [r for r in results if not r.success]
    if failed:
        click.echo(f"\n{len(failed)} file(s) failed:", err=True)
        for r in failed:
            click.echo(f"  {r.audio_path.name}: {r.error}", err=True)
        sys.exit(1)

    click.echo(f"\nDone. {len(results)} file(s) transcribed.")
