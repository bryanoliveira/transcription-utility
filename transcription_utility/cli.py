"""CLI: `transcribe run` for batch jobs, `transcribe serve` for the HTTP service."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .engine import TranscriptionEngine
from .pipeline import BatchPipeline
from .registry import MODEL_NAMES, get_transcriber

MODEL_HELP = "Model to use. One of: " + ", ".join(MODEL_NAMES)


def model_options(f):
    """Options shared by every command that loads a model."""
    for opt in reversed([
        click.option("--model", "-m", default="qwen3-1.7b", show_default=True,
                     help=MODEL_HELP),
        click.option("--language", "-l", default="pt", show_default=True,
                     help="Language ISO code (e.g. pt, en, es)."),
        click.option("--gpu-memory-utilization", default=0.9, show_default=True,
                     type=float, help="GPU memory fraction for vLLM backends."),
        click.option("--model-card", default=None,
                     help="Omnilingual only: model card to load."),
    ]):
        f = opt(f)
    return f


def build_engine(
    model, gpu_memory_utilization, model_card, keep_alive_minutes: float = 0.0
) -> TranscriptionEngine:
    try:
        transcriber = get_transcriber(
            model,
            gpu_memory_utilization=gpu_memory_utilization,
            model_card=model_card,
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    return TranscriptionEngine(transcriber, keep_alive_minutes=keep_alive_minutes)


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Transcribe audio files with a local ASR model."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@model_options
@click.option("--output-dir", "-o", default=None, type=click.Path(path_type=Path),
              help="Where to write transcripts. Defaults to alongside the input.")
@click.option("--output-format", default="txt", show_default=True,
              type=click.Choice(["txt", "json"]))
@click.option("--chunk-duration", default=None, type=int,
              help="Chunk length in seconds. Defaults to the model's preference.")
@click.option("--overlap", default=2, show_default=True, type=int,
              help="Overlap between consecutive chunks, in seconds.")
@click.option("--batch-size", default=None, type=int,
              help="Chunks per inference call. Defaults to the model's preference.")
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True,
              help="Skip files that already have a transcript.")
@click.option("--dry-run", is_flag=True, help="List what would be processed, then stop.")
def run(input_path, model, language, gpu_memory_utilization, model_card,
        output_dir, output_format, chunk_duration, overlap, batch_size,
        skip_existing, dry_run) -> None:
    """Transcribe an audio file or a directory of them.

    INPUT_PATH is a single audio file or a directory.
    """
    engine = build_engine(model, gpu_memory_utilization, model_card)
    try:
        results = BatchPipeline(
            engine,
            language=language,
            output_dir=output_dir,
            output_format=output_format,
            chunk_duration=chunk_duration,
            overlap=overlap,
            batch_size=batch_size,
            skip_existing=skip_existing,
            dry_run=dry_run,
        ).run(input_path)
    finally:
        engine.close()

    ok = sum(r.success for r in results)
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if not r.success and not r.skipped]

    click.echo(f"\nDone. {ok} transcribed" + (f", {len(skipped)} skipped" if skipped else "") + ".")
    if failed:
        click.echo(f"\n{len(failed)} file(s) failed:", err=True)
        for r in failed:
            click.echo(f"  {r.audio_path.name}: {r.error}", err=True)
        sys.exit(1)


@main.command()
@model_options
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Address to bind. Use 0.0.0.0 to accept requests from the network.")
@click.option("--port", default=8000, show_default=True, type=int,
              help="Port to bind.")
@click.option("--keep-alive", default=5.0, show_default=True, type=float,
              metavar="MINUTES",
              help="Unload the model after this many idle minutes. 0 pins it in memory.")
@click.option("--max-upload-mb", default=500, show_default=True, type=int,
              help="Reject uploads larger than this.")
@click.option("--preload", is_flag=True,
              help="Load the model at startup instead of on the first request.")
def serve(model, language, gpu_memory_utilization, model_card, host, port,
          keep_alive, max_upload_mb, preload) -> None:
    """Serve an OpenAI-compatible transcription API.

    The model loads on the first request that needs it and unloads again after
    --keep-alive idle minutes, so an idle server holds no GPU memory.
    """
    try:
        import uvicorn
        from .server import create_app
    except ImportError:
        raise click.ClickException(
            'the server needs extra deps: pip install "transcription-utility[server]"'
        )

    engine = build_engine(model, gpu_memory_utilization, model_card,
                          keep_alive_minutes=keep_alive)
    if preload:
        engine.load()

    warm = "pinned in memory" if keep_alive == 0 else f"unloaded after {keep_alive}min idle"
    click.echo(f"Serving {engine.model_name} on http://{host}:{port} ({warm})")
    try:
        app = create_app(engine, default_language=language, max_upload_mb=max_upload_mb)
        uvicorn.run(app, host=host, port=port)
    finally:
        engine.close()
