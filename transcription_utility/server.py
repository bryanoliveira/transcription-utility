"""OpenAI-compatible transcription service, sharing one engine with batch jobs."""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .audio import make_chunk_dir, sweep_stale_chunk_dirs
from .engine import TranscriptionEngine
from .pipeline import AudioTooShort, BatchPipeline, UnreadableAudio, transcribe_audio


class BatchRunner:
    """Runs one batch job at a time in a background thread, on the shared engine."""

    def __init__(self, engine: TranscriptionEngine):
        self.engine = engine
        self._thread: threading.Thread | None = None
        self.status: dict = {"state": "idle"}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, path: Path, language: str, output_format: str) -> None:
        if self.running:
            raise HTTPException(409, "a batch job is already running")

        def work():
            try:
                results = BatchPipeline(
                    self.engine, language=language, output_format=output_format
                ).run(path)
                self.status = {
                    "state": "done",
                    "path": str(path),
                    "transcribed": sum(r.success for r in results),
                    "skipped": sum(r.skipped for r in results),
                    "failed": sum(not r.success and not r.skipped for r in results),
                }
            except Exception as e:  # a crashed job must not look like success
                self.status = {"state": "error", "path": str(path), "error": str(e)}

        self.status = {"state": "running", "path": str(path)}
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()


def save_upload(src, dest: Path, max_bytes: int) -> None:
    """Copy an upload to disk, refusing anything over the limit."""
    written = 0
    with open(dest, "wb") as f:
        while chunk := src.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(413, f"file exceeds {max_bytes // (1024 * 1024)} MB")
            f.write(chunk)


def create_app(
    engine: TranscriptionEngine,
    default_language: str = "pt",
    max_upload_mb: int = 500,
) -> FastAPI:
    app = FastAPI(title="transcription-utility")
    batch = BatchRunner(engine)
    max_bytes = max_upload_mb * 1024 * 1024
    sweep_stale_chunk_dirs()  # reclaim whatever a previous kill left behind
    engine.start_idle_reaper()

    # Endpoints are sync `def`, so FastAPI runs them in its threadpool and the
    # engine lock — not the event loop — is what serializes GPU work.

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": engine.model_name, "object": "model", "owned_by": "local"}],
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": engine.model_name,
            "model_loaded": engine.loaded,
            "idle_seconds": round(engine.idle_seconds, 1),
            "keep_alive_seconds": engine.keep_alive,  # 0 = pinned once loaded
            "batch": batch.status,
        }

    @app.post("/unload")
    def unload():
        """Drop the model now, without waiting for the keep-alive timer."""
        return {"unloaded": engine.unload() is not None}

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: UploadFile = File(...),
        # Accepted for OpenAI compatibility and ignored — this server has one
        # model loaded, named in GET /v1/models.
        model: str = Form(None),
        language: str = Form(default_language),
        response_format: str = Form("json"),
    ):
        """OpenAI-compatible: POST an audio file, get its transcription back."""
        if response_format not in {"json", "text", "verbose_json"}:
            raise HTTPException(
                400,
                f"unsupported response_format {response_format!r}; "
                "expected json, text, or verbose_json",
            )

        # Keep the client's extension — ffprobe uses it to pick a demuxer.
        suffix = Path(file.filename or "audio").suffix or ".wav"
        # Tagged dir, so a kill mid-request leaves something the sweep reclaims.
        tmp = Path(make_chunk_dir()) / f"upload{suffix}"
        try:
            save_upload(file.file, tmp, max_bytes)

            try:
                chunks = transcribe_audio(engine, tmp, language=language)
            except AudioTooShort as e:
                raise HTTPException(400, str(e))
            except UnreadableAudio as e:
                raise HTTPException(400, str(e))

            text = " ".join(c["text"] for c in chunks if c["text"]).strip()

            if response_format == "text":
                return PlainTextResponse(text)
            if response_format == "verbose_json":
                return {
                    "task": "transcribe",
                    "language": language,
                    "duration": chunks[-1]["end"] if chunks else 0.0,
                    "text": text,
                    "segments": [
                        {"id": i, "start": c["start"], "end": c["end"], "text": c["text"]}
                        for i, c in enumerate(chunks)
                    ],
                }
            return {"text": text}
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

    @app.post("/batch")
    def start_batch(
        path: str = Form(...),
        language: str = Form(default_language),
        output_format: str = Form("txt"),
    ):
        """Kick off a directory batch on the same model this server is serving."""
        target = Path(path)
        if not target.exists():
            raise HTTPException(400, f"{path} does not exist")
        batch.start(target, language, output_format)
        return batch.status

    @app.get("/batch")
    def batch_status():
        return batch.status

    return app
