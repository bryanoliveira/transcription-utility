# Changelog

## 0.3.0 (2026-07-29)

### Added
- Ollama-style model lifecycle. The server binds and answers immediately holding no
  GPU memory, loads weights on the first request that needs them, and unloads after
  `--keep-alive` idle minutes (default 5). `--keep-alive 0` pins the model
- `POST /unload` to release the model immediately; `GET /health` now reports
  `idle_seconds` and `keep_alive_seconds`
- `--host` / `--port` documented as bind options; `--preload` to load at startup
- `scripts/install-service.sh`: installs `transcribe serve` as a systemd unit.
  `--user` by default (no root), `--system --run-as` for machine-wide, `--print` to
  review the unit first, `--uninstall` to remove it
- README covers batch, API, model sharing, and service install end to end

### Fixed
- Backend `unload()` now runs `gc.collect()` + `torch.cuda.empty_cache()`. Dropping
  the last reference alone left the allocator holding the VRAM, so an idle unload
  freed nothing another process could use

## 0.2.0 (2026-07-29)

Scope narrowed to one job: audio file in, transcription out. No backwards
compatibility with 0.1.x — `TranscriptionPipeline` is now `BatchPipeline` and takes
a `TranscriptionEngine` instead of a transcriber.

### Added
- `TranscriptionEngine`: owns the one loaded model and serializes access to it. A
  batch job and an API request in the same process queue on it rather than loading
  two copies of the weights. The lock is per inference batch, so a long batch job
  delays a request by one chunk-batch, not the whole job
- OpenAI-compatible HTTP service (`transcribe serve`, `pip install ".[server]"`):
  `POST /v1/audio/transcriptions` (`json` / `text` / `verbose_json`), `GET /v1/models`,
  `POST`/`GET /batch` to run and watch a directory job on the same model, `GET /health`
- `transcribe_audio()` — the single audio→text path both entry points call, so the
  CLI and the API cannot drift
- Upload size cap on the HTTP endpoint (`--max-upload-mb`, default 500), returning 413
- `test_pipeline.py`: one runnable self-check, no framework, no GPU, no weights

### Fixed
- The model no longer loads when there is nothing to transcribe. Loading is lazy and
  happens on the first inference call, so a re-run over an up-to-date directory exits
  without paying for weights
- Chunk directories orphaned by SIGKILL are now swept at the start of each batch run.
  Nothing runs on an uncatchable signal, so reclaiming them later is the only option

### Changed
- CLI is now two commands: `transcribe run PATH` and `transcribe serve`
- Dropped the unused `ffmpeg-python` dependency — the code shells out to `ffmpeg`
  and `ffprobe` directly and never imported it

## 0.1.1 (2026-07-29)

### Fixed
- Temp chunk files were never deleted: the cleanup call passed an empty chunk
  list, leaking ~230 MB of WAV per 2h recording into `/tmp` on every run
- Files skipped for being under `MIN_DURATION` were reported as failures, making
  the CLI exit `1` when a directory contained a stray short clip. They are now
  reported as skips (`TranscriptionResult.skipped`) and don't fail the run
- Restored the `omniASR_LLM_Unlimited_7B_v2` model card for the omnilingual
  backend, dropped in the 0.1.0 refactor — `ASRInferencePipeline` was being
  constructed with no model card at all
- Restored per-batch incremental flushing, also lost in the refactor: a crash
  mid-file no longer discards every chunk transcribed so far
- Backends returning the wrong number of transcripts now raise instead of
  silently truncating via `zip`
- Files skipped during discovery are no longer logged twice

### Added
- `--model-card` to override the omnilingual model card
- Per-backend `default_chunk_duration` / `default_batch_size`; `--chunk-duration`
  and `--batch-size` now fall back to them. Omnilingual defaults to 2h chunks
  again rather than being forced to 120s
- `TranscriptionResult.skipped` distinguishes pass-overs from failures

## 0.1.0 (2026-03-29)

Initial release.

### Features
- `TranscriptionPipeline`: core pipeline with idempotency, atomic writes, and batching
- CLI: `transcribe` command with full options
- Backends: Qwen3-ASR (vLLM), OpenAI Whisper, Omnilingual ASR
- Audio utilities: duration probing, stream counting, amix multi-stream mixing, chunking, corrupt file recovery
- Output formats: txt (timestamped) and json (structured with metadata)
- Refactored from `transcribe_qwen.py` and `omni/transcribe_omni.py` in the original transcription project
