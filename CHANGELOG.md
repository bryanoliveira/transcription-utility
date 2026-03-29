# Changelog

## 0.1.0 (2026-03-29)

Initial release.

### Features
- `TranscriptionPipeline`: core pipeline with idempotency, atomic writes, and batching
- CLI: `transcribe` command with full options
- Backends: Qwen3-ASR (vLLM), OpenAI Whisper, Omnilingual ASR
- Audio utilities: duration probing, stream counting, amix multi-stream mixing, chunking, corrupt file recovery
- Output formats: txt (timestamped) and json (structured with metadata)
- Refactored from `transcribe_qwen.py` and `omni/transcribe_omni.py` in the original transcription project
