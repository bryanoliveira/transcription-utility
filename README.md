# transcription-utility

A multi-model audio transcription CLI and Python package. Supports Qwen3-ASR, Whisper, and Omnilingual ASR backends with a unified interface.

## Features

- **Multiple model backends:** Qwen3-ASR (vLLM), OpenAI Whisper, Omnilingual ASR
- **Automatic audio preprocessing:** multi-stream mixing (amix), chunking, corrupt file recovery
- **Idempotent processing:** `.processing` locks + atomic `.txt.partial` → `.txt` renames
- **Batch processing:** process entire directories in one model-load pass
- **Two output formats:** timestamped `.txt` or structured `.json`
- **CLI and Python API**

## Installation

```bash
# Core only
pip install .

# With Qwen3-ASR (recommended for Portuguese meetings)
pip install ".[qwen]"

# With Whisper
pip install ".[whisper]"

# With all backends
pip install ".[all]"
```

**Requires:** `ffmpeg` on PATH.

## Quick Start

```bash
# Transcribe a single file (default: Qwen3-ASR-1.7B, Portuguese)
transcribe recording.m4a

# Transcribe all files in a directory
transcribe /path/to/meetings/

# Use Whisper large-v3 instead
transcribe recording.m4a --model whisper-large-v3

# Output as JSON with metadata
transcribe recording.m4a --output-format json --output-dir ./transcripts/

# Dry run to see what would be processed
transcribe /mnt/workspace/meetings/incoming/ --dry-run
```

## Supported Models

| Model name | Backend | Size | Notes |
|---|---|---|---|
| `qwen3-1.7b` | Qwen3-ASR (vLLM) | 1.7B | Best quality open-source, recommended |
| `qwen3-0.6b` | Qwen3-ASR (vLLM) | 0.6B | Faster, ~2000x throughput at concurrency 128 |
| `whisper-large-v3` | OpenAI Whisper | 1.5B | Strong multilingual baseline |
| `whisper-large-v2` | OpenAI Whisper | 1.5B | Previous generation |
| `whisper-medium` | OpenAI Whisper | 300M | Faster Whisper option |
| `omnilingual` | Omnilingual ASR | varies | Batch-capable pipeline |

## Output Formats

### `txt` (default)
```
[00:00:00]
Porque o ponto dessa derivação aqui é explicar por que que o PPO colapsa...

[00:02:00]
E aí a gente estava se perguntando o que que isso mudaria...
```

### `json`
```json
{
  "model": "QwenTranscriber",
  "language": "pt",
  "chunks": [
    {"start": 0.0, "end": 120.0, "text": "Porque o ponto..."},
    {"start": 118.0, "end": 238.0, "text": "E aí a gente..."}
  ]
}
```

## Python API

```python
from transcription_utility import TranscriptionPipeline
from transcription_utility.registry import get_transcriber

transcriber = get_transcriber("qwen3-1.7b")
pipeline = TranscriptionPipeline(
    transcriber=transcriber,
    language="pt",
    output_format="json",
    chunk_duration=120,
)
results = pipeline.run("/mnt/workspace/meetings/incoming/")
for r in results:
    print(r.output_path, "success:", r.success)
```

## Meeting Pipeline Integration

This package is used by the automated meeting pipeline on Jarbas:

- Incoming recordings: `/mnt/workspace/meetings/incoming/`
- SLURM job: `/mnt/workspace/meetings/jobs/transcribe_job.sh`
- The job activates the venv and calls `transcribe <dir> --model qwen3-1.7b --language pt`

## CLI Options

```
Usage: transcribe [OPTIONS] INPUT_PATH

  Transcribe audio file(s) using a configurable ASR model.

Options:
  -m, --model TEXT                Model to use  [default: qwen3-1.7b]
  -l, --language TEXT             Language ISO code  [default: pt]
  -o, --output-dir PATH           Output directory
  --output-format [txt|json]      Output format  [default: txt]
  --chunk-duration INTEGER        Chunk duration in seconds  [default: 120]
  --overlap INTEGER               Overlap between chunks  [default: 2]
  --gpu-memory-utilization FLOAT  vLLM GPU memory fraction  [default: 0.9]
  --batch-size INTEGER            Batch size (model-dependent)
  --skip-existing / --no-skip-existing  [default: skip-existing]
  --dry-run                       Print files without running inference
  --help                          Show this message and exit.
```
