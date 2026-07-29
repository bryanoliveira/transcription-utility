# transcription-utility

Audio in, transcription out. One local ASR model, reachable two ways: batch jobs over
a directory, and an OpenAI-compatible HTTP API. Both share the same loaded model.

## Features

- **Model backends:** Qwen3-ASR (vLLM), OpenAI Whisper, Omnilingual ASR
- **One model, two entry points:** a batch job and an API request in the same process
  queue on one engine — never two copies of the weights on the GPU
- **OpenAI-compatible:** `POST /v1/audio/transcriptions`, so existing clients just work
- **Ollama-style lifecycle:** the server loads the model on the first request that
  needs it and releases the GPU after a configurable idle timeout
- **Audio preprocessing:** multi-stream mixing (amix), chunking, corrupt file recovery
- **Idempotent batches:** `.processing` locks + atomic `.partial` → final renames
- **Crash-resilient:** output is flushed after every batch, so an OOM kill or SLURM
  preemption leaves the already-transcribed chunks on disk

## Installation

```bash
pip install ".[qwen]"        # Qwen3-ASR (recommended for Portuguese)
pip install ".[whisper]"     # OpenAI Whisper
pip install ".[server]"      # the HTTP API
pip install ".[all]"         # everything
```

**Requires:** `ffmpeg` on PATH.

## Batch jobs

```bash
# One file, or a whole directory
transcribe run recording.m4a
transcribe run /path/to/audio/

transcribe run recording.m4a --model whisper-large-v3
transcribe run audio/ --output-format json --output-dir ./transcripts/
transcribe run audio/ --dry-run
```

Re-running skips anything already transcribed, so a cron job or SLURM array can just
point at the same directory.

## HTTP service

```bash
transcribe serve --model qwen3-1.7b --host 127.0.0.1 --port 8000 --keep-alive 5
```

Like ollama: the server starts instantly holding no GPU memory, loads the model on
the first request that needs it, and unloads it again after `--keep-alive` idle
minutes. `--keep-alive 0` pins the model in memory once loaded.

| Option | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. `0.0.0.0` to accept requests from the network |
| `--port` | `8000` | Bind port |
| `--keep-alive` | `5` | Idle minutes before unloading. `0` = pin in memory |
| `--preload` | off | Load at startup instead of on first request |
| `--max-upload-mb` | `500` | Reject larger uploads with 413 |

```bash
curl localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.m4a \
  -F language=pt
# {"text": "..."}
```

Any OpenAI client works — point it at the server and the `model` field is ignored
(this server has exactly one model loaded; `GET /v1/models` names it):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
with open("meeting.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/transcriptions` | Transcribe an uploaded file. `response_format`: `json`, `text`, `verbose_json` |
| `GET /v1/models` | The model this server has loaded |
| `POST /batch` | Start a directory batch job on that same model (`path=`) |
| `GET /batch` | Status of the current or last batch job |
| `GET /health` | Liveness, model name, whether weights are loaded, idle seconds |
| `POST /unload` | Drop the model now, without waiting for the keep-alive timer |

`srt` / `vtt` are not offered: chunks are minutes long, which makes for useless
subtitles. Use `verbose_json` and re-segment if you need them.

### Sharing the model

The engine holds one model behind a lock, taken per inference batch. A batch job
started with `POST /batch` and a request arriving mid-job both queue on it, so the
request waits at most one chunk-batch rather than the whole job — and the GPU only
ever holds one copy of the weights.

Weights load lazily on first use. A `transcribe run` over a directory with nothing
new to do exits without ever loading the model.

```bash
curl localhost:8000/health
# {"status":"ok","model":"QwenTranscriber","model_loaded":false,
#  "idle_seconds":0.0,"keep_alive_seconds":300,"batch":{"state":"idle"}}

# Run a directory job on the same server, then watch it
curl -X POST localhost:8000/batch -F path=/data/recordings
curl localhost:8000/batch
```

## Running it as a service

`scripts/install-service.sh` writes a systemd unit and starts it. Defaults to a
`--user` service, so no root is needed:

```bash
./scripts/install-service.sh --port 8000 --model qwen3-1.7b --keep-alive 5
```

Inspect the unit before installing — always worth it for `--system`:

```bash
./scripts/install-service.sh --print --port 8000
```

Machine-wide, bound to the network, running as a dedicated account:

```bash
sudo useradd -r -s /usr/sbin/nologin asr
./scripts/install-service.sh --system --run-as asr --host 0.0.0.0 --keep-alive 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--name` | `transcribe` | Unit name (`transcribe.service`) |
| `--model`, `--language`, `--host`, `--port`, `--keep-alive` | see above | Passed to `transcribe serve` |
| `--exec` | autodetected | Path to the `transcribe` binary |
| `--extra` | — | Extra flags, e.g. `--extra "--max-upload-mb 2000"` |
| `--system` | off | Install to `/etc/systemd/system` instead of the user unit dir |
| `--run-as` | — | `User=` for a system service |
| `--print` | off | Print the unit and exit; install nothing |
| `--uninstall` | off | Stop, disable, and delete the unit |

The script resolves `transcribe` to an absolute path (systemd starts with a bare
environment) and puts `ffmpeg` on the unit's `PATH`. It sets `Restart=on-failure`
and a 600s start timeout so a slow first model load isn't killed. For a `--user`
service it also enables `loginctl enable-linger`, without which the service stops
when you log out.

```bash
systemctl --user status transcribe        # or: sudo systemctl status transcribe
journalctl --user -u transcribe -f
./scripts/install-service.sh --uninstall
```

## Supported models

| Model name | Backend | Notes |
|---|---|---|
| `qwen3-1.7b` | Qwen3-ASR (vLLM) | Best quality, recommended |
| `qwen3-0.6b` | Qwen3-ASR (vLLM) | Faster |
| `whisper-large-v3` | OpenAI Whisper | Strong multilingual baseline |
| `whisper-medium`, `whisper-small`, … | OpenAI Whisper | Smaller/faster |
| `omnilingual` | Omnilingual ASR | `omniASR_LLM_Unlimited_7B_v2`, long-context |

Chunk size and batch size default to what the backend wants (120s / 8 chunks; 7200s
for `omnilingual`). Override with `--chunk-duration` and `--batch-size`.

## Output formats

`txt`:
```
[00:00:00]
Porque o ponto dessa derivação aqui é explicar por que que o PPO colapsa...

[00:02:00]
E aí a gente estava se perguntando o que que isso mudaria...
```

`json`:
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
from transcription_utility import BatchPipeline, TranscriptionEngine, get_transcriber
from transcription_utility.pipeline import transcribe_audio

engine = TranscriptionEngine(get_transcriber("qwen3-1.7b"), keep_alive_minutes=5)
engine.start_idle_reaper()   # optional: unload after 5 idle minutes

# One file
chunks = transcribe_audio(engine, "meeting.m4a", language="pt")

# A directory, written to disk
results = BatchPipeline(engine, language="pt", output_format="json").run("audio/")
for r in results:
    if r.success:
        print(r.output_path)
    elif r.skipped:
        print(f"skipped {r.audio_path.name}: {r.error}")
    else:
        print(f"FAILED {r.audio_path.name}: {r.error}")

engine.close()
```

Share one `TranscriptionEngine` across threads and they share the model.

## Exit codes

`transcribe run` exits `1` only on genuine failures (unreadable audio, ffmpeg or
model errors). Deliberate pass-overs — already transcribed, locked by another run,
or shorter than 5s — are reported as skips and exit `0`, so a stray short clip does
not fail a batch job.

## Tests

```bash
python test_pipeline.py    # needs ffmpeg; no GPU, no model weights
```
