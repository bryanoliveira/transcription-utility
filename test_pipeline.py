"""Self-check: python test_pipeline.py. Needs ffmpeg, no GPU, no model weights.

Server checks are skipped if the [server] extra isn't installed.
"""
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from transcription_utility import BatchPipeline, TranscriptionEngine, get_transcriber
from transcription_utility.audio import CHUNK_DIR_PREFIX, sweep_stale_chunk_dirs
from transcription_utility.models import BaseTranscriber


class Stub(BaseTranscriber):
    default_batch_size = 2

    def __init__(self, fail_after=None, peek=None, delay=0.0):
        self.calls, self.fail_after, self.peek, self.delay = [], fail_after, peek, delay
        self.loads = 0
        self.concurrent = 0
        self.max_concurrent = 0

    def load(self):
        self.loads += 1

    def unload(self):
        pass

    def transcribe(self, paths, language="pt"):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.calls.append(len(paths))
            if self.peek is not None and len(self.calls) == 2:
                # What's on disk from batch 1, before batch 2 is written?
                self.peek.append(
                    self.peek[0].read_text() if self.peek[0].exists() else ""
                )
            if self.fail_after is not None and len(self.calls) > self.fail_after:
                raise RuntimeError("boom")
            time.sleep(self.delay)
            return [f"t{Path(p).stem}" for p in paths]
        finally:
            self.concurrent -= 1


def audio(path, seconds, streams=1):
    args = []
    for i in range(streams):
        args += ["-f", "lavfi", "-t", str(seconds), "-i", f"sine=f={300 + i * 100}"]
    for i in range(streams):
        args += ["-map", str(i)]
    subprocess.run(["ffmpeg", "-y", *args, str(path)], capture_output=True, check=True)


def chunkdirs():
    root = Path(tempfile.gettempdir())
    return {p for p in root.glob(f"{CHUNK_DIR_PREFIX}*")}


def check_batch(work):
    before = chunkdirs()
    stub = Stub()
    res = {r.audio_path.name: r
           for r in BatchPipeline(TranscriptionEngine(stub), chunk_duration=120).run(work)}

    assert not chunkdirs() - before, "chunk dirs leaked"
    assert res["long.wav"].success and res["multi.m4a"].success
    assert res["tiny.wav"].skipped and not res["tiny.wav"].success, "short file must skip"
    assert not res["broken.m4a"].skipped, "unreadable file must fail, not skip"
    assert stub.calls == [2, 1, 1], f"batching wrong: {stub.calls}"
    assert (work / "long.txt").read_text().count("[") == 3
    assert not list(work.glob("*.partial")) and not list(work.glob("*.processing"))

    # Completed files are not re-inferred.
    stub2 = Stub()
    BatchPipeline(TranscriptionEngine(stub2), chunk_duration=120).run(work)
    assert stub2.calls == [], f"re-transcribed completed files: {stub2.calls}"

    # Nothing to transcribe => the model is never loaded.
    assert stub2.loads == 0, "loaded the model with no work to do"

    # Output is flushed per batch, so a hard kill keeps finished chunks.
    for p in work.glob("*.txt"):
        p.unlink()
    peek = [work / "long.txt.partial"]
    BatchPipeline(TranscriptionEngine(Stub(peek=peek)), chunk_duration=120).run(work)
    assert peek[1].count("[") == 2, f"batch 1 not on disk mid-run: {peek[1]!r}"

    # Failure mid-file: no output, no lock, no partial, no temp left.
    for p in work.glob("*.txt"):
        p.unlink()
    before = chunkdirs()
    bad = BatchPipeline(TranscriptionEngine(Stub(fail_after=1)), chunk_duration=120).run(work)
    assert not any(r.success for r in bad if r.audio_path.name == "long.wav")
    assert not (work / "long.txt").exists() and not list(work.glob("*.partial"))
    assert not list(work.glob("*.processing")) and not chunkdirs() - before

    # JSON stays parseable.
    BatchPipeline(TranscriptionEngine(Stub()), output_format="json",
                  chunk_duration=120).run(work)
    data = json.loads((work / "long.json").read_text())
    assert len(data["chunks"]) == 3 and data["language"] == "pt"
    for p in list(work.glob("*.json")) + list(work.glob("*.txt")):
        p.unlink()


def check_engine():
    # One model, loaded once, no matter how many callers.
    stub = Stub(delay=0.05)
    engine = TranscriptionEngine(stub)
    assert not engine.loaded, "engine must load lazily"

    def hammer():
        for _ in range(5):
            engine.transcribe(["/x.wav"], language="pt")

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert stub.loads == 1, f"model loaded {stub.loads} times, expected 1"
    assert stub.max_concurrent == 1, "engine let two callers into the model at once"
    assert len(stub.calls) == 20

    # A backend dropping audio must raise, not silently truncate.
    class Liar(Stub):
        def transcribe(self, paths, language="pt"):
            return ["only one"]

    try:
        TranscriptionEngine(Liar()).transcribe(["/a.wav", "/b.wav"], language="pt")
        raise AssertionError("short transcript count must raise")
    except RuntimeError:
        pass


def check_keep_alive():
    # Loads on demand, unloads once idle past the keep-alive window.
    stub = Stub()
    engine = TranscriptionEngine(stub, keep_alive_minutes=1.5 / 60)  # 1.5s
    engine.start_idle_reaper()
    assert not engine.loaded, "must not load before there is work"

    engine.transcribe(["/x.wav"], language="pt")
    assert engine.loaded and stub.loads == 1

    for _ in range(100):  # reaper polls; give it room
        if not engine.loaded:
            break
        time.sleep(0.1)
    assert not engine.loaded, "model still resident past keep-alive"

    # A later request reloads it.
    engine.transcribe(["/x.wav"], language="pt")
    assert engine.loaded and stub.loads == 2
    engine.close()
    assert not engine.loaded, "close() must release the model"

    # Work in progress must never be unloaded underneath a caller.
    slow = Stub(delay=0.6)
    pinned_out = TranscriptionEngine(slow, keep_alive_minutes=0.2 / 60)  # 0.2s
    pinned_out.start_idle_reaper()
    for _ in range(4):
        pinned_out.transcribe(["/x.wav"], language="pt")
    assert slow.loads == 1, f"reaper unloaded mid-job ({slow.loads} loads)"
    pinned_out.close()

    # keep_alive=0 pins: no reaper, model stays put.
    pinned = TranscriptionEngine(Stub(), keep_alive_minutes=0)
    pinned.start_idle_reaper()
    pinned.transcribe(["/x.wav"], language="pt")
    time.sleep(1.0)
    assert pinned.loaded, "keep_alive=0 must pin the model"
    pinned.close()


def check_stale_sweep():
    old = Path(tempfile.mkdtemp(prefix=CHUNK_DIR_PREFIX))
    (old / "chunk_000.wav").write_bytes(b"x")
    fresh = Path(tempfile.mkdtemp(prefix=CHUNK_DIR_PREFIX))
    import os
    ancient = time.time() - 10 * 3600
    os.utime(old, (ancient, ancient))

    sweep_stale_chunk_dirs()
    assert not old.exists(), "stale chunk dir not swept"
    assert fresh.exists(), "swept a live chunk dir"
    fresh.rmdir()


def check_server(work):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("skip: server extra not installed")
        return

    from transcription_utility.server import create_app

    stub = Stub()
    engine = TranscriptionEngine(stub)
    client = TestClient(create_app(engine, default_language="pt"))

    assert client.get("/v1/models").json()["data"][0]["id"] == "Stub"
    health = client.get("/health").json()
    assert health["model_loaded"] is False, "server must not load a model at startup"
    assert health["keep_alive_seconds"] == 300, health  # 5min default

    with open(work / "long.wav", "rb") as f:
        r = client.post("/v1/audio/transcriptions", files={"file": ("long.wav", f)})
    assert r.status_code == 200, r.text
    assert r.json()["text"], "empty transcription"

    with open(work / "long.wav", "rb") as f:
        r = client.post("/v1/audio/transcriptions",
                        files={"file": ("long.wav", f)},
                        data={"response_format": "verbose_json"})
    body = r.json()
    assert len(body["segments"]) == 3 and body["segments"][0]["start"] == 0.0

    with open(work / "long.wav", "rb") as f:
        r = client.post("/v1/audio/transcriptions",
                        files={"file": ("long.wav", f)},
                        data={"response_format": "text"})
    assert r.headers["content-type"].startswith("text/plain") and r.text

    with open(work / "long.wav", "rb") as f:
        r = client.post("/v1/audio/transcriptions",
                        files={"file": ("long.wav", f)},
                        data={"response_format": "srt"})
    assert r.status_code == 400, "unsupported format must 400"

    with open(work / "tiny.wav", "rb") as f:
        r = client.post("/v1/audio/transcriptions", files={"file": ("tiny.wav", f)})
    assert r.status_code == 400, "too-short audio must 400"

    # /unload drops the model without waiting for the timer; the next request
    # brings it back.
    assert client.get("/health").json()["model_loaded"] is True
    assert client.post("/unload").json()["unloaded"] is True
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/unload").json()["unloaded"] is False, "second unload is a no-op"
    with open(work / "long.wav", "rb") as f:
        assert client.post("/v1/audio/transcriptions",
                           files={"file": ("long.wav", f)}).status_code == 200
    assert client.get("/health").json()["model_loaded"] is True

    # Uploads are capped — this endpoint is a network trust boundary.
    small = TestClient(create_app(TranscriptionEngine(Stub()), max_upload_mb=1))
    r = small.post("/v1/audio/transcriptions",
                   files={"file": ("big.wav", b"x" * (2 * 1024 * 1024))})
    assert r.status_code == 413, f"oversized upload must 413, got {r.status_code}"

    # The requirement: a batch job and a request in flight together share the
    # one loaded model. Slow the model down so the overlap is real, not luck.
    for p in list(work.glob("*.txt")) + list(work.glob("*.json")):
        p.unlink()
    stub.delay = 0.25
    loads_before = stub.loads
    assert engine.loaded, "model should already be resident from the requests above"

    assert client.post("/batch", data={"path": str(work)}).status_code == 200
    assert client.get("/batch").json()["state"] == "running"

    with open(work / "long.wav", "rb") as f:
        r = client.post("/v1/audio/transcriptions", files={"file": ("long.wav", f)})
    assert r.status_code == 200, r.text
    assert r.json()["text"], "request served while a batch job was running"
    # Still running => the request really did overlap the batch job.
    assert client.get("/batch").json()["state"] == "running", "batch finished too fast to prove overlap"

    for _ in range(200):
        if client.get("/batch").json()["state"] != "running":
            break
        time.sleep(0.05)
    assert client.get("/batch").json()["state"] == "done", client.get("/batch").json()
    assert stub.loads == loads_before, "batch job loaded a second copy of the model"
    assert stub.max_concurrent == 1, "batch and request hit the model concurrently"
    stub.delay = 0.0

    # Server temp uploads are not left behind.
    assert not list(Path(tempfile.gettempdir()).glob(f"{CHUNK_DIR_PREFIX}*/upload*"))


def main():
    work = Path(tempfile.mkdtemp())
    audio(work / "long.wav", 300)      # 3 chunks at 120s/2s overlap
    audio(work / "multi.m4a", 30, 2)   # exercises the amix path
    audio(work / "tiny.wav", 3)        # under MIN_DURATION
    (work / "broken.m4a").write_bytes(b"not audio")

    check_batch(work)
    check_engine()
    check_keep_alive()
    check_stale_sweep()
    check_server(work)

    assert get_transcriber("omnilingual").model_card == "omniASR_LLM_Unlimited_7B_v2"
    assert get_transcriber("omnilingual", model_card="x").model_card == "x"
    try:
        get_transcriber("nope")
        raise AssertionError("unknown model must raise")
    except ValueError:
        pass

    subprocess.run(["rm", "-rf", str(work)])
    print("ok")


if __name__ == "__main__":
    main()
