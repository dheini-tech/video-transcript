import json
import subprocess
from pathlib import Path
from typing import Callable

_RYZEN_PYTHON = Path(r"C:\Programme_DHI\MiniConda\envs\ryzen-ai-1.7.1\python.exe")
_WORKER = Path(__file__).parent.parent.parent / "npu_worker.py"


class TranscriberNPU:
    def __init__(self, model_name: str, device: str = "npu", on_progress: Callable = None):
        self._model_name = model_name
        self._device = device
        self._on_progress = on_progress or (lambda msg: None)

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[dict]:
        if not _RYZEN_PYTHON.exists():
            raise RuntimeError(f"Python ryzen-ai introuvable : {_RYZEN_PYTHON}")

        label = "NPU" if self._device == "npu" else "PyTorch CPU"
        self._on_progress(("log", f"Transcription {label} '{self._model_name}' en cours..."))

        cmd = [str(_RYZEN_PYTHON), str(_WORKER), str(audio_path), self._model_name]
        if language:
            cmd.append(language)
        cmd.append(self._device)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env={"RYZEN_AI_INSTALLATION_PATH": r"C:\Program Files\RyzenAI\1.7.1",
                 "PATH": r"C:\Programme_DHI\MiniConda\envs\ryzen-ai-1.7.1;C:\Programme_DHI\MiniConda\envs\ryzen-ai-1.7.1\Scripts"},
        )

        segments = []
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "log" in msg:
                self._on_progress(("log", msg["log"]))
            elif "progress" in msg:
                pct = 25.0 + float(msg["progress"]) * 0.24
                self._on_progress(("progress", pct))
            elif "segment_count" in msg:
                self._on_progress(("segment_count", int(msg["segment_count"])))
            elif "segments" in msg:
                segments = msg["segments"]
            elif "error" in msg:
                raise RuntimeError(f"NPU worker error: {msg['error']}")

        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(f"NPU worker exited {proc.returncode}: {stderr[:500]}")

        label = "NPU" if self._device == "npu" else "PyTorch CPU"
        self._on_progress(("log", f"Transcription {label} terminee : {len(segments)} segments."))
        return segments
