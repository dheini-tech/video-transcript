# -*- coding: utf-8 -*-
"""Quick benchmark: CPU (faster-whisper) vs DirectML (ONNX) on the test file."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.media.extractor import AudioExtractor

TEST_VIDEO = Path(r"C:\Users\domin\OneDrive\23 - Formation Investissement\03 - LBSS\0703 - Option 4.3 Le volume et l'open interest.mp4")
TEMP = config.TEMP_DIR
MODEL = "small"
LANGUAGE = "fr"


def log(msg):
    print(msg, flush=True)


def extract_audio() -> Path:
    TEMP.mkdir(parents=True, exist_ok=True)
    extractor = AudioExtractor(on_progress=lambda m: None)
    return extractor.extract(TEST_VIDEO, TEMP)


def bench_cpu(audio_path: Path):
    from src.transcription.transcriber import Transcriber
    t = Transcriber(model_name=MODEL, on_progress=lambda m: log(f"  [CPU] {m}"))
    t0 = time.monotonic()
    segs = t.transcribe(audio_path, language=LANGUAGE)
    elapsed = time.monotonic() - t0
    return elapsed, len(segs)


def bench_directml(audio_path: Path):
    from src.transcription.transcriber_onnx import TranscriberONNX
    t = TranscriberONNX(model_name=MODEL, on_progress=lambda m: log(f"  [DML] {m}"))
    t0 = time.monotonic()
    segs = t.transcribe(audio_path, language=LANGUAGE)
    elapsed = time.monotonic() - t0
    return elapsed, len(segs)


if __name__ == "__main__":
    log(f"Extraction audio depuis : {TEST_VIDEO.name}")
    audio = extract_audio()
    log(f"Audio extrait : {audio}")

    log("\n=== CPU (faster-whisper) ===")
    cpu_time, cpu_segs = bench_cpu(audio)
    log(f"  Duree : {cpu_time:.1f}s | Segments : {cpu_segs}")

    log("\n=== DirectML (ONNX) ===")
    dml_time, dml_segs = bench_directml(audio)
    log(f"  Duree : {dml_time:.1f}s | Segments : {dml_segs}")

    log(f"\n=== Resultat ===")
    if dml_time < cpu_time:
        log(f"  DirectML est {cpu_time/dml_time:.1f}x plus rapide que CPU")
    else:
        log(f"  CPU est {dml_time/cpu_time:.1f}x plus rapide que DirectML")
