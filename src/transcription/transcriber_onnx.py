from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

_HF_MODEL_MAP = {
    "tiny":     "openai/whisper-tiny",
    "base":     "openai/whisper-base",
    "small":    "openai/whisper-small",
    "medium":   "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}

_PROVIDER_ORDER = ["DmlExecutionProvider", "CPUExecutionProvider"]


class TranscriberONNX:
    def __init__(self, model_name: str, on_progress: Callable = None):
        self._model_name = model_name
        self._on_progress = on_progress or (lambda msg: None)
        self._pipe = None
        self._provider_used: str = ""

    def _load_model(self):
        if self._pipe is not None:
            return

        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoProcessor, pipeline

        hf_name = _HF_MODEL_MAP.get(self._model_name, "openai/whisper-large-v3")

        for provider in _PROVIDER_ORDER:
            try:
                self._on_progress(("log", f"Chargement ONNX '{self._model_name}' avec {provider}…"))
                model = ORTModelForSpeechSeq2Seq.from_pretrained(
                    hf_name,
                    export=True,
                    provider=provider,
                )
                processor = AutoProcessor.from_pretrained(hf_name)
                self._pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=30,
                    return_timestamps=True,
                )
                self._provider_used = provider
                self._on_progress(("log", f"Modèle ONNX chargé ({provider})."))
                return
            except Exception as exc:
                self._on_progress(("log", f"  >> {provider} echec : {exc}"))

        raise RuntimeError("Impossible de charger le modèle ONNX avec tous les providers disponibles.")

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[dict]:
        self._load_model()
        self._on_progress(("log", "Transcription ONNX en cours…"))

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        gen_kwargs = {}
        if language:
            gen_kwargs["language"] = language

        output = self._pipe(
            {"array": audio, "sampling_rate": sr},
            generate_kwargs=gen_kwargs if gen_kwargs else None,
        )

        chunks = output.get("chunks", [])
        result = []
        total = len(chunks) or 1
        for i, chunk in enumerate(chunks):
            ts = chunk.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
            start = ts[0] if ts[0] is not None else 0.0
            end   = ts[1] if ts[1] is not None else start
            text  = chunk.get("text", "").strip()
            if text:
                result.append({"start": start, "end": end, "text": text})
            pct = 25.0 + ((i + 1) / total) * 24.0
            self._on_progress(("progress", pct))
            self._on_progress(("segment_count", len(result)))

        self._on_progress(("log", f"Transcription ONNX terminée : {len(result)} segments."))
        return result
