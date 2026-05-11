from pathlib import Path
from typing import Callable
from faster_whisper import WhisperModel
from config import WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_CPU_THREADS, WHISPER_NUM_WORKERS, WHISPER_BEAM_SIZE, WHISPER_VAD_FILTER


class Transcriber:
    def __init__(self, model_name: str, on_progress: Callable = None):
        self._model_name = model_name
        self._on_progress = on_progress or (lambda msg: None)
        self._model: WhisperModel | None = None

    def _load_model(self):
        if self._model is None:
            self._on_progress(("log", f"Chargement du modèle Whisper '{self._model_name}' (première fois : téléchargement possible)..."))
            self._model = WhisperModel(
                self._model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=WHISPER_CPU_THREADS,
                num_workers=WHISPER_NUM_WORKERS,
            )
            self._on_progress(("log", "Modèle Whisper chargé."))

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[dict]:
        self._load_model()
        self._on_progress(("log", "Transcription en cours..."))

        segments_gen, info = self._model.transcribe(
            str(audio_path),
            beam_size=WHISPER_BEAM_SIZE,
            language=language,
            word_timestamps=False,
            vad_filter=WHISPER_VAD_FILTER,
        )

        lang_info = f"{info.language} (confiance : {info.language_probability:.0%})" if not language else language
        self._on_progress(("log", f"Langue : {lang_info}"))

        result = []
        duration = max(info.duration or 0, 1.0)
        for seg in segments_gen:
            result.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            pct = 25.0 + (min(seg.end, duration) / duration) * 24.0  # 25 % → 49 %
            self._on_progress(("progress", pct))
            self._on_progress(("segment_count", len(result)))

        self._on_progress(("log", f"Transcription terminée : {len(result)} segments."))
        return result
