from pathlib import Path
from typing import Callable

import soundfile as sf
import torch
from pyannote.audio import Pipeline


class Diarizer:
    def __init__(self, hf_token: str, on_progress: Callable = None):
        self._hf_token = hf_token
        self._on_progress = on_progress or (lambda msg: None)
        self._pipeline: Pipeline | None = None

    def _load_pipeline(self):
        if self._pipeline is None:
            self._on_progress(("log", "Chargement du modèle de diarisation pyannote (première fois : téléchargement possible)..."))
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self._hf_token,
            )
            self._on_progress(("log", "Modèle de diarisation chargé."))

    def diarize(self, audio_path: Path) -> list[dict]:
        self._load_pipeline()
        self._on_progress(("log", "Diarisation en cours (détection des locuteurs)..."))

        # Préchargement audio via soundfile pour éviter la dépendance torchcodec
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform_tensor = torch.from_numpy(waveform.T)  # (channels, time)
        audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

        output = self._pipeline(audio_input)
        # pyannote >= 3.3 retourne un DiarizeOutput, les versions antérieures une Annotation directe
        diarization = output.diarization if hasattr(output, "diarization") else output

        speakers = [
            {"start": turn.start, "end": turn.end, "speaker": label}
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]

        unique = len({s["speaker"] for s in speakers})
        self._on_progress(("log", f"Diarisation terminée : {unique} locuteur(s) détecté(s)."))
        return speakers
