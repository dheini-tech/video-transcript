import shutil
from pathlib import Path
from typing import Callable

from config import TRANSCRIPTS_DIR, TEMP_DIR
from src.media.downloader import YouTubeDownloader
from src.media.extractor import AudioExtractor
from src.transcription.transcriber import Transcriber
from src.transcription.diarizer import Diarizer
from src.output.formatter import TranscriptFormatter


class TranscriptionPipeline:
    def __init__(self, model: str, hf_token: str, on_progress: Callable = None):
        self._on_progress = on_progress or (lambda msg: None)
        self._transcriber = Transcriber(model_name=model, on_progress=on_progress)
        self._diarizer = Diarizer(hf_token=hf_token, on_progress=on_progress)
        self._formatter = TranscriptFormatter()

    def run(self, source_type: str, source: str, language: str | None = None) -> Path:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            self._on_progress(("step", 1))
            audio_path = self._prepare_audio(source_type, source)

            self._on_progress(("step", 2))
            segments = self._transcriber.transcribe(audio_path, language=language)

            self._on_progress(("step", 3))
            speakers = self._diarizer.diarize(audio_path)

            self._on_progress(("step", 4))
            merged = self._merge_speaker_labels(segments, speakers)
            transcript_path = self._save(merged, audio_path.stem)
            self._on_progress(("transcript", self._formatter.to_string(merged)))
            return transcript_path
        finally:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)

    def _prepare_audio(self, source_type: str, source: str) -> Path:
        if source_type == "youtube":
            return YouTubeDownloader(on_progress=self._on_progress).download_audio(source, TEMP_DIR)
        return AudioExtractor(on_progress=self._on_progress).extract(Path(source), TEMP_DIR)

    def _merge_speaker_labels(self, segments: list[dict], speakers: list[dict]) -> list[dict]:
        merged = []
        for seg in segments:
            label = self._best_speaker(seg["start"], seg["end"], speakers)
            merged.append({**seg, "speaker": label})
        return merged

    def _best_speaker(self, start: float, end: float, speakers: list[dict]) -> str:
        best_label = "INCONNU"
        best_overlap = 0.0
        for sp in speakers:
            overlap = min(end, sp["end"]) - max(start, sp["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = sp["speaker"]
        return best_label

    def _save(self, merged: list[dict], stem: str) -> Path:
        content = self._formatter.to_string(merged)
        out_path = TRANSCRIPTS_DIR / f"{stem}_transcript.txt"
        out_path.write_text(content, encoding="utf-8")
        self._on_progress(("log", f"Fichier sauvegardé : {out_path}"))
        return out_path
