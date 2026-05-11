from pathlib import Path
from typing import Callable
import yt_dlp


class YouTubeDownloader:
    def __init__(self, on_progress: Callable = None):
        self._on_progress = on_progress or (lambda msg: None)

    def download_audio(self, url: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "yt_audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }],
            "quiet": True,
            "progress_hooks": [self._hook],
        }

        self._on_progress(("log", "Téléchargement de l'audio YouTube..."))

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_path = output_dir / "yt_audio.wav"
        if not audio_path.exists():
            raise FileNotFoundError("Le fichier audio YouTube est introuvable après téléchargement.")

        return audio_path

    def _hook(self, d: dict):
        if d["status"] == "downloading":
            percent = d.get("_percent_str", "").strip()
            self._on_progress(("log", f"Téléchargement : {percent}"))
        elif d["status"] == "finished":
            self._on_progress(("log", "Audio téléchargé, conversion WAV en cours..."))
