import subprocess
from pathlib import Path
from typing import Callable


class AudioExtractor:
    def __init__(self, on_progress: Callable = None):
        self._on_progress = on_progress or (lambda msg: None)

    def extract(self, video_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / (video_path.stem + ".wav")

        self._on_progress(("log", f"Extraction audio : {video_path.name}"))

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ac", "1",       # mono
            "-ar", "16000",   # 16 kHz optimal pour Whisper
            "-vn",
            str(audio_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg est introuvable. Installez-le et ajoutez-le au PATH système.\n"
                               "Téléchargement : https://ffmpeg.org/download.html")
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg a échoué :\n{result.stderr}")

        self._on_progress(("log", "Extraction audio terminée."))
        return audio_path
