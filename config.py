import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent


def _ensure_ffmpeg_in_path() -> None:
    if shutil.which("ffmpeg"):
        return
    candidates = [
        # WinGet portable install (Gyan full_build)
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
        # Chemin partagé dans le projet
        BASE_DIR / "ffmpeg-shared" / "bin",
        BASE_DIR / "ffmpeg" / "bin",
    ]
    for root in candidates:
        if not root.exists():
            continue
        # Cherche récursivement ffmpeg.exe sous ce dossier (max 4 niveaux)
        for exe in root.rglob("ffmpeg.exe"):
            os.environ["PATH"] = (
                str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
            )
            return


_ensure_ffmpeg_in_path()
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
TEMP_DIR = BASE_DIR / ".tmp"
SETTINGS_FILE = BASE_DIR / "settings.json"

WHISPER_MODEL_DEFAULT = "large-v3"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_CPU_THREADS = os.cpu_count() - 4  # tous les cœurs logiques
WHISPER_NUM_WORKERS = 2                   # décodage de chunks en parallèle
WHISPER_BEAM_SIZE   = 2                   # 1 = greedy (rapide), 5 = précis (lent)
WHISPER_VAD_FILTER  = True                # ignore les silences → gain 20-50 %

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

WHISPER_LANGUAGES = [
    "Auto",
    "fr — Français",
    "en — English",
    "es — Español",
    "de — Deutsch",
    "it — Italiano",
    "pt — Português",
    "nl — Nederlands",
    "pl — Polski",
    "ru — Русский",
    "ar — العربية",
    "zh — 中文",
    "ja — 日本語",
]

SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
}
