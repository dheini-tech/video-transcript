import json
from config import SETTINGS_FILE


class Settings:
    def __init__(self):
        self.hf_token: str = ""
        self._load()

    def _load(self):
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                self.hf_token = data.get("hf_token", "")
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        SETTINGS_FILE.write_text(
            json.dumps({"hf_token": self.hf_token}, indent=2),
            encoding="utf-8",
        )
