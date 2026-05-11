import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import SUPPORTED_MEDIA_EXTENSIONS, WHISPER_MODEL_DEFAULT, WHISPER_MODELS, WHISPER_LANGUAGES
from src.gui.widgets import LogWidget
from src.settings import Settings
from src.transcription.pipeline import TranscriptionPipeline


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Transcript")
        self.root.geometry("800x620")
        self.root.resizable(True, True)

        self._settings = Settings()
        self._queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._step_start: float = 0.0
        self._current_step: int = 0
        self._seg_count: int = 0
        self._spinner_idx: int = 0

        self._build_ui()

    # ------------------------------------------------------------------ build

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        self._build_source_frame()
        self._build_options_frame()
        self._build_action_frame()
        self._build_log_frame()

    def _build_source_frame(self):
        frame = ttk.LabelFrame(self.root, text="Source", padding=10)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        frame.columnconfigure(1, weight=1)

        self._source_type = tk.StringVar(value="local")
        ttk.Radiobutton(frame, text="Fichier local", variable=self._source_type,
                        value="local", command=self._toggle_source).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="URL YouTube", variable=self._source_type,
                        value="youtube", command=self._toggle_source).grid(row=0, column=1, sticky="w")

        # --- local row
        self._local_frame = ttk.Frame(frame)
        self._local_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self._local_frame.columnconfigure(0, weight=1)

        self._file_path_var = tk.StringVar()
        ttk.Entry(self._local_frame, textvariable=self._file_path_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(self._local_frame, text="Parcourir…", command=self._browse_file).grid(row=0, column=1)

        # --- youtube row (hidden by default)
        self._youtube_frame = ttk.Frame(frame)
        self._youtube_frame.columnconfigure(0, weight=1)
        self._url_var = tk.StringVar()
        ttk.Label(self._youtube_frame, text="URL :").grid(row=0, column=0, sticky="w")
        ttk.Entry(self._youtube_frame, textvariable=self._url_var).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self._youtube_frame.columnconfigure(1, weight=1)

        self._toggle_source()

    def _toggle_source(self):
        if self._source_type.get() == "local":
            self._local_frame.grid()
            self._youtube_frame.grid_remove()
        else:
            self._local_frame.grid_remove()
            self._youtube_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def _browse_file(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_MEDIA_EXTENSIONS))
        path = filedialog.askopenfilename(
            title="Choisir un fichier vidéo ou audio",
            filetypes=[("Médias", exts), ("Tous les fichiers", "*.*")],
        )
        if path:
            self._file_path_var.set(path)

    def _build_options_frame(self):
        frame = ttk.LabelFrame(self.root, text="Options", padding=10)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        ttk.Label(frame, text="Modèle Whisper :").grid(row=0, column=0, sticky="w")
        self._model_var = tk.StringVar(value=WHISPER_MODEL_DEFAULT)
        ttk.Combobox(frame, textvariable=self._model_var, values=WHISPER_MODELS,
                     state="readonly", width=14).grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Label(frame, text="Langue :").grid(row=0, column=2, sticky="w")
        self._lang_var = tk.StringVar(value="Auto")
        ttk.Combobox(frame, textvariable=self._lang_var, values=WHISPER_LANGUAGES,
                     state="readonly", width=14).grid(row=0, column=3, sticky="w", padx=(5, 20))

        ttk.Label(frame, text="Token HuggingFace :").grid(row=0, column=4, sticky="w")
        self._hf_token_var = tk.StringVar(value=self._settings.hf_token)
        ttk.Entry(frame, textvariable=self._hf_token_var, show="*", width=32).grid(row=0, column=5, sticky="ew", padx=5)
        frame.columnconfigure(5, weight=1)

        ttk.Button(frame, text="Sauvegarder", command=self._save_settings).grid(row=0, column=6)

    def _save_settings(self):
        self._settings.hf_token = self._hf_token_var.get().strip()
        self._settings.save()
        messagebox.showinfo("Paramètres", "Token HuggingFace sauvegardé.")

    def _build_action_frame(self):
        frame = ttk.Frame(self.root, padding=(10, 5))
        frame.grid(row=2, column=0, sticky="ew", padx=10)
        frame.columnconfigure(3, weight=1)

        self._start_btn = ttk.Button(frame, text="▶  Démarrer la transcription", command=self._start)
        self._start_btn.grid(row=0, column=0, padx=(0, 8))

        self._stop_btn = ttk.Button(frame, text="■  Arrêter", command=self._stop, state="disabled")
        self._stop_btn.grid(row=0, column=1, padx=(0, 12))

        self._progress = ttk.Progressbar(frame, mode="determinate", maximum=100, length=200)
        self._progress.grid(row=0, column=3, sticky="ew")

        self._spinner_var = tk.StringVar(value="")
        tk.Label(frame, textvariable=self._spinner_var, foreground="#0078d4",
                 font=("Consolas", 10), width=2, anchor="center").grid(row=0, column=4, sticky="w", padx=(8, 2))

        self._step_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._step_var, foreground="#0078d4",
                  width=46, anchor="w").grid(row=0, column=5, sticky="w")

        self._status_var = tk.StringVar(value="Prêt.")
        ttk.Label(frame, textvariable=self._status_var, foreground="gray").grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))

    def _build_log_frame(self):
        frame = ttk.LabelFrame(self.root, text="Journal / Transcription", padding=10)
        frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(5, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._log = LogWidget(frame)
        self._log.grid(row=0, column=0, sticky="nsew")

        ttk.Button(frame, text="Copier tout", command=self._copy_log).grid(
            row=1, column=0, sticky="e", pady=(6, 0))

    # ------------------------------------------------------------------ actions

    def _copy_log(self):
        content = self._log.get_content()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)

    def _start(self):
        source_type = self._source_type.get()
        source = (self._file_path_var.get() if source_type == "local" else self._url_var.get()).strip()

        if not source:
            messagebox.showwarning("Source manquante", "Veuillez indiquer un fichier ou une URL YouTube.")
            return

        hf_token = self._hf_token_var.get().strip()
        if not hf_token:
            messagebox.showwarning("Token manquant", "Veuillez saisir votre token HuggingFace pour la diarisation.")
            return

        self._log.clear()
        self._set_running(True)

        lang = self._lang_var.get()
        language = None if lang == "Auto" else lang.split(" — ")[0]

        threading.Thread(
            target=self._run_pipeline,
            args=(source_type, source, self._model_var.get(), hf_token, language),
            daemon=True,
        ).start()
        self._poll_queue()

    def _run_pipeline(self, source_type: str, source: str, model: str, hf_token: str, language: str | None):
        pipeline = TranscriptionPipeline(model=model, hf_token=hf_token, on_progress=self._queue.put)
        try:
            path = pipeline.run(source_type=source_type, source=source, language=language)
            self._queue.put(("done", str(path)))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    _STEP_NAMES = {1: "Extraction audio", 2: "Transcription", 3: "Diarisation", 4: "Finalisation"}
    _STEP_BASE  = {1: 0, 2: 25, 3: 50, 4: 75}
    _SPINNER    = ['—', '\\', '|', '/']

    def _tick(self):
        if not self._is_running:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER)
        self._spinner_var.set(self._SPINNER[self._spinner_idx])
        elapsed = time.monotonic() - self._step_start
        mins, secs = divmod(int(elapsed), 60)
        name = self._STEP_NAMES.get(self._current_step, "")
        seg = f"  ·  {self._seg_count} seg." if self._current_step == 2 and self._seg_count else ""
        self._step_var.set(f"Étape {self._current_step}/4  ·  {name}  ·  {mins}:{secs:02d}{seg}")
        self.root.after(500, self._tick)

    def _poll_queue(self):
        try:
            while True:
                kind, content = self._queue.get_nowait()
                if kind == "log":
                    self._log.append(content)
                    self._status_var.set(content)
                elif kind == "step":
                    self._current_step = int(content)
                    self._step_start = time.monotonic()
                    self._seg_count = 0
                    self._progress["value"] = self._STEP_BASE.get(self._current_step, 0)
                elif kind == "progress":
                    self._progress["value"] = float(content)
                elif kind == "segment_count":
                    self._seg_count = int(content)
                elif kind == "transcript":
                    self._log.append("\n--- TRANSCRIPTION ---\n")
                    self._log.append(content)
                elif kind == "done":
                    self._progress["value"] = 100
                    self._spinner_var.set("✓")
                    self._step_var.set("Terminé")
                    self._set_running(False)
                    self._status_var.set(f"Terminé — {content}")
                    if messagebox.askyesno(
                        "Transcription terminée",
                        f"Fichier sauvegardé :\n{content}\n\nOuvrir le fichier ?",
                    ):
                        os.startfile(content)
                    return
                elif kind == "error":
                    self._set_running(False)
                    self._status_var.set("Erreur.")
                    messagebox.showerror("Erreur", content)
                    return
        except queue.Empty:
            pass

        if self._is_running:
            self.root.after(100, self._poll_queue)

    def _stop(self):
        self._is_running = False
        self._set_running(False)
        self._status_var.set("Arrêté par l'utilisateur.")

    def _set_running(self, running: bool):
        self._is_running = running
        self._start_btn.config(state="disabled" if running else "normal")
        self._stop_btn.config(state="normal" if running else "disabled")
        if running:
            self._progress["value"] = 0
            self._current_step = 0
            self._seg_count = 0
            self._spinner_idx = 0
            self._step_start = time.monotonic()
            self._spinner_var.set("")
            self._step_var.set("Démarrage…")
            self.root.after(500, self._tick)
        elif self._progress["value"] < 100:
            self._spinner_var.set("")
            self._step_var.set("")

    # ------------------------------------------------------------------ loop

    def run(self):
        self.root.mainloop()
