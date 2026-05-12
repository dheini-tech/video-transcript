# Video Transcript — Guide Claude Code

## Lancer l'application

```powershell
.\.venv\Scripts\python.exe main.py
```

## Structure du projet

```
main.py                          # Point d'entrée — lance App()
config.py                        # Constantes globales (chemins, modèles, moteurs)
src/
  gui/
    app.py                       # Interface tkinter principale
    widgets.py                   # LogWidget (zone texte scrollable)
  media/
    extractor.py                 # FFmpeg : extrait audio WAV 16kHz mono
    downloader.py                # yt-dlp : télécharge audio YouTube
  transcription/
    pipeline.py                  # Orchestrateur des 4 étapes
    transcriber.py               # Whisper CPU via faster-whisper
    transcriber_onnx.py          # Whisper GPU via onnxruntime-directml (DirectML)
    diarizer.py                  # Identification des locuteurs via pyannote
  output/
    formatter.py                 # Mise en forme texte du transcript
  settings.py                    # Lecture/écriture settings.json (token HF)
transcripts/                     # Fichiers .txt générés (gitignore)
.tmp/                            # Audio temporaire (supprimé après chaque run)
```

## Environnement Python

- Python 3.14 (venv `.venv`)
- `faster-whisper` — transcription CPU (CTranslate2)
- `pyannote.audio` — diarisation (identification locuteurs)
- `onnxruntime-directml 1.24.4` — accélération GPU AMD/Intel via DirectX 12
- `optimum 2.1.0` — installé mais `optimum.onnxruntime` retiré dans cette version (non utilisé)
- `yt-dlp` — téléchargement YouTube
- `soundfile` — chargement audio (évite la dépendance torchcodec)

## Dépendances critiques

### FFmpeg
Détecté automatiquement au démarrage via `config._ensure_ffmpeg_in_path()`.
Cherche dans : WinGet packages, `ffmpeg-shared/bin`, `ffmpeg/bin`.

### Token HuggingFace
Requis pour pyannote (diarisation). Saisi dans l'interface, sauvegardé dans `settings.json`.
Trois modèles gated à autoriser manuellement sur huggingface.co :
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-community-1`

### onnxruntime-directml
**Attention** : `optimum` et d'autres packages reinstallent `onnxruntime` (CPU-only) et écrasent directml.
Après tout `pip install optimum*`, toujours vérifier et réinstaller :
```powershell
.\.venv\Scripts\pip uninstall onnxruntime -y
.\.venv\Scripts\pip install onnxruntime-directml --force-reinstall --no-deps
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Doit afficher : ['DmlExecutionProvider', 'CPUExecutionProvider']
```

## Moteurs de transcription

Benchmarks mesurés sur ~4 min d'audio, modèle `small`, fichier WAV 16kHz mono.

| Moteur (GUI) | Classe | Backend | Durée mesurée | Notes |
|---|---|---|---|---|
| CPU | `Transcriber` | faster-whisper / CTranslate2 | ~60s | Beam search, meilleure qualité |
| Rapide (PyTorch CPU) | `TranscriberNPU` device=cpu | npu_worker.py → PyTorch HuggingFace | ~41s | Via conda ryzen-ai-1.7.1 |
| NPU (Ryzen AI) | `TranscriberNPU` device=npu | npu_worker.py → VitisAI ONNX AMD | ~50s (cache chaud) | ~600s premier run (compilation) |
| DirectML (GPU intégré) | `TranscriberONNX` | onnxruntime-directml + Xenova ONNX | ~180s | Pas de KV-cache → plus lent que CPU |

### Notes DirectML
- Utilise les modèles `Xenova/whisper-*` (ONNX float32 standard, pas les modèles AMD NPU quantifiés)
- Modèles stockés dans `.onnx_cache/` (gitignore)
- Sans KV-cache : O(n²) tokens → lent. Amélioration possible avec `decoder_model_merged.onnx`
- Le decoder bascule automatiquement en CPU si DML plante en cours de séquence

### Notes NPU
- Utilise `amd/whisper-*-onnx-npu` (UINT8 quantifié pour VitisAI, incompatible DML)
- Premier run : ~10 min de compilation NPU → cache dans `.npu_cache/`
- Python subprocess via `C:\Programme_DHI\MiniConda\envs\ryzen-ai-1.7.1\python.exe`

## Pipeline de traitement (4 étapes)

1. **Extraction audio** — FFmpeg → WAV 16kHz mono dans `.tmp/`
2. **Transcription** — Whisper (CPU ou DirectML) → liste de segments `{start, end, text}`
3. **Diarisation** — pyannote → liste `{start, end, speaker}`
4. **Fusion + sauvegarde** — segments + locuteurs → `transcripts/<nom>_transcript.txt`

## Branches Git

- `main` — version stable v1.0
- `feature/npu` — intégration accélération GPU/NPU (en cours)

## Commandes utiles

```powershell
# Lancer l'appli
.\.venv\Scripts\python.exe main.py

# Benchmark CPU vs DirectML
.\.venv\Scripts\python.exe benchmark.py

# Vérifier les providers onnxruntime
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Push sur GitHub
git add -A && git commit -m "message" && git push
```
