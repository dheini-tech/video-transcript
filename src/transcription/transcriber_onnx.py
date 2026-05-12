"""DirectML transcriber — Xenova Whisper ONNX float32 via DmlExecutionProvider.

KV-cache note: decoder_model_merged.onnx from Xenova requires use_cache_branch=False
for initialization, which triggers a Reshape bug in onnxruntime-directml 1.24.4
(If-node shape evaluation fails on the False branch). KV-cache is disabled until
onnxruntime-directml is updated. Decoder runs without cache (O(n^2) per chunk).
"""
import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
MAX_NEW_TOKENS = 448

_HF_ONNX_MAP = {
    "tiny":     "Xenova/whisper-tiny",
    "base":     "Xenova/whisper-base",
    "small":    "Xenova/whisper-small",
    "medium":   "Xenova/whisper-medium",
    "large-v2": "Xenova/whisper-large-v2",
    "large-v3": "Xenova/whisper-large-v3",
}
_HF_PT_MAP = {
    "tiny":     "openai/whisper-tiny",
    "base":     "openai/whisper-base",
    "small":    "openai/whisper-small",
    "medium":   "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}


class TranscriberONNX:
    def __init__(self, model_name: str, on_progress: Callable = None):
        self._model_name = model_name
        self._on_progress = on_progress or (lambda msg: None)
        self._encoder = None
        self._decoder = None
        self._cpu_decoder = None
        self._fe = None
        self._tok = None
        self._sot = None
        self._eos = None
        self._enc_input_name = None
        self._dec_input_ids_name = None
        self._dec_enc_hidden_name = None
        self._dml_dec_failed = False

    def _load(self):
        if self._encoder is not None:
            return

        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        hf_onnx = _HF_ONNX_MAP.get(self._model_name, "Xenova/whisper-small")
        hf_pt   = _HF_PT_MAP.get(self._model_name, "openai/whisper-small")

        self._on_progress(("log", f"Téléchargement modèle ONNX '{hf_onnx}'…"))
        cache = Path(__file__).parent.parent.parent / ".onnx_cache" / hf_onnx.replace("/", "--")
        cache.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=hf_onnx,
            local_dir=str(cache),
            # Skip merged decoder — incompatible with onnxruntime-directml 1.24.4
            ignore_patterns=["*merged*", "*fp16*", "*.msgpack", "*.safetensors", "flax_*", "tf_*"],
        )

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

        enc_path = os.path.join(local_dir, "onnx", "encoder_model.onnx")
        dec_path = os.path.join(local_dir, "onnx", "decoder_model.onnx")

        self._on_progress(("log", "Chargement encoder (DirectML)…"))
        self._encoder = ort.InferenceSession(enc_path, sess_options=opts, providers=providers)
        enc_provider = self._encoder.get_providers()[0]

        self._on_progress(("log", "Chargement decoder (DirectML)…"))
        self._decoder = ort.InferenceSession(dec_path, sess_options=opts, providers=providers)
        dec_provider = self._decoder.get_providers()[0]

        # CPU-only backup decoder for when DML crashes mid-sequence
        if dec_provider == "DmlExecutionProvider":
            cpu_opts = ort.SessionOptions()
            cpu_opts.log_severity_level = 3
            self._cpu_decoder = ort.InferenceSession(
                dec_path, sess_options=cpu_opts, providers=["CPUExecutionProvider"]
            )

        self._on_progress(("log", f"Modèle chargé — encoder: {enc_provider}, decoder: {dec_provider}."))

        # Detect input names dynamically
        self._enc_input_name = self._encoder.get_inputs()[0].name
        dec_inputs = {i.name: i for i in self._decoder.get_inputs()}
        if "input_ids" in dec_inputs:
            self._dec_input_ids_name  = "input_ids"
            self._dec_enc_hidden_name = "encoder_hidden_states"
        else:
            names  = list(dec_inputs.keys())
            id_n   = next((n for n in names if dec_inputs[n].type == "tensor(int64)"), names[0])
            self._dec_input_ids_name  = id_n
            self._dec_enc_hidden_name = next(n for n in names if n != id_n)

        self._fe  = WhisperFeatureExtractor.from_pretrained(hf_pt)
        self._tok = WhisperTokenizer.from_pretrained(hf_pt)
        self._sot = self._tok.convert_tokens_to_ids("<|startoftranscript|>")
        self._eos = self._tok.eos_token_id

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[dict]:
        self._load()

        if language:
            self._tok.set_prefix_tokens(language=language, task="transcribe")
            init_toks = list(self._tok.prefix_tokens)
        else:
            init_toks = [self._sot]

        self._on_progress(("log", "Transcription DirectML en cours…"))

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        chunk_size = SAMPLE_RATE * CHUNK_SECONDS
        overlap    = SAMPLE_RATE * 1
        starts     = list(range(0, len(audio), chunk_size - overlap))
        total      = len(starts)

        segments = []
        for i, start in enumerate(starts):
            end   = min(start + chunk_size, len(audio))
            chunk = audio[start:end]

            feats   = self._fe(chunk, sampling_rate=SAMPLE_RATE, return_tensors="np")
            enc_out = self._encoder.run(None, {self._enc_input_name: feats["input_features"]})[0]

            tokens = list(init_toks)
            for _ in range(MAX_NEW_TOKENS):
                tokens_arr = np.array([tokens], dtype=np.int64)
                feed = {
                    self._dec_input_ids_name:  tokens_arr,
                    self._dec_enc_hidden_name: enc_out,
                }
                if self._dml_dec_failed:
                    logits = self._cpu_decoder.run(None, feed)[0]
                else:
                    try:
                        logits = self._decoder.run(None, feed)[0]
                    except (RuntimeError, UnicodeDecodeError):
                        self._dml_dec_failed = True
                        self._on_progress(("log", "DirectML decoder instable — bascule CPU."))
                        dec = self._cpu_decoder or self._decoder
                        logits = dec.run(None, feed)[0]

                next_tok = int(np.argmax(logits[0, -1]))
                if next_tok == self._eos:
                    break
                tokens.append(next_tok)

            text = self._tok.decode(tokens[len(init_toks):], skip_special_tokens=True).strip()
            if text:
                segments.append({
                    "start": round(start / SAMPLE_RATE, 2),
                    "end":   round(end   / SAMPLE_RATE, 2),
                    "text":  text,
                })

            pct = 25.0 + ((i + 1) / total) * 24.0
            self._on_progress(("progress", pct))
            self._on_progress(("segment_count", len(segments)))

        self._on_progress(("log", f"Transcription DirectML terminée : {len(segments)} segments."))
        return segments
