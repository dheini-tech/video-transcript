"""
Worker — tourne dans le conda env ryzen-ai-1.7.1 (Python 3.12).
Appel : python npu_worker.py <audio_wav> <model_name> [language] [device]
  device : "npu" (defaut) ou "cpu"
Sortie : JSON sur stdout -> liste de segments {start, end, text}
"""
import json
import os
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
SCRIPT_DIR = Path(__file__).parent
NPU_CONFIG_DIR = SCRIPT_DIR / "npu_config"
NPU_CACHE_DIR  = SCRIPT_DIR / ".npu_cache"

_HF_ONNX_MAP = {
    "tiny":     "amd/whisper-tiny-onnx-npu",
    "base":     "amd/whisper-base-onnx-npu",
    "small":    "amd/whisper-small-onnx-npu",
    "medium":   "amd/whisper-medium-onnx-npu",
    "large-v3": "amd/whisper-large-v3-onnx-npu",
}
_HF_PT_MAP = {
    "tiny": "openai/whisper-tiny", "base": "openai/whisper-base",
    "small": "openai/whisper-small", "medium": "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2", "large-v3": "openai/whisper-large-v3",
}


def log(msg: str):
    print(json.dumps({"log": msg}), flush=True)


# ──────────────────────────────────────────── NPU (VitisAI ONNX) ──────────────

def transcribe_npu(audio: np.ndarray, model_name: str, language: str | None) -> list[dict]:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    hf_repo  = _HF_ONNX_MAP.get(model_name, "amd/whisper-small-onnx-npu")
    hf_name  = f"openai/whisper-{model_name}" if model_name != "large-v3" else "openai/whisper-large-v3"

    log(f"Telechargement modele NPU '{hf_repo}'...")
    local_dir    = snapshot_download(repo_id=hf_repo)
    encoder_path = os.path.join(local_dir, "encoder_model.onnx")
    decoder_path = os.path.join(local_dir, "decoder_model.onnx")

    enc_cfg = str(NPU_CONFIG_DIR / "vitisai_encoder.json")
    dec_cfg = str(NPU_CONFIG_DIR / "vitisai_decoder.json")
    NPU_CACHE_DIR.mkdir(exist_ok=True)

    def load_sess(path, cfg, key):
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        vai_opts = {"config_file": cfg, "cache_dir": str(NPU_CACHE_DIR), "cache_key": key}
        try:
            return ort.InferenceSession(path, sess_options=opts,
                providers=["VitisAIExecutionProvider"], provider_options=[vai_opts])
        except Exception as e:
            log(f"  >> VitisAI echec ({e}), fallback CPU")
            return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])

    log("Chargement encoder NPU...")
    encoder = load_sess(encoder_path, enc_cfg, f"{model_name}_encoder")
    log("Chargement decoder NPU...")
    decoder = load_sess(decoder_path, dec_cfg, f"{model_name}_decoder")

    fe  = WhisperFeatureExtractor.from_pretrained(hf_name)
    tok = WhisperTokenizer.from_pretrained(hf_name)
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    eos = tok.eos_token_id
    max_len = min(448, decoder.get_inputs()[0].shape[1])
    if not isinstance(max_len, int):
        raise ValueError("Forme dynamique non supportee")

    if language:
        tok.set_prefix_tokens(language=language, task="transcribe")
        init_toks = list(tok.prefix_tokens)
    else:
        init_toks = [sot]

    enc_in  = encoder.get_inputs()[0].name
    di = decoder.get_inputs()
    id_name, enc_out_name = di[0].name, di[1].name
    if di[0].type != "tensor(int64)":
        id_name, enc_out_name = enc_out_name, id_name

    return _run_chunks(audio, encoder, decoder, fe, tok, enc_in, id_name, enc_out_name,
                       init_toks, eos, max_len, onnx_mode=True)


# ──────────────────────────────────────────── CPU (PyTorch) ───────────────────

def transcribe_cpu(audio: np.ndarray, model_name: str, language: str | None) -> list[dict]:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    hf_name = _HF_PT_MAP.get(model_name, "openai/whisper-small")
    log(f"Chargement modele PyTorch '{hf_name}'...")
    processor = WhisperProcessor.from_pretrained(hf_name)
    model = WhisperForConditionalGeneration.from_pretrained(hf_name)
    model.eval()

    chunk_size = SAMPLE_RATE * CHUNK_SECONDS
    overlap    = SAMPLE_RATE * 1
    chunks     = list(range(0, len(audio), chunk_size - overlap))
    gen_kwargs = {"language": language} if language else {}

    log(f"Transcription PyTorch CPU ({len(chunks)} chunks)...")
    segments = []
    for i, start in enumerate(chunks):
        end   = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(inputs["input_features"], **gen_kwargs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            segments.append({"start": round(start / SAMPLE_RATE, 2),
                             "end":   round(end   / SAMPLE_RATE, 2),
                             "text":  text})
        pct = ((i + 1) / len(chunks)) * 100
        print(json.dumps({"progress": pct}), flush=True)
        print(json.dumps({"segment_count": len(segments)}), flush=True)

    return segments


# ──────────────────────────────────────────── chunk runner (NPU) ──────────────

def _run_chunks(audio, encoder, decoder, fe, tok, enc_in, id_name, enc_out_name,
                init_toks, eos, max_len, onnx_mode):
    chunk_size = SAMPLE_RATE * CHUNK_SECONDS
    overlap    = SAMPLE_RATE * 1
    chunks     = list(range(0, len(audio), chunk_size - overlap))
    log(f"Transcription NPU ({len(chunks)} chunks)...")
    segments = []
    t0 = time.monotonic()

    for i, start in enumerate(chunks):
        end   = min(start + chunk_size, len(audio))
        chunk = audio[start:end]

        feats   = fe(chunk, sampling_rate=SAMPLE_RATE, return_tensors="np")
        enc_out = encoder.run(None, {enc_in: feats["input_features"]})[0]

        tokens = list(init_toks)
        for _ in range(len(tokens), max_len):
            dec_in = np.full((1, max_len), eos, dtype=np.int64)
            dec_in[0, :len(tokens)] = tokens
            logits = decoder.run(None, {id_name: dec_in, enc_out_name: enc_out})[0]
            next_tok = int(np.argmax(logits[0, len(tokens) - 1]))
            if next_tok == eos:
                break
            tokens.append(next_tok)

        text = tok.decode(tokens[len(init_toks):], skip_special_tokens=True).strip()
        if text:
            segments.append({"start": round(start / SAMPLE_RATE, 2),
                             "end":   round(end   / SAMPLE_RATE, 2),
                             "text":  text})
        pct = ((i + 1) / len(chunks)) * 100
        print(json.dumps({"progress": pct}), flush=True)
        print(json.dumps({"segment_count": len(segments)}), flush=True)

    elapsed = time.monotonic() - t0
    log(f"Transcription terminee : {len(segments)} segments en {elapsed:.1f}s")
    return segments


# ──────────────────────────────────────────── main ────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: npu_worker.py <audio> <model> [language] [npu|cpu]"}))
        sys.exit(1)

    audio_path = sys.argv[1]
    model_name = sys.argv[2]
    language   = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("npu", "cpu") else None
    device     = next((a for a in sys.argv[3:] if a in ("npu", "cpu")), "npu")

    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if device == "cpu":
            segs = transcribe_cpu(audio, model_name, language)
        else:
            segs = transcribe_npu(audio, model_name, language)

        print(json.dumps({"segments": segs}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        sys.exit(1)
