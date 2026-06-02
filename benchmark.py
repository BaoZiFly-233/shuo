# Benchmark Log  (audio=24.24s)
# Method                    Params        Time(s)    RTF
# ───────────────────────────────────────────────────────
# cold_start (load+infer)   threads=0      38.44    1.59
# warm_infer                threads=0      27.58    1.14
# warm_infer                threads=2      17.96    0.74
# warm_infer                threads=4      11.93    0.49
# warm_infer                threads=6       6.59    0.27   ← best
# warm_infer                threads=8       6.82    0.28

import time, sys, shutil
from pathlib import Path

model_root = Path(__file__).parent / "Qwen3-ASR-0.6B-ONNX-CPU"
sys.path.insert(0, str(model_root))
from onnx_inference import OnnxAsrPipeline, load_audio

AUDIO = Path(__file__).parent / "assets" / "recording_test.wav"
SR = 16000

def ensure_tokenizer():
    mdir = model_root / "onnx_models"
    tok_src = model_root / "tokenizer.json"
    tok_dst = mdir / "tokenizer.json"
    if tok_src.exists() and not tok_dst.exists():
        shutil.copy2(str(tok_src), str(tok_dst))

def make_pipeline(num_threads=0):
    ensure_tokenizer()
    mdir = model_root / "onnx_models"
    return OnnxAsrPipeline(onnx_dir=str(mdir), num_threads=num_threads)

def bench(label, params, fn):
    t0 = time.time()
    result = fn()
    t = time.time() - t0
    print(f"  {label:30s} {params:12s} {t:.3f}")
    return t

def log(label, params, t):
    pass

if __name__ == "__main__":
    audio_path = str(AUDIO)
    import librosa
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    audio_dur = len(wav) / SR
    print(f"Audio: {audio_dur:.2f}s")

    ensure_tokenizer()

    # 1. Cold start (load + infer, threads=0)
    t = bench("cold_start", "threads=0", lambda: OnnxAsrPipeline(onnx_dir=str(model_root/"onnx_models")).transcribe(audio_path))

    # 2. Warm — one warm-up run, then time
    for threads in [0, 2, 4, 6, 8]:
        p = make_pipeline(threads)
        p.transcribe(audio_path)
        t = bench("warm_infer", f"threads={threads}", lambda p=p: p.transcribe(audio_path))
        del p

