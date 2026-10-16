"""원본 대조 검사: python3.13 verify.py [모델 이름 ...]

모델마다 조각을 차례로 이어 실행하고, 통째로 실행한 결과와 같은지 확인한다.
이름을 주지 않으면 전부 돌린다.
"""

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from common import load_entry, same, to_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "N"
SAMPLES = ROOT / "samples"

SPEECH = str(SAMPLES / "speech_en.wav")
IMAGE = str(SAMPLES / "bus.jpg")

MODELS = {
    "demo": (ROOT / "demo_entry.py", [1.0, 2.0, 3.0]),
    "whisper": (ASSETS / "sound" / "whisper" / "entry.py", SPEECH),
    "yamnet": (ASSETS / "sound" / "surrounding" / "entry.py", SPEECH),
    "panns": (ASSETS / "sound" / "PANN" / "entry.py", SPEECH),
    "ssdlite": (ASSETS / "site" / "SSD site" / "entry.py", IMAGE),
    "yolo": (ASSETS / "site" / "yolo" / "entry.py", IMAGE),
    "clip": (ASSETS / "site" / "classify_image" / "entry.py", IMAGE),
    "fasttext": (ASSETS / "text" / "fastText" / "entry.py", "Hello, this is a test of the thirty days project."),
    "qwen": (ASSETS / "text" / "Qwen" / "entry.py", "The capital of France is"),
}


def run(name, path, x):
    entry = load_entry(name, path)
    parts = entry.parts()

    # 조각을 순서대로 이어 붙임
    start = time.perf_counter()
    out = x
    for part in parts:
        out = entry.run_part(part, out)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"[{name}] {len(parts)} parts, {elapsed:.0f} ms")
    print(f"  -> {to_json(out)}")
    if hasattr(entry, "whole"):
        print(f"  whole match: {same(out, entry.whole(x))}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    failed = []

    for name in names:
        path, x = MODELS[name]
        try:
            run(name, path, x)
        except Exception as e:
            # 한 모델이 실패해도 나머지는 계속 돌림
            print(f"[{name}] FAILED: {e!r}")
            failed.append(name)
        gc.collect()

    if failed:
        print(f"\nfailed: {', '.join(failed)}")
        sys.exit(1)
