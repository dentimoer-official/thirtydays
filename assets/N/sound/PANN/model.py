"""공식 PANNs 소스에서 Cnn14 모델 클래스를 가져오는 진입점."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "PANNs-source" / "pytorch"
sys.path.insert(0, str(SOURCE))

from models import Cnn14  # noqa: E402


def build_model():
    return Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )


if __name__ == "__main__":
    print(build_model())
