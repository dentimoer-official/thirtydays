"""YAMNet 진입점: features -> reshape -> layer1~14 -> pool -> classify."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "yamnet.h5"
CLASS_MAP = ROOT / "yamnet_class_map.csv"
TOP_K = 5

sys.path.insert(0, str(ROOT))
import features as features_lib  # noqa: E402
import params as yamnet_params  # noqa: E402
import yamnet_model  # noqa: E402
sys.path.remove(str(ROOT))

_params = None
_model = None
_names = None
_groups = {}


def load():
    global _params, _model, _names
    _params = yamnet_params.Params()
    _model = yamnet_model.yamnet_frames_model(_params)
    _model.load_weights(str(WEIGHTS))
    _names = yamnet_model.class_names(str(CLASS_MAP))

    # Keras 레이어를 이름("layerN/...")과 타입으로 조각에 묶음
    for layer in _model.layers:
        kind = type(layer).__name__
        if layer.name.startswith("layer"):
            _groups.setdefault(layer.name.split("/")[0], []).append(layer)
        elif kind == "Reshape":
            _groups["reshape"] = [layer]
        elif kind == "GlobalAveragePooling2D":
            _groups["pool"] = [layer]
        elif kind in ("Dense", "Activation"):
            _groups.setdefault("classify", []).append(layer)


def parts():
    layers = sorted((k for k in _groups if k.startswith("layer")), key=lambda k: int(k.removeprefix("layer")))

    return ["features", "reshape", *layers, "pool", "classify"]


def part_modules():
    return {"features": [], **_groups}


def read_audio(x):
    if not isinstance(x, str):
        return np.asarray(x, dtype=np.float32)

    audio, sr = sf.read(x, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != _params.sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=int(_params.sample_rate))

    return audio


def features(x):
    # 파형 -> log-mel 패치 [패치 수, 96, 64]
    waveform = tf.constant(read_audio(x))
    padded = features_lib.pad_waveform(waveform, _params)
    _, patches = features_lib.waveform_to_log_mel_spectrogram_patches(padded, _params)

    return patches.numpy()


def apply(name, x):
    for layer in _groups[name]:
        x = layer(x)

    return x.numpy()


def summarize(scores):
    # 프레임 평균 점수 상위 TOP_K 클래스
    mean = np.asarray(scores).mean(axis=0)
    top = np.argsort(mean)[::-1][:TOP_K]

    return [{"label": str(_names[i]), "score": round(float(mean[i]), 4)} for i in top]


def run_part(name, x):
    if name == "features":
        return features(x)
    if name == "classify":
        return summarize(apply(name, x))
    if name in _groups:
        return apply(name, x)

    raise KeyError(f"unknown part: {name}")


def whole(x):
    # 비교 기준: frames 모델을 통째로 실행
    predictions, _, _ = _model(read_audio(x))

    return summarize(predictions.numpy())
