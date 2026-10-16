"""PANNs Cnn14 진입점: logmel -> bn0 -> block1~6 -> pool -> fc1 -> classify."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "Cnn14_mAP=0.431.pth"
LABELS = ROOT / "PANNs-source" / "metadata" / "class_labels_indices.csv"
SAMPLE_RATE = 32000
TOP_K = 5

# 블록별 풀링 크기 (Cnn14.forward와 같음)
POOLS = {1: (2, 2), 2: (2, 2), 3: (2, 2), 4: (2, 2), 5: (2, 2), 6: (1, 1)}

_model = None
_labels = None


def load():
    global _model, _labels
    # 친구가 만든 model.py의 build_model을 그대로 씀 (PANNs-source를 sys.path에 넣고 Cnn14를 가져옴)
    spec = importlib.util.spec_from_file_location("panns_model", ROOT / "model.py")
    panns_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panns_model)

    _model = panns_model.build_model()
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    _model.load_state_dict(checkpoint["model"])
    _model.eval()

    with open(LABELS) as f:
        _labels = [row["display_name"] for row in csv.DictReader(f)]


def parts():
    return ["logmel", "bn0", *[f"block{i}" for i in POOLS], "pool", "fc1", "classify"]


def part_modules():
    modules = {
        "logmel": [_model.spectrogram_extractor, _model.logmel_extractor],
        "bn0": [_model.bn0],
        "pool": [],
        "fc1": [_model.fc1],
        "classify": [_model.fc_audioset],
    }
    for i in POOLS:
        modules[f"block{i}"] = [getattr(_model, f"conv_block{i}")]

    return modules


def read_audio(x):
    if not isinstance(x, str):
        return np.asarray(x, dtype=np.float32)

    audio, sr = sf.read(x, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return audio


@torch.no_grad()
def logmel(x):
    # 32kHz 파형 -> log-mel [1, 1, 프레임, 64]
    waveform = torch.from_numpy(read_audio(x)).unsqueeze(0)

    return _model.logmel_extractor(_model.spectrogram_extractor(waveform))


@torch.no_grad()
def bn0(x):
    return _model.bn0(x.transpose(1, 3)).transpose(1, 3)


@torch.no_grad()
def block(i, x):
    x = getattr(_model, f"conv_block{i}")(x, pool_size=POOLS[i], pool_type="avg")

    return F.dropout(x, p=0.2, training=False)


@torch.no_grad()
def pool(x):
    # 주파수 평균 -> 시간축 max + mean -> [1, 2048]
    x = torch.mean(x, dim=3)
    x1, _ = torch.max(x, dim=2)
    x2 = torch.mean(x, dim=2)

    return x1 + x2


@torch.no_grad()
def fc1(x):
    # 오디오 임베딩 [1, 2048]
    return F.relu_(_model.fc1(x))


@torch.no_grad()
def classify(x):
    return summarize(torch.sigmoid(_model.fc_audioset(x)))


def summarize(clipwise):
    scores = clipwise[0]
    top = scores.topk(TOP_K)

    return [{"label": _labels[int(i)], "score": round(float(s), 4)} for s, i in zip(top.values, top.indices)]


def run_part(name, x):
    if name == "logmel":
        return logmel(x)
    if name == "bn0":
        return bn0(x)
    if name.startswith("block"):
        return block(int(name.removeprefix("block")), x)
    if name == "pool":
        return pool(x)
    if name == "fc1":
        return fc1(x)
    if name == "classify":
        return classify(x)

    raise KeyError(f"unknown part: {name}")


@torch.no_grad()
def whole(x):
    # 비교 기준: Cnn14를 통째로 실행
    waveform = torch.from_numpy(read_audio(x)).unsqueeze(0)

    return summarize(_model(waveform)["clipwise_output"])
