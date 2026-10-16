# 예: whisper/entry.py
import whisper

_model = None


def load():
    global _model
    _model = whisper.load_model("tiny")


def parts():
    # 이 모델을 어디서 자르는지 = 절단면 정의
    return ["mel", "encoder", "decoder"]


def run_part(name, x):
    if name == "mel":
        return whisper.log_mel_spectrogram(x)
    if name == "encoder":
        return _model.encoder(x)
    if name == "decoder":
        return _model.decoder(*x)