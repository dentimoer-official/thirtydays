"""Whisper tiny 진입점: mel -> 인코더(층 단위) -> decoder."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import whisper
from whisper.tokenizer import get_tokenizer

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = whisper.audio.SAMPLE_RATE

_model = None


def load():
    global _model
    _model = whisper.load_model("tiny", download_root=str(ROOT / "models"), device="cpu")


def parts():
    blocks = [f"enc_block{i}" for i in range(len(_model.encoder.blocks))]

    return ["mel", "enc_stem", *blocks, "enc_norm", "decoder"]


def part_modules():
    enc = _model.encoder
    modules = {
        "mel": [],
        "enc_stem": [enc.conv1, enc.conv2],
        "enc_norm": [enc.ln_post],
        "decoder": [_model.decoder],
    }
    for i, block in enumerate(enc.blocks):
        modules[f"enc_block{i}"] = [block]

    return modules


def read_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return audio


def mel(x):
    # 입력: 오디오 파일 경로 또는 16kHz 파형 -> 출력: [1, 80, 3000]
    audio = read_audio(x) if isinstance(x, str) else np.asarray(x, dtype=np.float32)
    audio = whisper.pad_or_trim(torch.from_numpy(audio))

    return whisper.log_mel_spectrogram(audio, n_mels=_model.dims.n_mels).unsqueeze(0)


@torch.no_grad()
def enc_stem(x):
    # conv 두 개 + 위치 임베딩: [1, 80, 3000] -> [1, 1500, 384]
    enc = _model.encoder
    x = F.gelu(enc.conv1(x))
    x = F.gelu(enc.conv2(x))
    x = x.permute(0, 2, 1)

    return (x + enc.positional_embedding).to(x.dtype)


@torch.no_grad()
def enc_block(i, x):
    # 블록마다 입출력이 [1, 1500, 384]로 같아서 어느 층 사이에서든 잘림
    return _model.encoder.blocks[i](x)


@torch.no_grad()
def enc_norm(x):
    return _model.encoder.ln_post(x)


@torch.no_grad()
def decoder(features):
    # 오디오 특징 -> 언어 감지 후 greedy 디코딩
    _, probs = _model.detect_language(features)
    language = max(probs[0], key=probs[0].get)

    tokenizer = get_tokenizer(
        _model.is_multilingual,
        num_languages=_model.num_languages,
        language=language,
        task="transcribe",
    )
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    tokens = torch.tensor([prefix])

    for _ in range(_model.dims.n_text_ctx // 2):
        logits = _model.decoder(tokens, features)[:, -1]
        next_token = logits.argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eot:
            break
        tokens = torch.cat([tokens, next_token], dim=-1)

    text = tokenizer.decode(tokens[0, len(prefix):].tolist()).strip()

    return {"language": language, "text": text}


def run_part(name, x):
    if name == "mel":
        return mel(x)
    if name == "enc_stem":
        return enc_stem(x)
    if name.startswith("enc_block"):
        return enc_block(int(name.removeprefix("enc_block")), x)
    if name == "enc_norm":
        return enc_norm(x)
    if name == "decoder":
        return decoder(x)

    raise KeyError(f"unknown part: {name}")


def whole(x):
    # 비교 기준: 공식 decode를 같은 greedy 조건으로 통째 실행
    options = whisper.DecodingOptions(
        fp16=False,
        without_timestamps=True,
        suppress_tokens="",
        suppress_blank=False,
    )
    result = whisper.decode(_model, mel(x), options)[0]

    return {"language": result.language, "text": result.text}
