"""Whisper MLX 구현 (openai-whisper와 같은 구조).

torch 없이 돈다: mel은 numpy로, 토크나이저는 openai-whisper의 tokenizer.py만 파일 경로로 읽는다.
조각: mel -> encode -> decode
"""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES = 30 * SAMPLE_RATE

_WHISPER_DIR = Path(importlib.util.find_spec("whisper").submodule_search_locations[0])


def _load_tokenizer_module():
    # whisper/__init__.py는 torch를 import하므로 tokenizer.py만 따로 읽음
    spec = importlib.util.spec_from_file_location("whisper_tokenizer", _WHISPER_DIR / "tokenizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_tokenizer_module = _load_tokenizer_module()


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(self, x, xa=None, mask=None):
        source = x if xa is None else xa
        B, L, D = x.shape
        q = self.query(x).reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = self.key(source).reshape(B, source.shape[1], self.n_head, -1).transpose(0, 2, 1, 3)
        v = self.value(source).reshape(B, source.shape[1], self.n_head, -1).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // self.n_head) ** -0.5, mask=mask)

        return self.out(out.transpose(0, 2, 1, 3).reshape(B, L, D))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention=False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        if cross_attention:
            self.cross_attn = MultiHeadAttention(n_state, n_head)
            self.cross_attn_ln = nn.LayerNorm(n_state)
        self.mlp1 = nn.Linear(n_state, n_state * 4)
        self.mlp2 = nn.Linear(n_state * 4, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x, xa=None, mask=None):
        x = x + self.attn(self.attn_ln(x), mask=mask)
        if xa is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)

        return x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))


class AudioEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv1 = nn.Conv1d(d["n_mels"], d["n_audio_state"], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d["n_audio_state"], d["n_audio_state"], kernel_size=3, stride=2, padding=1)
        self.positional_embedding = mx.zeros((d["n_audio_ctx"], d["n_audio_state"]))
        self.blocks = [ResidualAttentionBlock(d["n_audio_state"], d["n_audio_head"]) for _ in range(d["n_audio_layer"])]
        self.ln_post = nn.LayerNorm(d["n_audio_state"])

    def stem(self, mel):
        # mel [B, 3000, 80] (MLX는 채널이 마지막) -> [B, 1500, 384]
        x = nn.gelu(self.conv1(mel))
        x = nn.gelu(self.conv2(x))

        return x + self.positional_embedding

    def __call__(self, mel):
        x = self.stem(mel)
        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)


class TextDecoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.token_embedding = nn.Embedding(d["n_vocab"], d["n_text_state"])
        self.positional_embedding = mx.zeros((d["n_text_ctx"], d["n_text_state"]))
        self.blocks = [
            ResidualAttentionBlock(d["n_text_state"], d["n_text_head"], cross_attention=True)
            for _ in range(d["n_text_layer"])
        ]
        self.ln = nn.LayerNorm(d["n_text_state"])

    def __call__(self, tokens, xa):
        L = tokens.shape[-1]
        x = self.token_embedding(tokens) + self.positional_embedding[:L]
        mask = "causal" if L > 1 else None
        for block in self.blocks:
            x = block(x, xa, mask=mask)

        return self.token_embedding.as_linear(self.ln(x))


class Whisper(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(dims)
        self.decoder = TextDecoder(dims)

    @property
    def is_multilingual(self):
        return self.dims["n_vocab"] >= 51865

    @property
    def num_languages(self):
        return self.dims["n_vocab"] - 51765 - int(self.is_multilingual)


def load(path):
    """prepare.py가 만든 mlx/float32 또는 mlx/float16 폴더를 읽는다."""
    path = Path(path)
    dims = json.loads((path / "dims.json").read_text())
    model = Whisper(dims)
    model.load_weights(str(path / "weights.safetensors"))
    mx.eval(model.parameters())

    return model


def read_audio(x):
    if not isinstance(x, str):
        return np.asarray(x, dtype=np.float32)

    audio, sr = sf.read(x, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return audio


def mel(x, n_mels=80):
    """whisper.log_mel_spectrogram과 같은 계산을 numpy로: 30초로 자르거나 채운 뒤 [3000, 80]."""
    audio = read_audio(x)[:N_SAMPLES]
    audio = np.pad(audio, (0, N_SAMPLES - len(audio)))

    # torch.stft(center=True, pad_mode="reflect", 주기적 hann 창)와 같음
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    magnitudes = np.abs(np.fft.rfft(frames * window, axis=-1))[:-1] ** 2

    filters = np.load(_WHISPER_DIR / "assets" / "mel_filters.npz")[f"mel_{n_mels}"]
    log_spec = np.log10(np.maximum(filters @ magnitudes.T, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)

    return ((log_spec + 4.0) / 4.0).T.astype(np.float32)


def encode(model, mel_frames):
    # [3000, 80] -> 오디오 특징 [1, 1500, 384]
    dtype = model.encoder.conv1.weight.dtype
    features = model.encoder(mx.array(mel_frames)[None].astype(dtype))
    mx.eval(features)

    return features


def tokenizer_for(model, language=None):
    return _tokenizer_module.get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task="transcribe",
    )


def detect_language(model, features):
    # 첫 위치에서 언어 토큰 중 가장 높은 것 (whisper.detect_language와 같은 방식)
    tokenizer = tokenizer_for(model)
    logits = model.decoder(mx.array([[tokenizer.sot]]), features)[0, 0].astype(mx.float32)
    ids = list(tokenizer.all_language_tokens)
    best = int(mx.argmax(logits[mx.array(ids)]).item())

    return tokenizer.all_language_codes[best]


def decode(model, features, max_tokens=None):
    """오디오 특징 -> 언어 감지 후 greedy 디코딩."""
    language = detect_language(model, features)
    tokenizer = tokenizer_for(model, language)
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    tokens = list(prefix)

    for _ in range(max_tokens or model.dims["n_text_ctx"] // 2):
        logits = model.decoder(mx.array([tokens]), features)[0, -1]
        token = int(mx.argmax(logits).item())
        if token == tokenizer.eot:
            break
        tokens.append(token)

    return {"language": language, "text": tokenizer.decode(tokens[len(prefix):]).strip()}


def transcribe(model, x):
    return decode(model, encode(model, mel(x)))
