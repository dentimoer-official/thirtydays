"""Qwen2.5-0.5B-Instruct 진입점: tokenize -> embed -> layer0~23 -> norm -> lm_head -> next_token."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
TOP_K = 5

_model = None
_tokenizer = None


def load():
    global _model, _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
    _model = AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float32, attn_implementation="sdpa").eval()


def parts():
    layers = [f"layer{i}" for i in range(len(_model.model.layers))]

    return ["tokenize", "embed", *layers, "norm", "lm_head", "next_token"]


def part_modules():
    inner = _model.model
    tied = _model.lm_head.weight is inner.embed_tokens.weight
    modules = {
        "tokenize": [],
        "embed": [inner.embed_tokens],
        "norm": [inner.norm],
        # 입력 임베딩과 가중치를 공유하면 두 번 세지 않음
        "lm_head": [] if tied else [_model.lm_head],
        "next_token": [],
    }
    for i, layer in enumerate(inner.layers):
        modules[f"layer{i}"] = [layer]

    return modules


def tokenize(text):
    # 문장 -> 토큰 id [1, T]
    return _tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def embed(input_ids):
    # Qwen2Model.forward 앞부분과 같음: 임베딩, 위치, 인과 마스크, RoPE
    inner = _model.model
    hidden = inner.embed_tokens(input_ids)
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    mask = create_causal_mask(
        config=inner.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    return {
        "x": hidden,
        "mask": mask,
        "position_ids": position_ids,
        "position_embeddings": inner.rotary_emb(hidden, position_ids),
    }


@torch.no_grad()
def layer(i, p):
    # 층마다 입출력이 [1, T, 896]으로 같음
    out = _model.model.layers[i](
        p["x"],
        attention_mask=p["mask"],
        position_embeddings=p["position_embeddings"],
        position_ids=p["position_ids"],
        past_key_values=None,
        use_cache=False,
    )

    return {**p, "x": out}


@torch.no_grad()
def norm(p):
    return _model.model.norm(p["x"])


@torch.no_grad()
def lm_head(hidden):
    # 마지막 위치의 어휘 점수 [1, 151936]
    return _model.lm_head(hidden[:, -1, :])


def next_token(logits):
    probs = logits[0].softmax(dim=-1)
    top = probs.topk(TOP_K)

    return [{"token": _tokenizer.decode([int(i)]), "score": round(float(p), 4)} for p, i in zip(top.values, top.indices)]


def run_part(name, x):
    if name == "tokenize":
        return tokenize(x)
    if name == "embed":
        return embed(x)
    if name.startswith("layer"):
        return layer(int(name.removeprefix("layer")), x)
    if name == "norm":
        return norm(x)
    if name == "lm_head":
        return lm_head(x)
    if name == "next_token":
        return next_token(x)

    raise KeyError(f"unknown part: {name}")


@torch.no_grad()
def whole(text):
    # 비교 기준: transformers 모델을 통째로 실행
    logits = _model(tokenize(text), use_cache=False).logits

    return next_token(logits[:, -1, :])
