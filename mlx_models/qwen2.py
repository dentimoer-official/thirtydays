"""Qwen2 MLX 구현 (mlx_lm.models.qwen2와 같은 구조와 가중치 이름).

mlx_lm을 import하면 transformers와 torch까지 딸려 와서(약 300MB) 직접 구현했다.
keep_layers로 디코더 층 일부만 골라 조립할 수 있다.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        head_dim = dim // self.n_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)
        self.rope = nn.RoPE(head_dim, traditional=False, base=cfg["rope_theta"])

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)

        mask = "causal" if L > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, hidden = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def __call__(self, x, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), cache)

        return h + self.mlp(self.post_attention_layernorm(h))


class Inner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [DecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])


class Qwen2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = Inner(cfg)
        self.tied = cfg.get("tie_word_embeddings", True)
        if not self.tied:
            self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def head(self, h):
        return self.model.embed_tokens.as_linear(h) if self.tied else self.lm_head(h)

    def forward_embeddings(self, h, caches=None):
        # 토큰 대신 임베딩을 직접 받는 길 (이미지 같은 외부 벡터를 끼워 넣을 때)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, caches[i] if caches else None)

        return self.head(self.model.norm(h))

    def __call__(self, ids, caches=None):
        return self.forward_embeddings(self.model.embed_tokens(ids), caches)


class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update(self, k, v):
        self.keys = k if self.keys is None else mx.concatenate([self.keys, k], axis=2)
        self.values = v if self.values is None else mx.concatenate([self.values, v], axis=2)
        self.offset += k.shape[2]

        return self.keys, self.values


def load(path, keep_layers=None):
    """가중치 폴더(prepare.py가 만든 mlx/q4 등)를 읽는다. keep_layers를 주면 그 층만 남긴다."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    weights = mx.load(str(path / "model.safetensors"))

    model = Qwen2(cfg)
    if quant := cfg.get("quantization"):
        nn.quantize(
            model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            class_predicate=lambda p, m: f"{p}.scales" in weights,
        )
    model.load_weights(list(weights.items()))

    if keep_layers is not None:
        model.model.layers = [model.model.layers[i] for i in keep_layers]

    mx.eval(model.parameters())
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

    return model, tokenizer


def chat_prompt(user, system=SYSTEM, history=None):
    # Qwen2.5 기본 chat template과 같은 형식. history는 [(사용자, 답변), ...]
    turns = "".join(
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        for q, a in history or []
    )

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"{turns}"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, tokenizer, prompt, max_tokens=256, inject=None):
    """greedy 생성. <|im_end|> 또는 <|endoftext|>에서 멈춘다.

    inject: {토큰 문자열: 벡터}. 프롬프트에서 그 토큰 자리의 임베딩을 벡터로 바꿔 넣는다.
    """
    stop = {tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>")}
    caches = [KVCache() for _ in model.model.layers]

    ids = tokenizer.encode(prompt).ids
    h = model.model.embed_tokens(mx.array([ids]))
    for token, vector in (inject or {}).items():
        token_id = tokenizer.token_to_id(token)
        for pos in [i for i, t in enumerate(ids) if t == token_id]:
            h[0, pos] = vector.astype(h.dtype)

    out = []
    for _ in range(max_tokens):
        logits = model.forward_embeddings(h, caches)[:, -1, :]
        token = int(mx.argmax(logits, axis=-1).item())
        if token in stop:
            break
        out.append(token)
        h = model.model.embed_tokens(mx.array([[token]]))

    return tokenizer.decode(out, skip_special_tokens=True).strip()
