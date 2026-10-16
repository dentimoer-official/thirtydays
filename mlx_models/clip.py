"""CLIP 이미지 인코더 MLX 구현 (텍스트 인코더는 prepare.py에서 미리 임베딩하고 떼어냄).

조각: preprocess -> patchify -> block0~11 -> head -> match
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.n_head = width // 64
        self.in_proj = nn.Linear(width, width * 3)
        self.out_proj = nn.Linear(width, width)
        self.ln_1 = nn.LayerNorm(width)
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.ln_2 = nn.LayerNorm(width)

    def attention(self, x):
        B, L, D = x.shape
        q, k, v = mx.split(self.in_proj(x), 3, axis=-1)
        q, k, v = (t.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3) for t in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // self.n_head) ** -0.5)

        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, L, D))

    def __call__(self, x):
        x = x + self.attention(self.ln_1(x))
        h = self.c_fc(self.ln_2(x))

        return x + self.c_proj(h * mx.sigmoid(1.702 * h))


class VisionTransformer(nn.Module):
    def __init__(self, width, patch, layers, resolution, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        self.class_embedding = mx.zeros((width,))
        self.positional_embedding = mx.zeros(((resolution // patch) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = [ResidualAttentionBlock(width) for _ in range(layers)]
        self.ln_post = nn.LayerNorm(width)
        self.proj = mx.zeros((width, output_dim))

    def patchify(self, image):
        # [1, 224, 224, 3] -> 패치 49개 + CLS 토큰 [1, 50, 768]
        x = self.conv1(image)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        cls = mx.broadcast_to(self.class_embedding, (x.shape[0], 1, x.shape[-1])).astype(x.dtype)
        x = mx.concatenate([cls, x], axis=1) + self.positional_embedding

        return self.ln_pre(x)

    def head(self, x):
        # CLS 토큰 -> 이미지 임베딩 [1, 512]
        return self.ln_post(x[:, 0, :]) @ self.proj

    def __call__(self, image):
        x = self.patchify(image)
        for block in self.blocks:
            x = block(x)

        return self.head(x)


def remap(weights):
    # torch 이름(transformer.resblocks.N.attn.in_proj_weight 등) -> 이 구현의 이름
    out = {}
    for key, value in weights.items():
        key = key.replace("transformer.resblocks.", "blocks.")
        key = key.replace("attn.in_proj_weight", "in_proj.weight").replace("attn.in_proj_bias", "in_proj.bias")
        key = key.replace("attn.out_proj.", "out_proj.").replace("mlp.c_fc.", "c_fc.").replace("mlp.c_proj.", "c_proj.")
        out[key] = value

    return out


class Clip:
    def __init__(self, path):
        path = Path(path)
        meta = json.loads((path.parent / "labels.json").read_text())
        self.labels = meta["labels"]
        self.logit_scale = meta["logit_scale"]
        self.resolution = meta["resolution"]
        self.text_features = mx.array(np.load(path.parent / "text_features.npy"))

        self.visual = VisionTransformer(meta["width"], meta["patch"], meta["layers"], meta["resolution"], self.text_features.shape[-1])
        self.visual.load_weights(list(remap(mx.load(str(path / "weights.safetensors"))).items()))
        mx.eval(self.visual.parameters())
        self.dtype = self.visual.conv1.weight.dtype

    def preprocess(self, x):
        # torchvision Resize(224, bicubic) + CenterCrop(224) + Normalize와 같음 -> [1, 224, 224, 3]
        image = Image.open(x).convert("RGB") if isinstance(x, str) else x
        w, h = image.size
        size = self.resolution
        if w < h:
            new_w, new_h = size, int(size * h / w)
        else:
            new_w, new_h = int(size * w / h), size
        image = image.resize((new_w, new_h), Image.BICUBIC)

        top, left = int(round((new_h - size) / 2.0)), int(round((new_w - size) / 2.0))
        image = image.crop((left, top, left + size, top + size))
        array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD

        return mx.array(array[None]).astype(self.dtype)

    def embed(self, x):
        features = self.visual(self.preprocess(x)).astype(mx.float32)
        mx.eval(features)

        return features

    def match(self, image_features, top_k=5):
        image_features = image_features / mx.linalg.norm(image_features, axis=-1, keepdims=True)
        probs = mx.softmax(self.logit_scale * image_features @ self.text_features.T, axis=-1)[0]
        order = mx.argsort(-probs)[:top_k].tolist()

        return [{"label": self.labels[i], "score": round(float(probs[i]), 4)} for i in order]

    def classify(self, x, top_k=5):
        return self.match(self.embed(x), top_k)
