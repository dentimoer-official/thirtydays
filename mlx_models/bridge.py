"""CLIP 이미지 임베딩 -> Qwen 입력 임베딩 다리 (학습 없음).

CLIP은 이미지와 텍스트가 같은 공간에 있으므로, 영어 단어 닻(anchor)마다
  CLIP 텍스트 임베딩(512)  <->  Qwen 단어 임베딩(896)
짝을 만들어 두고 이미지를 Qwen 쪽으로 옮긴다.
  soft  : 이미지와 닻들의 CLIP 유사도로 닻의 Qwen 임베딩을 가중 평균 (주 방식)
  linear: 닻 짝으로 최소제곱 맞춤한 선형 변환 512 -> 896 (비교용)
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


class Bridge:
    def __init__(self, path, qwen_model, logit_scale):
        path = Path(path)
        self.words = json.loads((path / "anchor_words.json").read_text())
        self.clip_text = mx.array(np.load(path / "clip_text.npy"))
        ids = mx.array(np.load(path / "anchor_ids.npy"))

        # Qwen 쪽 닻 임베딩은 실행 중인 모델(4비트든 bf16이든)의 임베딩 층에서 꺼냄
        self.qwen = qwen_model.model.embed_tokens(ids).astype(mx.float32)
        self.norm = float(mx.linalg.norm(self.qwen, axis=-1).mean())
        self.logit_scale = logit_scale
        mx.eval(self.qwen)

    def nearest(self, image_features, top_k=5):
        sims = (self._normalize(image_features) @ self.clip_text.T)[0]
        order = mx.argsort(-sims)[:top_k].tolist()

        return [(self.words[i], round(float(sims[i]), 4)) for i in order]

    def soft(self, image_features, top_k=None):
        # CLIP 제로샷과 같은 온도로 닻들에 대한 분포를 만들고, 그 분포로 Qwen 임베딩을 섞음
        sims = (self._normalize(image_features) @ self.clip_text.T)[0]
        if top_k:
            cut = mx.sort(sims)[-top_k]
            sims = mx.where(sims >= cut, sims, -mx.inf)
        weights = mx.softmax(self.logit_scale * sims)
        vector = weights @ self.qwen

        return vector * (self.norm / mx.linalg.norm(vector))

    def fit_linear(self, ridge=1e-2, holdout=500):
        """닻 짝으로 선형 변환을 맞추고, 남겨 둔 단어로 맞는 단어를 찾는지 확인."""
        x = np.asarray(self.clip_text)
        y = np.asarray(self.qwen)
        train, test = slice(holdout, None), slice(0, holdout)

        xt = np.hstack([x[train], np.ones((x[train].shape[0], 1), dtype=np.float32)])
        self.W = np.linalg.solve(xt.T @ xt + ridge * np.eye(xt.shape[1]), xt.T @ y[train])

        pred = np.hstack([x[test], np.ones((holdout, 1), dtype=np.float32)]) @ self.W
        pred /= np.linalg.norm(pred, axis=-1, keepdims=True)
        targets = y / np.linalg.norm(y, axis=-1, keepdims=True)
        hits = (pred @ targets.T).argmax(axis=-1) == np.arange(holdout)

        return float(hits.mean())

    def linear(self, image_features):
        x = np.asarray(self._normalize(image_features))
        vector = mx.array(np.hstack([x, np.ones((1, 1), dtype=np.float32)]) @ self.W)[0]

        return vector * (self.norm / mx.linalg.norm(vector))

    @staticmethod
    def _normalize(features):
        return features / mx.linalg.norm(features, axis=-1, keepdims=True)
