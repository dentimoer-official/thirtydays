"""LLM 수술 도구: 층에 구멍을 뚫고(제거), 다른 모델의 같은 자리 부위를 이식한다.

원본 모델 객체는 건드리지 않고, 층 목록만 새로 조립해 끼웠다가 되돌린다.
계획(plan)은 {층 번호: 동작}이고 동작은 아래 중 하나다.
  "remove"  그 층을 빼서 구멍으로 둠 (잔차 연결 덕분에 입력이 그대로 다음 층으로 감)
  "layer"   기증 모델의 같은 번호 층으로 통째 교체
  "attn"    어텐션(+그 앞 정규화)만 기증 모델 것으로
  "mlp"     MLP(+그 앞 정규화)만 기증 모델 것으로
"""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


class HybridLayer(nn.Module):
    """어텐션과 MLP를 서로 다른 모델에서 가져온 층. DecoderLayer와 같은 계산."""

    def __init__(self, attn_from, mlp_from):
        super().__init__()
        self.input_layernorm = attn_from.input_layernorm
        self.self_attn = attn_from.self_attn
        self.post_attention_layernorm = mlp_from.post_attention_layernorm
        self.mlp = mlp_from.mlp

    def __call__(self, x, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), cache)

        return h + self.mlp(self.post_attention_layernorm(h))


def assemble(recipient_layers, donor_layers, plan):
    layers = []
    for i, layer in enumerate(recipient_layers):
        action = plan.get(i)
        if action == "remove":
            continue
        if action == "layer":
            layers.append(donor_layers[i])
        elif action == "attn":
            layers.append(HybridLayer(donor_layers[i], layer))
        elif action == "mlp":
            layers.append(HybridLayer(layer, donor_layers[i]))
        else:
            layers.append(layer)

    return layers


@contextmanager
def operated(model, donor, plan):
    """with 블록 안에서만 수술된 상태로 둠."""
    original = model.model.layers
    model.model.layers = assemble(original, donor.model.layers if donor else [], plan)
    try:
        yield model
    finally:
        model.model.layers = original


def save(model, source_dir, target_dir):
    """수술된 모델을 그대로 불러 쓸 수 있는 가중치 폴더로 저장 (설정과 토크나이저는 원본 폴더에서 복사)."""
    import shutil
    from pathlib import Path

    from mlx.utils import tree_flatten

    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(target_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source_dir / name, target_dir / name)


def log_probs(model, ids):
    logits = model(mx.array([ids]))[0, :-1].astype(mx.float32)

    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def evaluate(model, texts_ids, reference=None):
    """perplexity와, reference(원본 log-prob 목록)가 있으면 원본 대비 평균 KL."""
    nll, kl, count = 0.0, 0.0, 0
    for k, ids in enumerate(texts_ids):
        lp = log_probs(model, ids)
        targets = mx.array(ids[1:])
        nll += float(-mx.take_along_axis(lp, targets[:, None], axis=-1).sum())
        if reference is not None:
            ref = reference[k]
            kl += float((mx.exp(ref) * (ref - lp)).sum())
        count += len(ids) - 1

    result = {"ppl": float(mx.exp(mx.array(nll / count)))}
    if reference is not None:
        result["kl"] = kl / count

    return result
