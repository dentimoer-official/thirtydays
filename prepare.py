"""MLX용 파생 가중치를 만드는 준비 단계: python3.13 prepare.py

원본 가중치(assets/N/**/models)에서 단계별로 쓰는 MLX 가중치를 만든다. 이미 있으면 건너뛴다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "N"

QWEN_SOURCE = ASSETS / "text" / "Qwen" / "models" / "Qwen2.5-0.5B-Instruct"
QWEN_MLX = ASSETS / "text" / "Qwen" / "models" / "mlx"


def prepare_qwen():
    from mlx_lm import convert

    variants = {
        "q4": {"quantize": True, "q_bits": 4, "q_group_size": 64},
        "q8": {"quantize": True, "q_bits": 8, "q_group_size": 64},
        "bf16": {"dtype": "bfloat16"},
    }
    for name, options in variants.items():
        target = QWEN_MLX / name
        if target.exists():
            print(f"qwen {name}: 있음")
            continue
        convert(str(QWEN_SOURCE), mlx_path=str(target), **options)
        print(f"qwen {name}: 생성")

    # 이식용 기증자: 같은 구조의 base 모델을 수혜자와 같은 정밀도로 (받아 둔 경우에만)
    base = QWEN_SOURCE.parent / "Qwen2.5-0.5B"
    if not (base / "model.safetensors").exists():
        return
    for name, options in variants.items():
        target = QWEN_MLX / f"base-{name}"
        if not target.exists():
            convert(str(base), mlx_path=str(target), **options)
            print(f"qwen base-{name}: 생성")


GRAFT_LAYERS = range(8, 16)
# 실사용 채점에서 층 통째("layer")나 MLP 이식은 지시 따르기(답변 언어 등)를 잃었고,
# 어텐션만 이식("attn")은 원본과 같은 점수를 유지했다.
GRAFT_ACTION = "attn"


def prepare_graft():
    """수혜자(Instruct)의 8~15층 자리에 기증자(base)의 같은 층을 이식한 완성품을 정밀도별로 저장."""
    import gc

    from mlx_models import qwen2, surgery

    for name in ("q4", "q8", "bf16"):
        target = QWEN_MLX / f"graft-{name}"
        if target.exists():
            print(f"qwen graft-{name}: 있음")
            continue
        model, _ = qwen2.load(QWEN_MLX / name)
        donor, _ = qwen2.load(QWEN_MLX / f"base-{name}")
        with surgery.operated(model, donor, {i: GRAFT_ACTION for i in GRAFT_LAYERS}):
            surgery.save(model, QWEN_MLX / name, target)
        del model, donor
        gc.collect()
        print(f"qwen graft-{name}: 생성 ({GRAFT_LAYERS.start}~{GRAFT_LAYERS.stop - 1}층 {GRAFT_ACTION} 이식)")


WHISPER_SOURCE = ASSETS / "sound" / "whisper" / "models" / "tiny.pt"
WHISPER_MLX = ASSETS / "sound" / "whisper" / "models" / "mlx"


def prepare_whisper():
    import json

    import mlx.core as mx
    import torch

    checkpoint = torch.load(WHISPER_SOURCE, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]

    weights = {}
    for key, value in state.items():
        value = value.float().numpy()
        # torch Conv1d [out, in, kernel] -> MLX Conv1d [out, kernel, in]
        if key.endswith(("conv1.weight", "conv2.weight")):
            value = value.transpose(0, 2, 1)
        # nn.Sequential(Linear, GELU, Linear)의 0, 2번 -> mlp1, mlp2
        weights[key.replace(".mlp.0.", ".mlp1.").replace(".mlp.2.", ".mlp2.")] = mx.array(value)

    for dtype in ("float32", "float16"):
        target = WHISPER_MLX / dtype
        if (target / "weights.safetensors").exists():
            print(f"whisper {dtype}: 있음")
            continue
        target.mkdir(parents=True, exist_ok=True)
        cast = {k: v.astype(getattr(mx, dtype)) for k, v in weights.items()}
        mx.save_safetensors(str(target / "weights.safetensors"), cast)
        (target / "dims.json").write_text(json.dumps(checkpoint["dims"]))
        print(f"whisper {dtype}: 생성")


CLIP_DIR = ASSETS / "site" / "classify_image"
CLIP_MLX = CLIP_DIR / "models" / "mlx"
CLIP_SCENES = [
    "street", "road", "city", "building", "park", "beach", "forest", "mountain",
    "kitchen", "office", "bedroom", "living room", "classroom", "restaurant", "store", "sky",
]


def prepare_clip():
    import importlib.util
    import json

    import mlx.core as mx
    import numpy as np
    import torch
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    if (CLIP_MLX / "float16" / "weights.safetensors").exists():
        print("clip: 있음")
        return

    spec = importlib.util.spec_from_file_location("clip_entry", CLIP_DIR / "entry.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.load()
    model = entry._model

    # 텍스트 인코더는 여기서 라벨 문장을 미리 임베딩하고, 실행 때는 떼어냄
    coco = [c for c in SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.meta["categories"] if c not in ("__background__", "N/A")]
    labels = coco + CLIP_SCENES
    with torch.no_grad():
        features = model.encode_text(entry.tokenize([entry.PROMPT.format(label) for label in labels]))
        features = features / features.norm(dim=-1, keepdim=True)

    # 이미지 인코더만 MLX 가중치로
    weights = {}
    for key, value in model.visual.state_dict().items():
        value = value.float().numpy()
        # torch Conv2d [out, in, kh, kw] -> MLX Conv2d [out, kh, kw, in]
        if key == "conv1.weight":
            value = value.transpose(0, 2, 3, 1)
        weights[key] = mx.array(value)

    for dtype in ("float32", "float16"):
        target = CLIP_MLX / dtype
        target.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(target / "weights.safetensors"), {k: v.astype(getattr(mx, dtype)) for k, v in weights.items()})

    np.save(CLIP_MLX / "text_features.npy", features.numpy().astype(np.float32))
    (CLIP_MLX / "labels.json").write_text(json.dumps({
        "labels": labels,
        "prompt": entry.PROMPT,
        "logit_scale": float(model.logit_scale.exp()),
        "width": model.visual.conv1.out_channels,
        "patch": model.visual.conv1.kernel_size[0],
        "layers": len(model.visual.transformer.resblocks),
        "resolution": model.visual.input_resolution,
    }, ensure_ascii=False))
    print(f"clip: 생성 (라벨 {len(labels)}개)")


BRIDGE = CLIP_MLX / "bridge"
BRIDGE_ANCHORS = 8000


def prepare_bridge():
    """CLIP -> Qwen 이미지 다리용 닻: Qwen 어휘의 영어 단어 토큰과 그 CLIP 텍스트 임베딩."""
    import importlib.util
    import json

    import numpy as np
    import torch
    from tokenizers import Tokenizer

    if (BRIDGE / "clip_text.npy").exists():
        print("bridge: 있음")
        return

    # Qwen 어휘에서 "공백 + 영어 소문자 단어" 토큰을 id가 작은(자주 쓰이는) 순서로
    tokenizer = Tokenizer.from_file(str(QWEN_SOURCE / "tokenizer.json"))
    candidates = []
    for token, token_id in tokenizer.get_vocab().items():
        word = token[1:]
        if token.startswith("Ġ") and word.isascii() and word.isalpha() and word.islower() and 3 <= len(word) <= 15:
            candidates.append((token_id, word))
    candidates = sorted(candidates)[:BRIDGE_ANCHORS]

    spec = importlib.util.spec_from_file_location("clip_entry", CLIP_DIR / "entry.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.load()

    features = []
    with torch.no_grad():
        for start in range(0, len(candidates), 256):
            batch = [entry.PROMPT.format(word) for _, word in candidates[start:start + 256]]
            f = entry._model.encode_text(entry.tokenize(batch))
            features.append(f / f.norm(dim=-1, keepdim=True))

    BRIDGE.mkdir(parents=True, exist_ok=True)
    np.save(BRIDGE / "clip_text.npy", torch.cat(features).numpy().astype(np.float32))
    np.save(BRIDGE / "anchor_ids.npy", np.array([i for i, _ in candidates], dtype=np.int64))
    (BRIDGE / "anchor_words.json").write_text(json.dumps([w for _, w in candidates]))
    print(f"bridge: 생성 (닻 {len(candidates)}개)")


STEPS = {
    "qwen": prepare_qwen,
    "whisper": prepare_whisper,
    "clip": prepare_clip,
    "bridge": prepare_bridge,
    "graft": prepare_graft,
}


if __name__ == "__main__":
    for name in sys.argv[1:] or list(STEPS):
        STEPS[name]()
