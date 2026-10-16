"""CLIP ViT-B/32 진입점: preprocess -> patchify -> block0~11 -> head -> match(제로샷 분류)."""

import importlib.util
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "ViT-B-32.pt"
LABELS = [
    "bus", "person", "car", "truck", "bicycle", "motorcycle", "train", "airplane",
    "dog", "cat", "bird", "horse", "street", "building", "tree", "food",
]
PROMPT = "a photo of a {}."
TOP_K = 5

_model = None
_text_features = None
_tokenizer = None
_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def import_local(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def tokenize(texts):
    sot = _tokenizer.encoder["<|startoftext|>"]
    eot = _tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), _model.context_length, dtype=torch.long)

    for i, text in enumerate(texts):
        tokens = [sot] + _tokenizer.encode(text) + [eot]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


@torch.no_grad()
def load():
    global _model, _text_features, _tokenizer
    clip_model = import_local("clip_model", "model.py")
    clip_tokenizer = import_local("clip_tokenizer", "simple_tokenizer.py")

    # 공식 가중치는 TorchScript 아카이브. state_dict만 꺼내 구조에 주입하고, CPU용 fp32로 되돌림
    state = torch.jit.load(str(WEIGHTS), map_location="cpu").state_dict()
    _model = clip_model.build_model(state).float().eval()
    _tokenizer = clip_tokenizer.SimpleTokenizer()

    # 라벨 문장은 텍스트 인코더로 미리 임베딩해 둠
    features = _model.encode_text(tokenize([PROMPT.format(label) for label in LABELS]))
    _text_features = features / features.norm(dim=-1, keepdim=True)


def parts():
    blocks = [f"block{i}" for i in range(len(_model.visual.transformer.resblocks))]

    return ["preprocess", "patchify", *blocks, "head", "match"]


def part_modules():
    visual = _model.visual
    modules = {
        "preprocess": [],
        "patchify": [visual.conv1, visual.class_embedding, visual.positional_embedding, visual.ln_pre],
        "head": [visual.ln_post, visual.proj],
        "match": [_model.logit_scale],
    }
    for i, block in enumerate(visual.transformer.resblocks):
        modules[f"block{i}"] = [block]

    return modules


def preprocess(x):
    # 이미지 -> [1, 3, 224, 224]
    image = Image.open(x).convert("RGB") if isinstance(x, str) else x

    return _preprocess(image).unsqueeze(0)


@torch.no_grad()
def patchify(x):
    # 32x32 패치 49개 + CLS 토큰 -> [50, 1, 768] (트랜스포머용 LND 순서)
    visual = _model.visual
    x = visual.conv1(x.type(_model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    return x.permute(1, 0, 2)


@torch.no_grad()
def block(i, x):
    # 블록마다 입출력이 [50, 1, 768]로 같음
    return _model.visual.transformer.resblocks[i](x)


@torch.no_grad()
def head(x):
    # CLS 토큰 -> 이미지 임베딩 [1, 512]
    visual = _model.visual
    x = visual.ln_post(x.permute(1, 0, 2)[:, 0, :])

    return x @ visual.proj


@torch.no_grad()
def match(image_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    probs = (_model.logit_scale.exp() * image_features @ _text_features.t()).softmax(dim=-1)[0]
    top = probs.topk(TOP_K)

    return [{"label": LABELS[i], "score": round(float(p), 4)} for p, i in zip(top.values, top.indices)]


def run_part(name, x):
    if name == "preprocess":
        return preprocess(x)
    if name == "patchify":
        return patchify(x)
    if name.startswith("block"):
        return block(int(name.removeprefix("block")), x)
    if name == "head":
        return head(x)
    if name == "match":
        return match(x)

    raise KeyError(f"unknown part: {name}")


@torch.no_grad()
def whole(x):
    # 비교 기준: encode_image를 통째로 실행
    return match(_model.encode_image(preprocess(x)))
