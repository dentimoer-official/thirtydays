"""SSDLite320-MobileNetV3 진입점: preprocess -> backbone -> head -> postprocess."""

from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "ssdlite320_mobilenet_v3_large_coco.pth"
CATEGORIES = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.meta["categories"]
SCORE_MIN = 0.5

_model = None


def load():
    global _model
    _model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    _model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    _model.eval()


def parts():
    return ["preprocess", "backbone", "head", "postprocess"]


def part_modules():
    return {
        "preprocess": [],
        "backbone": [_model.backbone],
        "head": [_model.head],
        "postprocess": [],
    }


def read_image(x):
    return to_tensor(Image.open(x).convert("RGB")) if isinstance(x, str) else x


@torch.no_grad()
def preprocess(x):
    # 이미지 -> 320x320 정규화 텐서 [1, 3, 320, 320]
    image = read_image(x)
    original_sizes = [tuple(image.shape[-2:])]
    images, _ = _model.transform([image])

    return {"x": images.tensors, "images": images, "original_sizes": original_sizes}


@torch.no_grad()
def backbone(p):
    # 6개 스케일의 특징 맵
    features = list(_model.backbone(p["x"]).values())

    return {**p, "x": features}


@torch.no_grad()
def head(p):
    # 앵커별 클래스 점수와 박스 회귀 [1, 3234, 91], [1, 3234, 4]
    outputs = _model.head(p["x"])

    return {**p, "x": outputs["cls_logits"], "head_outputs": outputs, "features": p["x"]}


@torch.no_grad()
def postprocess(p):
    images = p["images"]
    anchors = _model.anchor_generator(images, p["features"])
    detections = _model.postprocess_detections(p["head_outputs"], anchors, images.image_sizes)
    detections = _model.transform.postprocess(detections, images.image_sizes, p["original_sizes"])

    return summarize(detections[0])


def summarize(det):
    keep = det["scores"] >= SCORE_MIN

    return [
        {"label": CATEGORIES[int(label)], "score": round(float(score), 4), "box": [round(float(v), 1) for v in box]}
        for label, score, box in zip(det["labels"][keep], det["scores"][keep], det["boxes"][keep])
    ]


def run_part(name, x):
    if name == "preprocess":
        return preprocess(x)
    if name == "backbone":
        return backbone(x)
    if name == "head":
        return head(x)
    if name == "postprocess":
        return postprocess(x)

    raise KeyError(f"unknown part: {name}")


@torch.no_grad()
def whole(x):
    # 비교 기준: torchvision 모델을 통째로 실행
    return summarize(_model([read_image(x)])[0])
