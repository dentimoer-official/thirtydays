"""YOLO26n 진입점: preprocess -> backbone -> neck -> detect -> postprocess."""

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "yolo26n.pt"
IMAGE_SIZE = 640
CONF_MIN = 0.25
IOU_MAX = 0.45

_net = None
_spans = {}


def load():
    global _net
    _net = YOLO(str(WEIGHTS)).model.eval()

    # yaml의 backbone 길이로 경계를 정함. 마지막 층은 Detect
    n_backbone = len(_net.yaml["backbone"])
    n_layers = len(_net.model)
    _spans.update({
        "backbone": (0, n_backbone),
        "neck": (n_backbone, n_layers - 1),
        "detect": (n_layers - 1, n_layers),
    })


def parts():
    return ["preprocess", "backbone", "neck", "detect", "postprocess"]


def part_modules():
    modules = {name: list(_net.model[start:end]) for name, (start, end) in _spans.items()}

    return {"preprocess": [], **modules, "postprocess": []}


@torch.no_grad()
def preprocess(x):
    # 이미지 -> letterbox 640x640, RGB, 0~1 [1, 3, 640, 640]
    image = cv2.imread(x)
    boxed = LetterBox(new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=False)(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0

    return {"x": tensor.unsqueeze(0), "saved": [], "original_shape": image.shape[:2]}


@torch.no_grad()
def run_layers(p, name):
    # tasks.py의 _predict_once와 같은 방식: m.f로 앞 층 출력을 끌어오고, 필요한 출력은 saved에 쌓음
    x, saved = p["x"], list(p["saved"])
    start, end = _spans[name]

    for m in _net.model[start:end]:
        if m.f != -1:
            x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
        x = m(x)
        saved.append(x if m.i in _net.save else None)

    return {**p, "x": x, "saved": saved}


def postprocess(p):
    # Detect 출력은 (예측, 보조 출력) 튜플. 예측 [1, 84, 8400]만 NMS에 넣음
    raw = p["x"][0] if isinstance(p["x"], (list, tuple)) else p["x"]

    return summarize(raw, p["original_shape"])


def summarize(raw, original_shape):
    det = non_max_suppression(raw, conf_thres=CONF_MIN, iou_thres=IOU_MAX)[0]
    det[:, :4] = scale_boxes((IMAGE_SIZE, IMAGE_SIZE), det[:, :4], original_shape)

    return [
        {"label": _net.names[int(cls)], "score": round(float(conf), 4), "box": [round(float(v), 1) for v in box]}
        for *box, conf, cls in det.tolist()
    ]


def run_part(name, x):
    if name == "preprocess":
        return preprocess(x)
    if name in _spans:
        return run_layers(x, name)
    if name == "postprocess":
        return postprocess(x)

    raise KeyError(f"unknown part: {name}")


@torch.no_grad()
def whole(x):
    # 비교 기준: DetectionModel을 통째로 실행
    p = preprocess(x)
    out = _net(p["x"])

    return summarize(out[0] if isinstance(out, (list, tuple)) else out, p["original_shape"])
