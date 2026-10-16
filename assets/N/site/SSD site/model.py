"""PyTorch 공식 SSDLite320-MobileNetV3 객체 탐지 모델."""

from pathlib import Path

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "ssdlite320_mobilenet_v3_large_coco.pth"


def load_model():
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval()


if __name__ == "__main__":
    print(load_model())
