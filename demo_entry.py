"""워커 진입점 예시: load / parts / run_part 세 함수만 맞추면 된다."""

import numpy as np

_scale = None


def load():
    global _scale
    _scale = 2.0


def parts():
    return ["double", "add_one"]


def run_part(name, x):
    x = np.asarray(x, dtype=np.float32)

    if name == "double":
        return x * _scale
    if name == "add_one":
        return x + 1.0

    raise KeyError(f"unknown part: {name}")


def whole(x):
    # 비교 기준: 조각 없이 한 번에 계산
    return np.asarray(x, dtype=np.float32) * _scale + 1.0
