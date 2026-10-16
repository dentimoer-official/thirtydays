import time

# 무거운 import(torch 등)를 흉내: 모듈이 실행될 때 1초 걸림
print("[python] model.py 로드 중... (1초)")
time.sleep(1)

_model = None      # load_model()이 채우는 전역 상태
_calls = 0


def load_model(name):
    global _model
    _model = {"name": name, "scale": 2}
    return f"loaded {name}"


def infer(x):
    global _calls
    if _model is None:
        raise RuntimeError("load_model()을 먼저 호출해야 함")
    _calls += 1
    return x * _model["scale"]


def call_count():
    return _calls