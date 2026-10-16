# 변수
MODEL_NAME = "demo-model"
SAMPLE_RATE = 44100
THRESHOLD = 0.75
ENABLED = True
LABELS = ["kick", "snare", "hihat"]
CONFIG = {"layers": 12, "heads": 8}
HEADER = b"\x52\x49\x46\x46"  # "RIFF"

# 함수
def get_name():
    return MODEL_NAME

def add(a, b):
    return a + b

def greet(name, times):
    return " ".join([f"hi {name}"] * times)

def encode(text):
    return text.encode("utf-8")

def stats(values):
    return (min(values), max(values), sum(values) / len(values))
    