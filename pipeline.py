"""멀티모달 질의응답 덩어리: 이미지 + 음성 + 질문 -> 답변.

단계(TIERS)마다 해체한 조각을 다른 정밀도와 조합으로 재조립한다.
- MLX: Qwen(LLM), Whisper(받아쓰기), CLIP(이미지 인코더만. 텍스트 인코더는 prepare.py에서 미리 계산해 떼어냄)
- CPU: YOLO, SSDLite(물체 탐지), PANNs, YAMNet(소리 분류), fastText(질문 언어)
하위 단계는 조각을 하나씩 올렸다 내리고(sequential), 4단계는 전부 올려 둔 채 돈다(resident).
"""

import gc
import re
import time
from pathlib import Path

import memory
from common import load_entry

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "N"
CLIP_MLX = ASSETS / "site" / "classify_image" / "models" / "mlx"
IMAGE_TOKEN = "<|image_pad|>"
SILENCE_RMS = 0.003     # 이보다 작은 소리 크기(RMS)는 말소리 없음으로 봄
HISTORY_TURNS = 3       # Qwen에게 같이 넘기는 직전 대화 수
CHAT_CACHE_MB = 64      # 모델을 올려 둔 채로 쓸 때 MLX 계산 캐시 상한

TIERS = {
    1: {"budget_gb": 1, "qwen": "q4", "whisper": "float16", "clip": "float16",
        "detectors": [], "sounds": [], "language_id": True, "resident": False, "max_tokens": 96},
    2: {"budget_gb": 3, "qwen": "q8", "whisper": "float16", "clip": "float16",
        "detectors": ["yolo"], "sounds": [], "language_id": True, "resident": False, "max_tokens": 128},
    3: {"budget_gb": 5, "qwen": "bf16", "whisper": "float32", "clip": "float32",
        "detectors": ["yolo"], "sounds": ["panns"], "language_id": True, "resident": False, "max_tokens": 192},
    4: {"budget_gb": 8, "qwen": "bf16", "whisper": "float32", "clip": "float32",
        "detectors": ["yolo", "ssdlite"], "sounds": ["panns", "yamnet"], "language_id": True, "resident": True, "max_tokens": 256},
}

ENTRIES = {
    "yolo": ASSETS / "site" / "yolo" / "entry.py",
    "ssdlite": ASSETS / "site" / "SSD site" / "entry.py",
    "panns": ASSETS / "sound" / "PANN" / "entry.py",
    "yamnet": ASSETS / "sound" / "surrounding" / "entry.py",
    "fasttext": ASSETS / "text" / "fastText" / "entry.py",
}

SYSTEM = "You answer questions about an image and an audio recording using only the observations given."

# fastText 언어 코드 -> 답변 언어 지시에 쓰는 이름
LANGUAGE_NAMES = {
    "ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "ru": "Russian",
}


# ---- 부품: load()는 무거운 것을 올리고, run()은 관찰 결과를 돌려준다 ----

class MlxPart:
    def release(self):
        import mlx.core as mx
        self.model = None
        gc.collect()
        mx.clear_cache()


class ClipPart(MlxPart):
    name = "clip"

    def __init__(self, dtype):
        self.dtype = dtype
        self.recipe = f"이미지 인코더만 ({dtype}), 텍스트 인코더는 미리 계산한 라벨 임베딩으로 대체"

    def load(self):
        from mlx_models.clip import Clip
        self.model = Clip(CLIP_MLX / self.dtype)

    def run(self, inputs):
        # 이미지 임베딩은 이미지 이식(GRAFT_IMAGE)에서 Qwen이 다시 씀
        features = self.model.embed(inputs["image"])
        inputs["image_features"] = features

        return {"scene": self.model.match(features, top_k=3)}


class WhisperPart(MlxPart):
    name = "whisper"

    def __init__(self, dtype):
        self.dtype = dtype
        self.recipe = f"mel(numpy) -> encoder -> decoder ({dtype})"

    def load(self):
        from mlx_models import whisper
        self.model = whisper.load(ASSETS / "sound" / "whisper" / "models" / "mlx" / self.dtype)

    def run(self, inputs):
        import numpy as np

        from mlx_models import whisper

        # Whisper는 조용한 소리에서 "Thank you." 같은 말을 지어내므로, 너무 작으면 받아쓰지 않음
        audio = whisper.read_audio(inputs["audio"])
        level = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if level < SILENCE_RMS:
            return {"speech": {"language": None, "text": "", "level": level}}

        return {"speech": {**whisper.transcribe(self.model, audio), "level": level}}


class QwenPart(MlxPart):
    name = "qwen"

    def __init__(self, variant, max_tokens, graft_llm=False, graft_image=False):
        self.variant = f"graft-{variant}" if graft_llm else variant
        self.max_tokens = max_tokens
        self.graft_image = graft_image

        layers = "24층 (8~15층 어텐션은 base에서 이식)" if graft_llm else "24층"
        self.recipe = f"embed -> {layers} -> norm -> lm_head(임베딩 공유) ({variant}), 최대 {max_tokens}토큰"
        if graft_image:
            self.recipe += ", <|image_pad|> 자리에 CLIP 이미지 임베딩 이식 (개수 질문은 제외)"

    def load(self):
        import json

        from mlx_models import qwen2
        self.model, self.tokenizer = qwen2.load(ASSETS / "text" / "Qwen" / "models" / "mlx" / self.variant)
        if self.graft_image:
            from mlx_models.bridge import Bridge
            logit_scale = json.loads((CLIP_MLX / "labels.json").read_text())["logit_scale"]
            self.bridge = Bridge(CLIP_MLX / "bridge", self.model, logit_scale)

    def run(self, inputs):
        from mlx_models import qwen2

        # 개수 질문에는 이미지 벡터가 숫자를 흔들어서(실사용 채점), 탐지기 숫자만 씀
        use_image = self.graft_image and not asks_count(inputs["question"])
        inputs["graft_image"] = use_image
        inject = None
        if use_image:
            # 이미지와 가까운 닻 단어 5개의 Qwen 임베딩을 CLIP 유사도로 섞은 벡터
            inject = {IMAGE_TOKEN: self.bridge.soft(inputs["image_features"], top_k=5)}

        context = build_context(inputs)
        history = inputs.get("history", [])[-HISTORY_TURNS:]
        answer = qwen2.generate(
            self.model, self.tokenizer, qwen2.chat_prompt(context, system=SYSTEM, history=history),
            max_tokens=self.max_tokens, inject=inject,
        )
        # 대화 기록이 쌓이면 한국어 질문에 한자가 섞이는 일이 있어서, 그때는 기록 없이 한 번 더
        if history and inputs["observations"].get("question_language") == "ko" and HAN.search(answer):
            answer = qwen2.generate(
                self.model, self.tokenizer, qwen2.chat_prompt(context, system=SYSTEM),
                max_tokens=self.max_tokens, inject=inject,
            )

        return {"answer": answer}

    def release(self):
        self.tokenizer = None
        self.bridge = None
        super().release()


class CpuPart:
    """assets의 entry.py를 조각(parts) 순서대로 이어 실행하는 부품."""

    def __init__(self, name, key):
        self.name = name
        self.key = key
        self.entry = None

    @property
    def recipe(self):
        return " -> ".join(self.entry.parts()) if self.entry else "(로드 전)"

    def load(self):
        self.entry = load_entry(self.name, ENTRIES[self.name])

    def run(self, inputs):
        x = inputs[self.key]
        for part in self.entry.parts():
            x = self.entry.run_part(part, x)

        return {self.name: x}

    def release(self):
        self.entry = None
        gc.collect()


def build_parts(cfg, graft_llm=False, graft_image=False):
    parts = [ClipPart(cfg["clip"])]
    parts += [CpuPart(name, "image") for name in cfg["detectors"]]
    parts += [WhisperPart(cfg["whisper"])]
    parts += [CpuPart(name, "audio") for name in cfg["sounds"]]
    if cfg["language_id"]:
        parts += [CpuPart("fasttext", "question")]
    parts += [QwenPart(cfg["qwen"], cfg["max_tokens"], graft_llm, graft_image)]

    return parts


HAN = re.compile(r"[一-鿿]")
COUNT_QUESTION = re.compile(r"몇\s*(개|대|명|마리|채|권|장|그루)|how many|number of", re.IGNORECASE)


def asks_count(question):
    return bool(COUNT_QUESTION.search(question))


# ---- 관찰 결과 합치기 ----

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter

    return inter / union if union > 0 else 0.0


def merge_detections(groups, iou_min=0.5):
    # 여러 탐지기의 박스를 모아 같은 라벨끼리 겹치면 점수 높은 쪽만 남김
    boxes = sorted((d for g in groups for d in g), key=lambda d: -d["score"])
    kept = []
    for d in boxes:
        if all(k["label"] != d["label"] or iou(k["box"], d["box"]) < iou_min for k in kept):
            kept.append(d)

    return kept


def merge_tags(groups, top_k=5):
    # 여러 소리 분류기의 같은 라벨 점수를 평균
    scores = {}
    for g in groups:
        for t in g:
            scores.setdefault(t["label"], []).append(t["score"])
    merged = sorted(((label, sum(s) / len(groups)) for label, s in scores.items()), key=lambda x: -x[1])

    return [{"label": label, "score": round(score, 4)} for label, score in merged[:top_k]]


def count_objects(detections):
    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    return counts


PLURALS = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis", "scissors": "scissors"}


def plural(label):
    if label in PLURALS:
        return PLURALS[label]
    if label.endswith(("s", "x", "ch", "sh")):
        return label + "es"
    if label.endswith("y") and label[-2:-1] not in "aeiou":
        return label[:-1] + "ies"

    return label + "s"


def phrase(label, n):
    if n > 1:
        return plural(label)

    return f"an {label}" if label[0] in "aeiou" else f"a {label}"


def build_context(inputs):
    # 0.5B LLM은 줄 하나에도 답이 흔들려서, 여러 형식을 시험해 가장 안정적인 배치를 골랐다.
    # - 관찰마다 역할을 이름 붙임: 받아쓰기는 "실제로 말한 문장"
    # - 물체 개수는 물체마다 한 줄씩 ("Number of buses: 1")
    # - 소리 분류는 받아쓰기와 섞이지 않게 별도 섹션
    # - 탐지가 없는 1단계(4비트)는 장면 후보 3개와 확률이 있을 때 가장 안정적이었음
    obs = inputs["observations"]
    lines = ["Here is what other models found.", "", "IMAGE"]
    # 이 자리의 임베딩이 CLIP 이미지 임베딩에서 옮겨 온 벡터로 바뀜
    image_line = f"- The image itself: {IMAGE_TOKEN}"
    image_mode = inputs.get("image_mode", "first") if inputs.get("graft_image") else "off"
    if image_mode == "first":
        lines.append(image_line)
    if obs.get("objects"):
        counts = sorted(obs["objects"].items(), key=lambda item: -item[1])
        names = [phrase(label, n) for label, n in counts]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        lines.append(f"- Objects in the photo: {listed}")
        # "Number of buses: 1" 같은 표기는 bf16 이식본에서 앞 숫자(사람 4)에 끌려갔고,
        # 문장형은 시험한 네 모델(원본/이식 x 8비트/bf16) 모두에서 개수를 맞혔다
        lines += [
            f"- There is 1 {label} in the photo." if n == 1 else f"- There are {n} {plural(label)} in the photo."
            for label, n in counts
        ]
        lines.append(f"- The photo looks like: {phrase(obs['scene'][0]['label'], 1)}")
    elif "objects" in obs:
        # 탐지기는 돌았지만 아는 물체를 못 찾음
        lines.append("- No common objects were detected in the photo.")
        lines.append("- The photo looks like: " + ", ".join(f"{s['label']} ({s['score']:.0%})" for s in obs["scene"]))
    else:
        lines.append("- The photo looks like: " + ", ".join(f"{s['label']} ({s['score']:.0%})" for s in obs["scene"]))
        # 탐지기가 없는 단계에서 개수를 지어내지 않도록 명시
        lines.append("- Object counts: unknown (no object detector was used)")
    if image_mode == "last":
        lines.append(image_line)

    lines += ["", "AUDIO"]
    if obs["speech"]["text"]:
        lines.append(f"- Exact words spoken in the recording: \"{obs['speech']['text']}\"")
    else:
        # "No speech was detected"는 다른 관찰을 녹음 내용으로 지어내게 했고, 이 문구가 가장 안정적이었음
        lines.append("- The recording is silent. Nobody said anything.")
    if obs.get("sounds"):
        lines += ["", "SOUND CATEGORY", "- " + ", ".join(t["label"].lower() for t in obs["sounds"][:2])]

    lines += ["", f"Question: {inputs['question']}"]
    language = obs.get("question_language")
    if language:
        lines.append(f"Answer only in {LANGUAGE_NAMES.get(language, language)}.")
    else:
        lines.append("Answer in the same language as the question.")

    return "\n".join(lines)


def collect(inputs, part_name, result):
    obs = inputs["observations"]
    if part_name in ("yolo", "ssdlite"):
        inputs.setdefault("_detections", []).append(result[part_name])
        obs["objects"] = count_objects(merge_detections(inputs["_detections"]))
    elif part_name in ("panns", "yamnet"):
        inputs.setdefault("_sounds", []).append(result[part_name])
        obs["sounds"] = merge_tags(inputs["_sounds"])
    elif part_name == "fasttext":
        obs["question_language"] = result["fasttext"][0]["label"]
    else:
        obs.update(result)


# ---- 실행 ----

class Session:
    """부품을 조립해 두고 질문마다 돌린다.

    keep_loaded=True면 처음에 전부 올려 두고 계속 씀(채팅용, 4단계 기본).
    False면 질문마다 하나씩 올리고, 실행하고, 내림(1~3단계 기본).
    """

    def __init__(self, tier, graft_llm=False, graft_image=False, keep_loaded=None):
        self.tier = tier
        self.cfg = TIERS[tier]
        self.graft = {"llm": graft_llm, "image": graft_image}
        self.parts = build_parts(self.cfg, graft_llm, graft_image)
        self.keep_loaded = self.cfg["resident"] if keep_loaded is None else keep_loaded
        self.load_stats = []
        if self.keep_loaded:
            # 계속 올려 두고 쓸 때는 MLX가 쌓아 두는 계산용 캐시(수백 MB)를 작게 묶어 둠
            import mlx.core as mx
            mx.set_cache_limit(CHAT_CACHE_MB * memory.MB)
            for part in self.parts:
                self._step(self.load_stats, part, "load", part.load)

    @staticmethod
    def _step(stats, part, phase, fn):
        start = time.perf_counter()
        fn()
        stats.append({
            "part": part.name, "phase": phase, "ms": (time.perf_counter() - start) * 1000,
            "now_mb": memory.current() / memory.MB, "peak_mb": memory.peak() / memory.MB,
        })

    def ask(self, image, audio, question="", history=None, with_llm=True):
        """with_llm=False면 Qwen 없이 사진과 소리 관찰만 돌림."""
        inputs = {
            "image": str(image), "audio": str(audio), "question": question,
            "history": history or [], "graft_image": self.graft["image"], "observations": {},
        }
        parts = self.parts if with_llm else [p for p in self.parts if p.name != "qwen"]
        stats = []

        for part in parts:
            if not self.keep_loaded:
                self._step(stats, part, "load", part.load)
            self._step(stats, part, "run", lambda p=part: collect(inputs, p.name, p.run(inputs)))
            if not self.keep_loaded:
                part.recipe_used = part.recipe
                self._step(stats, part, "release", part.release)

        return {
            "tier": self.tier,
            "config": self.cfg,
            "graft": self.graft,
            "recipes": {p.name: getattr(p, "recipe_used", None) or p.recipe for p in self.parts},
            "observations": inputs["observations"],
            "context": build_context(inputs) if with_llm else None,
            "stats": stats,
            "peak_mb": memory.peak() / memory.MB,
            "budget_mb": self.cfg["budget_gb"] * 1024,
        }

    def close(self):
        for part in self.parts:
            part.release()


def run(tier, image, audio, question, graft_llm=False, graft_image=False, log=print):
    session = Session(tier, graft_llm, graft_image)
    result = session.ask(image, audio, question)
    result["stats"] = session.load_stats + result["stats"]

    return result
