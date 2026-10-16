"""이식 실험: python3.13 transplant.py [llm|image]

llm  : Qwen2.5-0.5B-Instruct(수혜자)에 구멍을 뚫고, Qwen2.5-0.5B(기증자, 같은 구조의 base)의
       같은 자리 부위를 이식해서 원본, 구멍만 뚫은 것, 이식한 것을 비교한다.
         ppl: 평가 문장 perplexity (낮을수록 좋음)
         kl : 원본 Instruct와 다음 토큰 분포가 얼마나 달라졌는지 (0이면 원본과 같음)
image: CLIP 이미지 인코더의 출력을 학습 없이 Qwen 입력 자리에 이식해서, 사진을 "보고" 답하는지 본다.
"""

import sys
from pathlib import Path

import mlx.core as mx

from mlx_models import qwen2
from mlx_models.surgery import evaluate, log_probs, operated

ROOT = Path(__file__).resolve().parent
QWEN = ROOT / "assets" / "N" / "text" / "Qwen" / "models" / "mlx"
RECIPIENT = QWEN / "bf16"
DONOR = QWEN / "base-bf16"
CLIP_MLX = ROOT / "assets" / "N" / "site" / "classify_image" / "models" / "mlx"
IMAGES = [ROOT / "samples" / "bus.jpg", ROOT / "samples" / "zidane.jpg"]
IMAGE_QUESTIONS = [
    "Here is an image: <|image_pad|>\nWhat is in the image? Answer in one short sentence.",
    "이미지: <|image_pad|>\n이 이미지에 무엇이 있어? 한 문장으로 답해.",
]

BLOCK = range(8, 16)

EVAL_TEXTS = [
    "The bus stopped at the corner of the street, and four people stepped off into the cold morning air. "
    "One of them checked a map, while the others looked around for a place to buy coffee before work.",
    "Paris is the capital of France. It is known for the Eiffel Tower, the Louvre museum, and its cafes. "
    "The river Seine flows through the middle of the city.",
    "서울은 대한민국의 수도이며, 한강이 도시 가운데를 흐른다. 많은 사람들이 지하철과 버스를 타고 출근한다.",
    qwen2.chat_prompt("What is the capital of France?") + "The capital of France is Paris.<|im_end|>",
    qwen2.chat_prompt("사람은 몇 명이야? 사진에는 사람 4명과 버스 1대가 있어.") + "사진에는 사람이 4명 있습니다.<|im_end|>",
    "def square(n):\n    \"\"\"Return the square of n.\"\"\"\n    return n * n\n\n\nprint(square(4))\n",
]

PROMPTS = [
    "What is the capital of France?",
    "사진에 버스 한 대와 사람 네 명이 있어. 한 문장으로 설명해줘.",
    "Write a Python function that returns the square of a number.",
]


def row(name, r):
    kl = f"{r['kl']:.4f}" if "kl" in r else "-"
    print(f"  {name:<34} ppl {r['ppl']:>8.2f}   kl {kl:>8}")


def image_graft():
    from mlx_models.bridge import Bridge
    from mlx_models.clip import Clip

    model, tokenizer = qwen2.load(RECIPIENT)
    clip = Clip(CLIP_MLX / "float32")
    bridge = Bridge(CLIP_MLX / "bridge", model, clip.logit_scale)

    accuracy = bridge.fit_linear()
    print(f"== 닻 {len(bridge.words)}개, 선형 변환이 남겨 둔 단어 500개 중 맞는 Qwen 단어를 찾은 비율: {accuracy:.1%}")

    for image in IMAGES:
        features = clip.embed(str(image))
        print(f"\n== {image.name}  (CLIP 기준 가까운 닻: {bridge.nearest(features)})")
        grafts = {
            "대조군(이식 없음)": None,
            "soft": bridge.soft(features),
            "soft top-5": bridge.soft(features, top_k=5),
            "linear": bridge.linear(features),
        }
        for question in IMAGE_QUESTIONS:
            print(f"  Q: {question.splitlines()[-1]}")
            for name, vector in grafts.items():
                inject = {"<|image_pad|>": vector} if vector is not None else None
                answer = qwen2.generate(model, tokenizer, qwen2.chat_prompt(question), max_tokens=40, inject=inject)
                print(f"    [{name}] {answer.replace(chr(10), ' ')[:120]}")


def llm_graft():
    model, tokenizer = qwen2.load(RECIPIENT)
    donor, _ = qwen2.load(DONOR)
    ids = [tokenizer.encode(t).ids for t in EVAL_TEXTS]
    reference = [log_probs(model, i) for i in ids]
    mx.eval(reference)

    print("== 기준")
    row("수혜자: Qwen2.5-0.5B-Instruct", evaluate(model, ids, reference))
    row("기증자: Qwen2.5-0.5B (base)", evaluate(donor, ids, reference))

    print("\n== 층 하나씩: 구멍만 뚫기 vs 기증자 층 이식")
    print(f"  {'층':<4} {'구멍 ppl':>9} {'구멍 kl':>9}   {'이식 ppl':>9} {'이식 kl':>9}")
    for i in range(len(model.model.layers)):
        with operated(model, donor, {i: "remove"}):
            hole = evaluate(model, ids, reference)
        with operated(model, donor, {i: "layer"}):
            graft = evaluate(model, ids, reference)
        print(f"  {i:<4} {hole['ppl']:>9.2f} {hole['kl']:>9.4f}   {graft['ppl']:>9.2f} {graft['kl']:>9.4f}")

    print(f"\n== 가운데 {len(BLOCK)}개 층({BLOCK.start}~{BLOCK.stop - 1}) 한꺼번에")
    plans = {
        "구멍만 (8개 층 제거)": "remove",
        "층 통째 이식": "layer",
        "어텐션만 이식": "attn",
        "MLP만 이식": "mlp",
    }
    for name, action in plans.items():
        with operated(model, donor, {i: action for i in BLOCK}):
            row(name, evaluate(model, ids, reference))

    print("\n== 실제 생성")
    configs = {
        "원본": {},
        f"구멍 {BLOCK.start}~{BLOCK.stop - 1}": {i: "remove" for i in BLOCK},
        f"이식 {BLOCK.start}~{BLOCK.stop - 1}": {i: "layer" for i in BLOCK},
    }
    for prompt in PROMPTS:
        print(f"\n  Q: {prompt}")
        for name, plan in configs.items():
            with operated(model, donor, plan):
                answer = qwen2.generate(model, tokenizer, qwen2.chat_prompt(prompt), max_tokens=60)
            print(f"    [{name}] {answer.replace(chr(10), ' ')[:150]}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["llm", "image"]
    if "llm" in targets:
        llm_graft()
    if "image" in targets:
        image_graft()
