"""thirtydays 실행 진입점: python3.13 main.py

TIER만 바꾸면 된다.
  1: 적은 램, 적은 성능 (예산 1GB)  ...  4: 많은 램, 강한 성능 (예산 8GB)
명령줄로 덮어쓸 수도 있다: python3.13 main.py 3

이식 스위치 (어느 TIER에서든 켤 수 있음, 먼저 python3.13 prepare.py 필요)
  GRAFT_LLM  : Qwen-Instruct의 8~15층 자리에 Qwen-base의 같은 층을 이식한 모델을 씀
  GRAFT_IMAGE: CLIP 이미지 임베딩을 학습 없이 Qwen 입력(<|image_pad|> 자리)에 직접 이식
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pipeline  # noqa: E402

TIER = 3
GRAFT_LLM = True
GRAFT_IMAGE = True

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "samples" / "bus.jpg"
AUDIO = ROOT / "samples" / "speech_en.wav"
QUESTION = "사진에는 무엇이 있고, 녹음에서는 뭐라고 말했어?"


def show(result):
    cfg = result["config"]
    grafts = [name for name, on in (("LLM 이식", result["graft"]["llm"]), ("이미지 이식", result["graft"]["image"])) if on]
    print(f"===== TIER {result['tier']} (예산 {cfg['budget_gb']}GB, {'전부 올려 두고 실행' if cfg['resident'] else '하나씩 올렸다 내림'}"
          f"{', ' + ', '.join(grafts) if grafts else ''}) =====")

    print("\n[조립 레시피]")
    for name, recipe in result["recipes"].items():
        print(f"  {name:<9} {recipe}")

    print("\n[관찰 -> LLM 입력]")
    for line in result["context"].splitlines():
        print(f"  {line}")

    print("\n[답변]")
    print(f"  {result['observations']['answer']}")

    print("\n[단계별 시간과 메모리]")
    for s in result["stats"]:
        print(f"  {s['part']:<9} {s['phase']:<8} {s['ms']:>8.0f} ms   현재 {s['now_mb']:>6.0f} MB   최대 {s['peak_mb']:>6.0f} MB")

    total_ms = sum(s["ms"] for s in result["stats"])
    ok = result["peak_mb"] <= result["budget_mb"]
    print(f"\n총 {total_ms / 1000:.1f}초, 최대 메모리 {result['peak_mb']:.0f} MB / 예산 {result['budget_mb']:.0f} MB -> {'OK' if ok else '초과'}")

    return ok


if __name__ == "__main__":
    tier = int(sys.argv[1]) if len(sys.argv) > 1 else TIER
    result = pipeline.run(tier, IMAGE, AUDIO, QUESTION, graft_llm=GRAFT_LLM, graft_image=GRAFT_IMAGE)
    sys.exit(0 if show(result) else 1)
