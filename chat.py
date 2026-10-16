"""실시간 채팅: python3.13 chat.py

카메라와 마이크가 뒤에서 계속 켜져 있고, 메시지를 보내는 순간의 카메라 화면 한 장과
직전 몇 초의 소리를 가져다 답한다. 엔터만 치면 Qwen 없이 지금 보고 듣는 것만 보여준다.

옵션
  --tier N          시작 단계 (기본 2)
  --graft           이식 켜고 시작 (LLM 이식 + 이미지 이식)
  --listen 초       메시지마다 쓰는 직전 소리 길이 (기본 5초)
  --camera 번호     카메라 장치 번호 (기본 0)
  --mic 번호|이름   마이크 장치 (기본: 시스템 기본 입력)
  --list-devices    마이크 장치 목록만 보고 끝냄
  --no-camera       카메라 대신 --image 파일 사용
  --no-mic          마이크 대신 --audio 파일 사용

채팅 중 명령: /tier N, /graft on|off, /listen 초, /save, /reset, /help, /quit
"""

import argparse
import collections
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import memory  # noqa: E402
import pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent
SAVED = ROOT / "samples" / "live"
SAMPLE_RATE = 16000
HELP = """명령
  (엔터만)        Qwen 없이 지금 보고 듣는 것만 보여줌
  /tier N         단계 바꾸기 (1~4, 모델을 다시 올림)
  /graft on|off   이식 켜기/끄기 (LLM 이식 + 이미지 이식)
  /listen 초      메시지마다 쓰는 직전 소리 길이
  /save           마지막에 쓴 사진과 소리를 samples/live/에 저장
  /reset          대화 기록 지우기
  /quit           끝내기"""


class Camera:
    """뒤에서 계속 프레임을 읽어 두고, 요청하면 가장 최근 프레임을 저장한다."""

    def __init__(self, index):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("카메라를 열 수 없어요")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def snapshot(self, path, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                frame = self.frame
            if frame is not None:
                self.cv2.imwrite(str(path), frame)
                return path
            time.sleep(0.05)

        raise RuntimeError("카메라 화면을 받지 못했어요 (권한을 확인해 주세요)")

    def close(self):
        self.running = False
        self.cap.release()


class Microphone:
    """뒤에서 계속 녹음해 최근 소리만 들고 있다가, 요청하면 직전 몇 초를 저장한다."""

    def __init__(self, device=None, keep_seconds=30):
        import sounddevice as sd

        self.chunks = collections.deque()
        self.samples = 0
        self.keep = keep_seconds * SAMPLE_RATE
        self.lock = threading.Lock()
        self.stream = sd.InputStream(
            device=device, samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._callback,
        )
        self.name = sd.query_devices(self.stream.device)["name"]
        self.stream.start()

    def _callback(self, data, frames, time_info, status):
        with self.lock:
            self.chunks.append(data[:, 0].copy())
            self.samples += frames
            while self.samples - len(self.chunks[0]) > self.keep:
                self.samples -= len(self.chunks.popleft())

    def snapshot(self, path, seconds):
        with self.lock:
            audio = np.concatenate(list(self.chunks)) if self.chunks else np.zeros(0, dtype=np.float32)
        sf.write(str(path), audio[-int(seconds * SAMPLE_RATE):], SAMPLE_RATE)

        return path

    def close(self):
        self.stream.stop()
        self.stream.close()


def show_perception(obs):
    scene = ", ".join(f"{s['label']} ({s['score']:.0%})" for s in obs["scene"])
    if "objects" in obs:
        objects = ", ".join(f"{label} {n}" for label, n in sorted(obs["objects"].items(), key=lambda x: -x[1])) or "없음"
        print(f"  [사진] 탐지: {objects} | 장면: {scene}")
    else:
        print(f"  [사진] 장면: {scene} (이 단계엔 물체 탐지기가 없어요)")

    speech = obs["speech"]
    said = f"말: \"{speech['text']}\" ({speech['language']})" if speech["text"] else "말: (말소리 없음)"
    kinds = ""
    if obs.get("sounds"):
        kinds = " | 종류: " + ", ".join(f"{t['label']} ({t['score']:.0%})" for t in obs["sounds"][:3])
    print(f"  [소리] {said}{kinds}  (크기 {speech.get('level', 0):.3f})")


def open_devices(args):
    camera = mic = None
    if not args.no_camera:
        try:
            camera = Camera(args.camera)
            print(f"카메라 {args.camera}번 켬")
        except Exception as e:
            print(f"카메라를 못 켰어요: {e}\n  -> 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 이 터미널을 허용해 주세요. 지금은 {args.image}를 씀")
    if not args.no_mic:
        try:
            device = int(args.mic) if args.mic and args.mic.isdigit() else args.mic
            mic = Microphone(device)
            print(f"마이크 켬: {mic.name}")
        except Exception as e:
            print(f"마이크를 못 켰어요: {e}\n  -> 시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 이 터미널을 허용해 주세요. 지금은 {args.audio}를 씀")

    return camera, mic


def load_session(tier, graft):
    print(f"TIER {tier} 모델 올리는 중{' (이식 켬)' if graft else ''}...")
    start = time.perf_counter()
    session = pipeline.Session(tier, graft_llm=graft, graft_image=graft, keep_loaded=True)
    print(f"준비 완료 ({time.perf_counter() - start:.1f}초, 메모리 {memory.current() / memory.MB:.0f} MB)")

    return session


def main():
    parser = argparse.ArgumentParser(description="thirtydays 실시간 채팅")
    parser.add_argument("--tier", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--graft", action="store_true")
    parser.add_argument("--listen", type=float, default=5.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mic", default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-mic", action="store_true")
    parser.add_argument("--image", default=str(ROOT / "samples" / "bus.jpg"))
    parser.add_argument("--audio", default=str(ROOT / "samples" / "speech_en.wav"))
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  {i}: {d['name']}")
        return 0

    camera, mic = open_devices(args)
    tier, graft, listen = args.tier, args.graft, args.listen
    session = load_session(tier, graft)
    history = []
    work = Path(tempfile.mkdtemp(prefix="thirtydays-"))
    last = {}
    print("\n질문을 입력하세요. 엔터만 치면 지금 보고 듣는 것만 보여줘요. /help 로 명령 보기")

    try:
        while True:
            try:
                text = input("\n나 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if text.startswith("/"):
                command, *rest = text.split()
                if command == "/quit":
                    break
                elif command == "/help":
                    print(HELP)
                elif command == "/tier" and rest and rest[0] in "1234":
                    session.close()
                    tier = int(rest[0])
                    session = load_session(tier, graft)
                elif command == "/graft" and rest and rest[0] in ("on", "off"):
                    session.close()
                    graft = rest[0] == "on"
                    session = load_session(tier, graft)
                elif command == "/listen" and rest:
                    listen = float(rest[0])
                    print(f"직전 {listen:g}초의 소리를 씀")
                elif command == "/reset":
                    history.clear()
                    print("대화 기록을 지웠어요")
                elif command == "/save" and last:
                    SAVED.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    for kind, path in last.items():
                        shutil.copy(path, SAVED / f"{stamp}{Path(path).suffix}")
                    print(f"저장: samples/live/{stamp}.*")
                else:
                    print(HELP)
                continue

            image = camera.snapshot(work / "frame.jpg") if camera else args.image
            audio = mic.snapshot(work / "sound.wav", listen) if mic else args.audio
            last = {"image": image, "audio": audio}

            start = time.perf_counter()
            result = session.ask(image, audio, text, history=history, with_llm=bool(text))
            elapsed = time.perf_counter() - start

            if text:
                answer = result["observations"]["answer"]
                print(f"Qwen > {answer}")
                history.append((text, answer))
            show_perception(result["observations"])
            print(f"  ({elapsed:.1f}초, 메모리 {memory.current() / memory.MB:.0f} MB, TIER {tier}{', 이식 켬' if graft else ''})")
    finally:
        for device in (camera, mic):
            if device:
                device.close()
        session.close()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
