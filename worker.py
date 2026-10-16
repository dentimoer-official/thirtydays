import gc
import itertools
import json
import sys
import time

from common import describe, load_entry, part_info, to_json

# 모델 코드의 print가 프로토콜(stdout)을 오염시키지 않도록 분리
protocol = sys.stdout
sys.stdout = sys.stderr

entries = {}    # 모델 이름 -> 진입점 모듈
tensors = {}    # handle -> 실제 값 (워커 안에만 존재)
_ids = itertools.count()


def store(value):
    handle = f"t{next(_ids)}"
    tensors[handle] = value

    return {"handle": handle, **describe(value)}


def resolve(arg):
    # {"handle": "t3"} 형태면 워커 안의 실제 값으로 바꿈
    if isinstance(arg, dict) and "handle" in arg:
        return tensors[arg["handle"]]
    if isinstance(arg, list):
        return [resolve(a) for a in arg]

    return arg


def dispatch(req):
    op = req["op"]

    if op == "load":
        entries[req["model"]] = load_entry(req["model"], req["path"])
        return None
    if op == "parts":
        return entries[req["model"]].parts()
    if op == "info":
        return part_info(entries[req["model"]])
    if op == "put":
        return store(req["value"])
    if op == "run_part":
        x = resolve(req["input"])
        start = time.perf_counter()
        out = entries[req["model"]].run_part(req["part"], x)
        elapsed = (time.perf_counter() - start) * 1000
        return {**store(out), "elapsed_ms": elapsed}
    if op == "get":
        return to_json(tensors[req["handle"]])
    if op == "free":
        tensors.pop(req["handle"], None)
        return None
    if op == "call":
        fn = getattr(entries[req["model"]], req["fn"])
        return fn(*resolve(req.get("args", [])))
    if op == "unload":
        entries.pop(req["model"], None)
        gc.collect()
        return None

    raise ValueError(f"unknown op: {op}")


for line in sys.stdin:
    if not line.strip():
        continue

    try:
        resp = {"ok": True, "result": dispatch(json.loads(line))}
    except Exception as e:
        resp = {"ok": False, "error": repr(e)}

    try:
        out = json.dumps(resp)
    except TypeError as e:
        out = json.dumps({"ok": False, "error": f"unserializable result: {e!r}"})

    protocol.write(out + "\n")
    protocol.flush()
