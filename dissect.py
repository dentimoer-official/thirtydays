"""해체 쇼: 진입점의 조각을 차례로 실행하며 조각별 수치를 뽑고, 통째 실행과 비교한다.

사용법: python dissect.py <entry.py> <입력>
  입력이 JSON이면 값으로, 아니면 문자열(파일 경로 등)로 넘긴다.
"""

import json
import sys
import time

import psutil

from common import describe, load_entry, max_diff, part_info, same, to_json

MB = 1024 * 1024


def parse_input(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def mb(value):
    return "-" if value is None else f"{value / MB:.2f}"


def fmt_shape(shape):
    if shape is None:
        return "-"
    if shape and isinstance(shape[0], list):
        return f"{len(shape)} tensors"

    return "x".join(map(str, shape))


def dissect(path, raw):
    proc = psutil.Process()

    rss_before = proc.memory_info().rss
    start = time.perf_counter()
    entry = load_entry("dissect", path)
    load_ms = (time.perf_counter() - start) * 1000
    print(f"load: {load_ms:.0f} ms, rss +{mb(proc.memory_info().rss - rss_before)} MB")
    print()

    info = {i["part"]: i for i in part_info(entry)}
    header = f"{'part':<12} {'params':>11} {'w MB':>8} {'out shape':<18} {'dtype':<14} {'out MB':>7} {'ms':>8} {'rss MB':>8}"
    print(header)
    print("-" * len(header))

    x = parse_input(raw)
    total_ms = 0.0
    for part in entry.parts():
        start = time.perf_counter()
        x = entry.run_part(part, x)
        elapsed = (time.perf_counter() - start) * 1000
        total_ms += elapsed

        d = describe(x)
        dtype = (d["dtype"] or d["type"]).removeprefix("torch.")
        print(
            f"{part:<12} {info[part]['params']:>11,} {mb(info[part]['bytes']):>8} {fmt_shape(d['shape']):<18} "
            f"{dtype:<14} {mb(d['bytes']):>7} {elapsed:>8.1f} {mb(proc.memory_info().rss):>8}"
        )

    print("-" * len(header))
    total_params = sum(i["params"] for i in info.values())
    total_bytes = sum(i["bytes"] for i in info.values())
    print(f"{'total':<12} {total_params:>11,} {mb(total_bytes):>8} {'':<18} {'':<14} {'':>7} {total_ms:>8.1f}")
    print()
    print(f"pieces : {to_json(x)}")

    if hasattr(entry, "whole"):
        start = time.perf_counter()
        ref = entry.whole(parse_input(raw))
        whole_ms = (time.perf_counter() - start) * 1000
        print(f"whole  : {to_json(ref)}  ({whole_ms:.1f} ms)")
        print(f"match  : {same(x, ref)}")

        diff = max_diff(x, ref)
        if diff is not None:
            print(f"max abs diff : {diff:.3e}")


if __name__ == "__main__":
    dissect(sys.argv[1], sys.argv[2])
