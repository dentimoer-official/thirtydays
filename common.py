"""진입점 공통 도우미.

조각 사이에 여러 값을 넘길 때는 dict로 묶고, 대표 텐서는 "x" 키에 둔다.
"""

import importlib.util


def load_entry(name, path):
    spec = importlib.util.spec_from_file_location(f"entry_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "load"):
        module.load()

    return module


def is_tensor(value):
    return hasattr(value, "shape") and hasattr(value, "dtype")


def nbytes(value):
    if hasattr(value, "element_size") and hasattr(value, "numel"):
        return value.element_size() * value.numel()
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if hasattr(value, "numpy") and is_tensor(value):
        # TF 텐서와 변수
        return int(value.numpy().nbytes)
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        sizes = [nbytes(v) for v in value]
        return sum(s for s in sizes if s is not None) if any(s is not None for s in sizes) else None

    return None


def describe(value):
    main = value["x"] if isinstance(value, dict) and "x" in value else value

    if isinstance(main, (list, tuple)) and main and all(is_tensor(v) for v in main):
        shape = [list(v.shape) for v in main]
        dtype = str(main[0].dtype)
    else:
        shape = list(main.shape) if is_tensor(main) else None
        dtype = str(main.dtype) if is_tensor(main) else None

    return {
        "type": type(value).__name__,
        "shape": shape,
        "dtype": dtype,
        "bytes": nbytes(value),
    }


def to_json(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]

    return value


def part_info(entry):
    # 진입점이 part_modules()를 주면 조각별 파라미터 수를 셈
    modules = entry.part_modules() if hasattr(entry, "part_modules") else {}
    info = []

    for part in entry.parts():
        params = {id(p): p for m in modules.get(part, []) for p in parameters(m)}.values()
        info.append({
            "part": part,
            "params": sum(count(p) for p in params),
            "bytes": sum(nbytes(p) or 0 for p in params),
        })

    return info


def parameters(module):
    # torch 모듈, 단일 파라미터 텐서, keras 레이어 지원
    if is_tensor(module):
        return [module]
    if hasattr(module, "parameters"):
        return list(module.parameters())
    if hasattr(module, "weights"):
        return list(module.weights)

    return []


def count(param):
    if hasattr(param, "numel"):
        return param.numel()

    size = 1
    for d in param.shape:
        size *= int(d)

    return size


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if hasattr(a, "equal") and hasattr(b, "equal"):
        return bool(a.shape == b.shape and a.equal(b))
    if is_tensor(a) and is_tensor(b):
        import numpy as np
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))

    return a == b


def max_diff(a, b):
    # 텐서끼리 최대 절대 오차. 비교할 텐서가 없으면 None
    import numpy as np

    if isinstance(a, dict) and isinstance(b, dict):
        diffs = [max_diff(a[k], b[k]) for k in a if k in b]
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        diffs = [max_diff(x, y) for x, y in zip(a, b)]
    elif is_tensor(a) and is_tensor(b):
        a = a.detach().float().numpy() if hasattr(a, "detach") else np.asarray(a, dtype=np.float64)
        b = b.detach().float().numpy() if hasattr(b, "detach") else np.asarray(b, dtype=np.float64)
        return float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else None
    else:
        return None

    diffs = [d for d in diffs if d is not None]

    return max(diffs) if diffs else None
