"""fastText lid.176 진입점: tokenize -> lookup -> average -> classify(계층적 소프트맥스)."""

from pathlib import Path

import fasttext
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "lid.176.bin"
EOS = "</s>"
TOP_K = 5

_model = None
_input = None    # [단어 + 서브워드 버킷, 16]
_output = None   # [라벨 수, 16] (hs에서는 트리 내부 노드 가중치)
_labels = None
_tree = None


def load():
    global _model, _input, _output, _labels, _tree
    _model = fasttext.load_model(str(WEIGHTS))
    _input = _model.get_input_matrix()
    _output = _model.get_output_matrix()
    labels, counts = _model.get_labels(include_freq=True)
    _labels = [label.removeprefix("__label__") for label in labels]
    _tree = build_tree(counts)


def build_tree(counts):
    # fastText loss.cc의 HierarchicalSoftmaxLoss::buildTree와 같은 허프만 트리
    osz = len(counts)
    tree = [{"left": -1, "right": -1, "count": 1e15} for _ in range(2 * osz - 1)]
    for i in range(osz):
        tree[i]["count"] = counts[i]

    leaf, node = osz - 1, osz
    for i in range(osz, 2 * osz - 1):
        mini = []
        for _ in range(2):
            if leaf >= 0 and tree[leaf]["count"] < tree[node]["count"]:
                mini.append(leaf)
                leaf -= 1
            else:
                mini.append(node)
                node += 1
        tree[i]["left"], tree[i]["right"] = mini
        tree[i]["count"] = tree[mini[0]]["count"] + tree[mini[1]]["count"]

    return tree


def parts():
    return ["tokenize", "lookup", "average", "classify"]


def part_modules():
    return {"tokenize": [], "lookup": [_input], "average": [], "classify": [_output]}


def tokenize(text):
    # 공백 단위 토큰. get_line이 줄 끝 토큰(</s>)까지 붙여 줌 (predict가 붙이는 "\n"과 같은 역할)
    words, _ = _model.get_line(text)

    return words if words and words[-1] == EOS else [*words, EOS]


def lookup(tokens):
    # 토큰마다 단어 id와 문자 n-gram 버킷 id -> 입력 행렬의 행들 [n, 16]
    ids = [i for token in tokens for i in _model.get_subwords(token)[1]]

    return _input[ids]


def average(rows):
    return rows.mean(axis=0, dtype=np.float32)


def std_log(x):
    return np.log(x + 1e-5)


def classify(hidden):
    # 루트에서 잎까지 내려가며 log(sigmoid) 누적. 잎 = 라벨
    osz = len(_labels)
    floor = std_log(0.0)
    scores = []

    def dfs(node, score):
        # 원본과 같은 가지치기: 누적 점수가 log(1e-5)보다 낮으면 버림
        if score < floor:
            return
        if _tree[node]["left"] == -1:
            scores.append((score, node))
            return
        f = 1.0 / (1.0 + np.exp(-float(_output[node - osz] @ hidden)))
        dfs(_tree[node]["left"], score + std_log(1.0 - f))
        dfs(_tree[node]["right"], score + std_log(f))

    dfs(2 * osz - 2, 0.0)
    top = sorted(scores, reverse=True)[:TOP_K]

    return [{"label": _labels[node], "score": round(float(np.exp(score)), 4)} for score, node in top]


def run_part(name, x):
    if name == "tokenize":
        return tokenize(x)
    if name == "lookup":
        return lookup(x)
    if name == "average":
        return average(x)
    if name == "classify":
        return classify(x)

    raise KeyError(f"unknown part: {name}")


def whole(text):
    # 비교 기준: fastText 공식 predict
    labels, probs = _model.predict(text, k=TOP_K)

    return [{"label": label.removeprefix("__label__"), "score": round(float(p), 4)} for label, p in zip(labels, probs)]
