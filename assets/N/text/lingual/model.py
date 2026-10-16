"""LingualSense GRU 언어 식별 모델의 실제 구조.

공식 GRU_Model.keras 안의 config.json과 동일한 레이어를 코드로 풀어 쓴 파일이다.
"""

from pathlib import Path

from tensorflow import keras


VOCAB_SIZE = 6635
MAX_LENGTH = 200
NUM_LANGUAGES = 28


def build_model():
    """공식 가중치와 같은 GRU 네트워크 구조를 생성한다."""
    return keras.Sequential(
        [
            keras.layers.Input(shape=(MAX_LENGTH,), name="input_tokens"),
            keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=128,
                mask_zero=True,
                name="embedding",
            ),
            keras.layers.GRU(128, name="gru"),
            keras.layers.Dropout(0.5, name="dropout"),
            keras.layers.Dense(64, activation="relu", name="hidden"),
            keras.layers.Dense(
                NUM_LANGUAGES,
                activation="softmax",
                name="language_probabilities",
            ),
        ],
        name="LingualSense_GRU",
    )


def load_trained_model(path=None):
    """저장 파일의 버전별 config 차이를 피하고 실제 가중치만 구조에 주입한다."""
    model = build_model()
    model.load_weights(path or Path(__file__).parent / "models" / "GRU_Model.keras")
    return model


if __name__ == "__main__":
    load_trained_model().summary()
