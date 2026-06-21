# Autoglueon

import shutil

from autogluon.tabular import TabularPredictor

from src.config import MODEL_DIR, TARGET_COLUMN
from src.data.prepare_data import load_processed_data


def train_model(time_limit: int = 60) -> TabularPredictor:
    training_data = load_processed_data()

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        path=str(MODEL_DIR),
        problem_type="multiclass",
        eval_metric="accuracy",
    )

    predictor.fit(
        train_data=training_data,
        time_limit=time_limit,
        presets="medium_quality",
    )

    return predictor


def load_or_train_model() -> TabularPredictor:
    if not MODEL_DIR.exists():
        return train_model()
    try:
        return TabularPredictor.load(str(MODEL_DIR))
    except AssertionError:
        print(
            "Istniejący model jest niezgodny z aktualnym środowiskiem. "
            "Model zostanie wytrenowany ponownie."
        )
        return train_model()


if __name__ == "__main__":
    trained_predictor = train_model()
    print(f"Model zapisany w katalogu: {MODEL_DIR}")
    print(trained_predictor.leaderboard(silent=True))