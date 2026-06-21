# Funkcje zwracające informacje o danych i modelu.

import pandas as pd
from autogluon.tabular import TabularPredictor

from src.config import PROCESSED_DATA_FILE, TARGET_COLUMN
from src.data.prepare_data import load_processed_data


def get_data_records(limit: int = 100) -> list[dict]:
    data_frame = load_processed_data()
    return data_frame.head(limit).to_dict(orient="records")


def get_data_statistics() -> dict:
    data_frame = load_processed_data()
    class_counts = data_frame[TARGET_COLUMN].value_counts().to_dict()

    return {
        "rows": int(len(data_frame)),
        "columns": list(data_frame.columns),
        "class_counts": class_counts,
        "data_file": str(PROCESSED_DATA_FILE),
    }


def get_model_info(predictor: TabularPredictor) -> dict:

    leaderboard = predictor.leaderboard(silent=True)

    return {
        "target_column": TARGET_COLUMN,
        "model_names": predictor.model_names(),
        "best_model": predictor.model_best,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }

# dotrenowanie

def append_training_sample(sample_data: dict, quality_class: str) -> None:

    data_frame = load_processed_data()

    new_row = sample_data.copy()
    new_row[TARGET_COLUMN] = quality_class

    updated_data = pd.concat(
        [data_frame, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    updated_data.to_csv(PROCESSED_DATA_FILE, index=False)