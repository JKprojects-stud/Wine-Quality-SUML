import pandas as pd
from autogluon.tabular import TabularPredictor

from src.config import FEATURE_COLUMNS


def predict_quality(
    predictor: TabularPredictor,
    sample_data: dict,
) -> tuple[str, dict[str, float]]:
    input_frame = pd.DataFrame([sample_data], columns=FEATURE_COLUMNS)

    prediction = predictor.predict(input_frame).iloc[0]
    probabilities = predictor.predict_proba(input_frame).iloc[0].to_dict()

    formatted_probabilities = {
        str(label): float(probability)
        for label, probability in probabilities.items()
    }

    return str(prediction), formatted_probabilities