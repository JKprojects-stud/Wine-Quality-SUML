# FastAPI

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.app.schemas import (
    PredictionResponse,
    TrainingResponse,
    TrainingSample,
    WineSample,
)
from src.config import TARGET_COLUMN
from src.data.prepare_data import load_processed_data
from src.model.info import (
    append_training_sample,
    get_data_records,
    get_data_statistics,
    get_model_info,
)
from src.model.predict import predict_quality
from src.model.train import load_or_train_model, train_model


class ModelState:
    predictor = None


MODEL_STATE = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    MODEL_STATE.predictor = load_or_train_model()
    yield


app = FastAPI(
    title="Aplikacja do przewidywania jakości wina",
    description="Klasyfikacja jakości wina z użyciem AutoML.",
    lifespan=lifespan,
)


@app.get("/", tags=["intro"])
def index() -> dict:
    return {"message": "Przewidywanie jakości wina"}

@app.get("/health", tags=["intro"])
def health() -> dict:
    """Zwraca status aplikacji."""
    return {"status": "OK"}


@app.post(
    "/model/predict",
    tags=["model"],
    response_model=PredictionResponse,
    status_code=200,
)
def get_prediction(sample: WineSample) -> PredictionResponse:
    if MODEL_STATE.predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    predicted_quality, probabilities = predict_quality(
        MODEL_STATE.predictor,
        sample.to_model_input(),
    )

    return PredictionResponse(
        predicted_quality=predicted_quality,
        probabilities=probabilities,
    )

@app.post(
    "/model/train",
    tags=["model"],
    response_model=TrainingResponse,
    status_code=200,
)
def retrain_model(sample: TrainingSample) -> TrainingResponse:
    sample_data = sample.to_model_input()
    append_training_sample(sample_data, sample.quality_class.value)

    MODEL_STATE.predictor = train_model(time_limit=60)

    updated_data = load_processed_data()

    return TrainingResponse(
        status="Model został dotrenowany.",
        added_quality_class=sample.quality_class.value,
        rows_after_training=int(len(updated_data)),
    )

@app.get("/model/info", tags=["model"])
def model_info() -> dict:
    if MODEL_STATE.predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return get_model_info(MODEL_STATE.predictor)


@app.get("/data/records", tags=["data"])
def data_records(limit: int = 100) -> dict:
    return {
        "records": get_data_records(limit=limit),
        "target_column": TARGET_COLUMN,
    }

@app.get("/data/statistics", tags=["data"])
def data_statistics() -> dict:
    return get_data_statistics()