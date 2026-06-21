# Modele danych dla FastAPI.

from enum import Enum

from pydantic import BaseModel, Field

from src.config import FEATURE_COLUMNS


class QualityClass(str, Enum):

    LOW = "Niska jakość"
    MEDIUM = "Przeciętna jakość"
    HIGH = "Wysoka jakość"

# Parametry wina

class WineSample(BaseModel):

    fixed_acidity: float = Field(..., ge=0)
    volatile_acidity: float = Field(..., ge=0)
    citric_acid: float = Field(..., ge=0)
    residual_sugar: float = Field(..., ge=0)
    chlorides: float = Field(..., ge=0)
    free_sulfur_dioxide: float = Field(..., ge=0)
    total_sulfur_dioxide: float = Field(..., ge=0)
    density: float = Field(..., ge=0)
    ph_value: float = Field(..., ge=0)
    sulphates: float = Field(..., ge=0)
    alcohol: float = Field(..., ge=0)

    def to_model_input(self) -> dict:
        return {
            "fixed acidity": self.fixed_acidity,
            "volatile acidity": self.volatile_acidity,
            "citric acid": self.citric_acid,
            "residual sugar": self.residual_sugar,
            "chlorides": self.chlorides,
            "free sulfur dioxide": self.free_sulfur_dioxide,
            "total sulfur dioxide": self.total_sulfur_dioxide,
            "density": self.density,
            "pH": self.ph_value,
            "sulphates": self.sulphates,
            "alcohol": self.alcohol,
        }


class TrainingSample(WineSample):
    quality_class: QualityClass


class PredictionResponse(BaseModel):
    predicted_quality: str
    probabilities: dict[str, float]


class TrainingResponse(BaseModel):
    status: str
    added_quality_class: str
    rows_after_training: int


def get_empty_sample() -> dict:
    return dict.fromkeys(FEATURE_COLUMNS, 0.0)