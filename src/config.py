# Konfiguracja ścieżek i nazw kolumn

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "winequality-red.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "winequality_processed.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_DIR = MODELS_DIR / "wine_quality_predictor"

TARGET_COLUMN = "quality_class"

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

QUALITY_CLASSES = [
    "Niska jakość",
    "Przeciętna jakość",
    "Wysoka jakość",
]