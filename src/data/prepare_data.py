import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)


def map_quality_to_class(quality_value: int) -> str:
    if quality_value <= 5:
        return "Niska jakość"

    if quality_value == 6:
        return "Przeciętna jakość"

    return "Wysoka jakość"


def prepare_data() -> pd.DataFrame:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Brak pliku danych: {RAW_DATA_FILE}. "
            "Dodaj plik winequality-red.csv do katalogu data/raw."
        )

    data_frame = pd.read_csv(RAW_DATA_FILE, sep=";")
    data_frame[TARGET_COLUMN] = data_frame["quality"].apply(map_quality_to_class)
    data_frame = data_frame.drop(columns=["quality"])

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(PROCESSED_DATA_FILE, index=False)

    return data_frame


def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_DATA_FILE.exists():
        return prepare_data()

    return pd.read_csv(PROCESSED_DATA_FILE)


if __name__ == "__main__":
    prepared_data = prepare_data()
    print(f"Zapisano dane: {PROCESSED_DATA_FILE}")
    print(f"Liczba rekordów: {len(prepared_data)}")