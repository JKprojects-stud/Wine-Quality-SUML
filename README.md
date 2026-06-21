# Aplikacja SUML - Predyktor Jakości Wina

Aplikacja ML służąca do klasyfikacji jakości wina na podstawie jego parametrów smakowych i chemicznych. Projekt wykorzystuje modele AutoML oparte o bibliotekę AutoGluon, backend FastAPI oraz frontend Streamlit.

Aplikacja klasyfikuje próbkę wina do jednej z trzech klas:

* Niska jakość
* Przeciętna jakość
* Wysoka jakość

Projekt został przygotowany z podziałem logiki działania na warstwy:

* `data` — przygotowanie i przetwarzanie danych,
* `model` — trenowanie, ładowanie i wykorzystywanie modelu ML,
* `app` — serwowanie aplikacji przez API oraz interfejs webowy.

## Wykorzystane technologie

* Python
* AutoGluon
* FastAPI
* Streamlit
* Pandas
* XGBoost
* Docker
* Docker Compose
* Pylint

## Zbiór danych

Projekt wykorzystuje zbiór danych Wine Quality Dataset pochodzący z UCI (https://archive.ics.uci.edu/dataset/186/wine+quality), zawierający parametry fizykochemiczne próbek wina oraz ocenę ich jakości.

W projekcie wykorzystywany jest plik:

```text
data/raw/winequality-red.csv
```

Oryginalna kolumna `quality` jest przekształcana na trzy klasy jakości:

```text
quality <= 5  -> Niska jakość
quality == 6  -> Przeciętna jakość
quality >= 7  -> Wysoka jakość
```

Dane po przetworzeniu są zapisywane w pliku:

```text
data/processed/winequality_processed.csv
```

## Struktura projektu

```text
Wine-Quality-SUML/
│
├── README.md
├── requirements.txt
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── winequality-red.csv
│   └── processed/
│       └── winequality_processed.csv
│
├── models/
│   └── wine_quality_predictor/
│
└── src/
    ├── __init__.py
    ├── config.py
    │
    ├── data/
    │   ├── __init__.py
    │   └── prepare_data.py
    │
    ├── model/
    │   ├── __init__.py
    │   ├── train.py
    │   ├── predict.py
    │   └── info.py
    │
    └── app/
        ├── __init__.py
        ├── schemas.py
        ├── backend.py
        └── frontend.py
```

## Opis działania aplikacji

Aplikacja składa się z dwóch części:

1. Backend FastAPI:

   * ładuje (lub trenuje jeśli nie załadowano) wytrenowany model AutoGluon,
   * udostępnia endpoint do predykcji jakości wina,
   * udostępnia endpoint do dodania nowej próbki treningowej i ponownego trenowania modelu,
   * udostępnia informacje o danych i modelu.

2. Frontend Streamlit:

   * zawiera formularz do wpisania parametrów próbki wina,
   * prezentuje wynik predykcji,
   * umożliwia dodanie nowej próbki i przetrenowanie modelu,
   * pokazuje dane treningowe,
   * pokazuje rozkład klas jakości,
   * prezentuje informacje o modelach wytrenowanych przez AutoGluon.

## Uruchomienie lokalne

### 1. Utworzenie środowiska wirtualnego

```
python -m venv .venv
```

Aktywacja środowiska w Windows PowerShell:

```
.venv\Scripts\activate
```

### 2. Instalacja zależności

```
pip install -r requirements.txt
```

### 3. Przygotowanie danych

```
python -m src.data.prepare_data
```

### 4. Trenowanie modelu

```
python -m src.model.train
```

Model zostanie zapisany w katalogu:

```
models/wine_quality_predictor/
```

Jeżeli model istnieje, backend ładuje go przy starcie aplikacji. Jeżeli model nie istnieje, aplikacja może wytrenować go automatycznie.

### 5. Uruchomienie backendu

```
uvicorn src.app.backend:app --reload --port 8008
```

Backend będzie dostępny pod adresem:

```
http://localhost:8008
```

Dokumentacja API FastAPI będzie dostępna pod adresem:

```
http://localhost:8008/docs
```

### 6. Uruchomienie frontendu

W drugim terminalu należy uruchomić:

```
streamlit run src/app/frontend.py
```

Frontend będzie dostępny pod adresem:

```
http://localhost:8501
```

## Uruchomienie za pomocą Dockera

Projekt zawiera pliki Dockerfile dla backendu i frontendu oraz plik `docker-compose.yml`.

Aby uruchomić całą aplikację, należy wykonać w katalogu głównym projektu:

```
docker compose up --build
```

Po uruchomieniu:

```
Frontend Streamlit: http://localhost:8501
Backend FastAPI:    http://localhost:8008
Dokumentacja API:   http://localhost:8008/docs
```

Zatrzymanie aplikacji:

```
docker compose down
```

## Najważniejsze endpointy API

### Sprawdzenie działania backendu

```http
GET /
```

### Sprawdzenie statusu aplikacji

```http
GET /health
```

### Predykcja jakości wina

```http
POST /model/predict
```

### Dodanie próbki i ponowne trenowanie modelu

```http
POST /model/train
```

### Informacje o modelu

```http
GET /model/info
```

### Dane treningowe

```http
GET /data/records
```

### Statystyki danych

```http
GET /data/statistics
```

## Model Machine Learning

Do trenowania modelu wykorzystano bibliotekę AutoGluon. Aplikacja porównuje kilka modeli klasyfikacyjnych, w tym między innymi:

* Random Forest,
* Extra Trees,
* XGBoost,
* Weighted Ensemble.

Najlepszy model jest wybierany automatycznie na podstawie wyniku walidacyjnego. W obecnej wersji projektu model osiągnął wynik walidacyjny accuracy na poziomie około 0.76.

## Ponowne trenowanie modelu

Aplikacja umożliwia dodanie nowej próbki danych wraz z rzeczywistą klasą jakości wina. Po dodaniu próbki backend dopisuje ją do pliku danych treningowych i ponownie trenuje model AutoGluon.

Model jest trenowany ponownie na powiększonym zbiorze danych.

## Jakość kodu

Kod projektu został sprawdzony w Pylint.

Uzyskany wynik:

```
8.10/10
```

## Autor

Jeremiasz Kuśmierz - s27808 - 2526L_win_SUML_13_16_17
