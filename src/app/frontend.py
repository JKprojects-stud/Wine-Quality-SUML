# Streamlit

import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8008")


def post_prediction(sample_data: dict) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/model/predict",
        json=sample_data,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def post_training_sample(sample_data: dict) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/model/train",
        json=sample_data,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def get_data_records() -> pd.DataFrame:
    response = requests.get(f"{BACKEND_URL}/data/records?limit=200", timeout=120)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data["records"])


def get_data_statistics() -> dict:
    response = requests.get(f"{BACKEND_URL}/data/statistics", timeout=120)
    response.raise_for_status()
    return response.json()


def get_model_info() -> dict:
    response = requests.get(f"{BACKEND_URL}/model/info", timeout=120)
    response.raise_for_status()
    return response.json()


def wine_form(form_name: str) -> dict:
    st.subheader(form_name)

    fixed_acidity = st.number_input("Kwasowość stała", value=7.4, min_value=0.0)
    volatile_acidity = st.number_input("Kwasowość lotna", value=0.70, min_value=0.0)
    citric_acid = st.number_input("Kwas cytrynowy", value=0.00, min_value=0.0)
    residual_sugar = st.number_input("Cukier resztkowy", value=1.9, min_value=0.0)
    chlorides = st.number_input("Chlorki", value=0.076, min_value=0.0)

    free_sulfur_dioxide = st.number_input(
        "Wolny dwutlenek siarki",
        value=11.0,
        min_value=0.0,
    )
    total_sulfur_dioxide = st.number_input(
        "Całkowity dwutlenek siarki",
        value=34.0,
        min_value=0.0,
    )

    density = st.number_input("Gęstość", value=0.9978, min_value=0.0, format="%.4f")
    ph_value = st.number_input("pH", value=3.51, min_value=0.0)
    sulphates = st.number_input("Siarczany", value=0.56, min_value=0.0)
    alcohol = st.number_input("Alkohol", value=9.4, min_value=0.0)

    return {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "ph_value": ph_value,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }


st.set_page_config(
    page_title="Predyktor jakości wina - SUML",
    page_icon="🍷",
    layout="wide",
)

st.title("🍷 Predyktor jakości wina")
st.write(
    "Aplikacja wykorzystuje model AutoGluon do klasyfikacji jakości wina "
    "na podstawie jego parametrów samakowych i chemicznych."
)

tab_prediction, tab_training, tab_data, tab_model = st.tabs(
    [
        "Predykcja",
        "Dodaj dane i przetrenuj",
        "Dane",
        "Model",
    ]
)

with tab_prediction:
    with st.form("prediction_form"):
        prediction_sample = wine_form("Parametry próbki wina")
        prediction_button = st.form_submit_button("Przewidź jakość")

    if prediction_button:
        try:
            prediction_result = post_prediction(prediction_sample)

            st.success(
                f"Przewidywana klasa: "
                f"**{prediction_result['predicted_quality']}**"
            )

            probabilities = pd.DataFrame(
                prediction_result["probabilities"].items(),
                columns=["Klasa", "Prawdopodobieństwo"],
            )
            st.dataframe(probabilities, use_container_width=True)

        except requests.exceptions.RequestException as error:
            st.error(f"Nie udało się wykonać predykcji: {error}")

with tab_training:
    with st.form("training_form"):
        training_sample = wine_form("Nowa próbka treningowa")

        quality_class = st.selectbox(
            "Rzeczywista klasa jakości tej próbki",
            [
                "Niska jakość",
                "Przeciętna jakość",
                "Wysoka jakość",
            ],
        )

        training_button = st.form_submit_button("Dodaj próbkę i przetrenuj model")

    if training_button:
        try:
            training_payload = training_sample.copy()
            training_payload["quality_class"] = quality_class

            training_result = post_training_sample(training_payload)

            st.success(training_result["status"])
            st.write(
                f"Liczba rekordów po przetrenowaniu: "
                f"**{training_result['rows_after_training']}**"
            )
            st.toast("Model został przetrenowany 🎉", icon="✅")
            st.balloons()

        except requests.exceptions.RequestException as error:
            st.error(f"Nie udało się przetrenować modelu: {error}")

with tab_data:
    st.subheader("Dane treningowe")

    try:
        statistics = get_data_statistics()
        records = get_data_records()

        st.write(f"Liczba rekordów: **{statistics['rows']}**")
        st.dataframe(records, use_container_width=True)

        if not records.empty:
            st.subheader("Rozkład klas jakości wina")

            class_counts = records["quality_class"].value_counts()
            st.bar_chart(class_counts)

    except requests.exceptions.RequestException as error:
        st.error(f"Nie udało się pobrać danych: {error}")

with tab_model:
    st.subheader("Informacje o modelu")

    try:
        model_info = get_model_info()

        st.write(f"Kolumna docelowa: `{model_info['target_column']}`")
        st.write(f"Najlepszy model: `{model_info['best_model']}`")

        st.write("Modele wytrenowane przez AutoGluon:")
        st.write(model_info["model_names"])

        leaderboard = pd.DataFrame(model_info["leaderboard"])
        st.subheader("Leaderboard")
        st.dataframe(leaderboard, use_container_width=True)

    except requests.exceptions.RequestException as error:
        st.error(f"Nie udało się pobrać informacji o modelu: {error}")