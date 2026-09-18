from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"


st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)


st.title(
    "🎓 AI-Based Student Performance & Academic Risk Analyzer"
)


st.write(
    "Enter student information to generate a Machine Learning "
    "prediction of academic performance."
)


if not (MODEL_DIR / "regression_model.joblib").exists():

    st.error(
        "Models not found. Please train the models first."
    )

    st.stop()


reg = joblib.load(
    MODEL_DIR / "regression_model.joblib"
)


clf = joblib.load(
    MODEL_DIR / "classification_model.joblib"
)


st.header("Student Information")


age = st.slider(
    "Age",
    15,
    22,
    18
)


studytime = st.slider(
    "Study Time",
    1,
    4,
    2
)


failures = st.slider(
    "Previous Failures",
    0,
    4,
    0
)


absences = st.slider(
    "Absences",
    0,
    90,
    4
)


if st.button("Predict"):

    sample = pd.DataFrame([{

        "school": "GP",

        "sex": "M",

        "age": age,

        "address": "U",

        "famsize": "GT3",

        "Pstatus": "T",

        "Medu": 2,

        "Fedu": 2,

        "Mjob": "other",

        "Fjob": "other",

        "reason": "course",

        "guardian": "mother",

        "traveltime": 1,

        "studytime": studytime,

        "failures": failures,

        "schoolsup": "no",

        "famsup": "yes",

        "paid": "no",

        "activities": "yes",

        "nursery": "yes",

        "higher": "yes",

        "internet": "yes",

        "romantic": "no",

        "famrel": 4,

        "freetime": 3,

        "goout": 3,

        "Dalc": 1,

        "Walc": 1,

        "health": 4,

        "absences": absences

    }])


    grade = reg.predict(sample)[0]


    risk = clf.predict(sample)[0]


    st.success(
        f"Predicted Grade: {grade:.2f} / 20"
    )


    st.info(
        f"Academic Risk Category: {risk}"
    )
