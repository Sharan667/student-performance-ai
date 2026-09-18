from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "student-mat.csv"


def load_data():

    return pd.read_csv(DATA_PATH, sep=";")


def risk_label(g):

    if g < 10:
        return "At Risk"

    elif g < 14:
        return "Moderate"

    elif g < 17:
        return "Good"

    return "High"


def prepare_data(include_previous_grades=False):

    df = load_data()

    drop = ["G3"]

    if not include_previous_grades:
        drop += ["G1", "G2"]

    X = df.drop(columns=drop)

    y_reg = df["G3"]

    y_cls = df["G3"].apply(risk_label)

    numeric = X.select_dtypes(include="number").columns

    categorical = X.select_dtypes(exclude="number").columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric),
        ("cat", cat_pipeline, categorical)
    ])

    return X, y_reg, y_cls, preprocessor
