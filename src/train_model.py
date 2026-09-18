from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import prepare_data


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


X, y_reg, y_cls, preprocessor = prepare_data()


X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(

    X,
    y_reg,
    y_cls,

    test_size=0.2,

    random_state=42,

    stratify=y_cls
)


reg = Pipeline([

    ("preprocessor", preprocessor),

    ("model", RandomForestRegressor(

        n_estimators=300,

        random_state=42

    ))

])


clf = Pipeline([

    ("preprocessor", preprocessor),

    ("model", RandomForestClassifier(

        n_estimators=300,

        random_state=42

    ))

])


print("Training Regression Model...")

reg.fit(X_train, y_reg_train)


print("Training Classification Model...")

clf.fit(X_train, y_cls_train)


joblib.dump(

    reg,

    MODEL_DIR / "regression_model.joblib"

)


joblib.dump(

    clf,

    MODEL_DIR / "classification_model.joblib"

)


joblib.dump({

    "X_test": X_test,

    "y_reg": y_reg_test,

    "y_cls": y_cls_test

}, MODEL_DIR / "test_split.joblib")


print("Training Completed.")
