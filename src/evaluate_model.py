from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"


reg = joblib.load(
    MODEL_DIR / "regression_model.joblib"
)


clf = joblib.load(
    MODEL_DIR / "classification_model.joblib"
)


split = joblib.load(
    MODEL_DIR / "test_split.joblib"
)


X = split["X_test"]


pred = reg.predict(X)


print("================================")
print("REGRESSION RESULTS")
print("================================")


print(
    "MAE:",
    mean_absolute_error(
        split["y_reg"],
        pred
    )
)


print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            split["y_reg"],
            pred
        )
    )
)


print(
    "R2:",
    r2_score(
        split["y_reg"],
        pred
    )
)


cls_pred = clf.predict(X)


print("\n================================")
print("CLASSIFICATION RESULTS")
print("================================")


print(
    "Accuracy:",
    accuracy_score(
        split["y_cls"],
        cls_pred
    )
)


print("\nClassification Report:")


print(
    classification_report(
        split["y_cls"],
        cls_pred
    )
)


print("\nConfusion Matrix:")


print(
    confusion_matrix(
        split["y_cls"],
        cls_pred
    )
)
