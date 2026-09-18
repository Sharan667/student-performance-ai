# AI-Based Student Performance & Academic Risk Analyzer

Overview

The AI-Based Student Performance & Academic Risk Analyzer is a Machine Learning project that analyzes student-related academic, demographic, social, and school information to estimate final academic performance and classify students into project-defined academic-risk categories.

The project demonstrates data preprocessing, supervised learning, regression, classification, model evaluation, and an interactive Streamlit interface.

Objectives

Load and preprocess student performance data.

Handle numerical and categorical features.

Predict the final grade (G3) using regression.

Classify students into project-defined risk categories.

Evaluate model performance using standard metrics.

Provide an interactive prediction dashboard.

Document a reproducible project workflow.

ML Tasks

Regression

Target: G3 — final grade on a 0–20 scale.

Metrics:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Classification

Project-defined categories:

Final Grade

Category

0–9

At Risk

10–13

Moderate

14–16

Good

17–20

High

Metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Dataset

UCI Machine Learning Repository — Student Performance Dataset.

The project uses the Mathematics-course file student-mat.csv.

Dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

Technologies

Python

Pandas

NumPy

Scikit-learn

Streamlit

Matplotlib

Joblib

Project Structure

student-performance-ai/
├── app/
│   └── app.py
├── data/
├── docs/
│   └── problem_statement.md
├── models/
├── src/
│   ├── __init__.py
│   ├── download_dataset.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── tests/
│   └── test_preprocessing.py
├── README.md
├── requirements.txt
└── .gitignore

Installation

git clone https://github.com/YOUR_USERNAME/student-performance-ai.git
cd student-performance-ai
python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Download Dataset

python src/download_dataset.py

Train Models

python src/train_model.py

The training script creates:

regression_model.joblib

classification_model.joblib

test_split.joblib

inside the models/ directory.

Evaluate Models

python src/evaluate_model.py

The evaluation script reports MAE, RMSE, R², accuracy, precision, recall, F1-score, and the confusion matrix.

Run Dashboard

streamlit run app/app.py

Methodology

UCI Dataset
     ↓
Data Loading
     ↓
Preprocessing
     ↓
Numerical Scaling + Categorical Encoding
     ↓
Train/Test Split
     ↓
Random Forest Regression + Classification
     ↓
Model Evaluation
     ↓
Streamlit Prediction Dashboard

Important Modeling Decision

The initial model excludes G1 and G2 from the input features because they are earlier-period grades and are closely related to the final grade G3. This reduces direct leakage from earlier grades into the final-grade prediction and makes the experiment more meaningful for exploring other student attributes.

Limitations

The dataset represents students from two Portuguese secondary schools and should not be treated as a universal representation of students everywhere.

The risk categories are project-defined labels and are not official educational diagnoses.

Predictions are model estimates and should not be used as the sole basis for educational decisions.

Model performance depends on the dataset and preprocessing choices.

Future Enhancements

Compare Logistic Regression, Decision Tree, Random Forest, KNN, and SVM.

Add exploratory data analysis visualizations.

Add feature-importance and explainability views.

Add ROC curves and richer evaluation plots.

Improve the Streamlit dashboard.

Provide an API using FastAPI.

Add cloud deployment.

Author

Sharan667

B.Tech Computer Science and Engineering

Academic Use

This project is developed for educational and academic evaluation purposes.
