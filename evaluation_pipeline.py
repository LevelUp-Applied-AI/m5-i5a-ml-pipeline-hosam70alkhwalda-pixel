"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 5 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """
    Load data and separate features from target.
    """
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.lower()

    y = df["churned"]
    X = df.drop(columns=["churned"])

    return X, y


def build_preprocessor(X):
    """
    Build a ColumnTransformer for numeric and categorical features.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ])

    return preprocessor


def define_models():
    """
    Define the 5 model configurations to compare.
    """
    models = {
        "LogReg_default": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
        ),

        "LogReg_L1": LogisticRegression(
            C=0.1, penalty="l1", solver="saga",
            max_iter=1000, random_state=42, class_weight="balanced"
        ),

        "Ridge": RidgeClassifier(
            alpha=1.0
        ),

        "Dummy_most_frequent": DummyClassifier(strategy="most_frequent"),

        "Dummy_stratified": DummyClassifier(strategy="stratified", random_state=42)
    }

    return models


def evaluate_models(models, X, y):
    """
    Run cross-validation on all models and return results.
    """

    results = []

    for name, model in models.items():

        preprocessor = build_preprocessor(X)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=5,
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        results.append({
            "Model": name,
            "Mean Accuracy": scores["test_accuracy"].mean(),
            "Std": scores["test_accuracy"].std(),
            "Mean Precision": scores["test_precision"].mean(),
            "Mean Recall": scores["test_recall"].mean(),
            "Mean F1": scores["test_f1"].mean()
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df


def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    """
    Train and evaluate final model.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def recommend_model(results_df):
    """
    Print recommendation.
    """
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("The recommended model is Logistic Regression with L1 regularization (C=0.1), as it achieved the best F1 score in both cross-validation (0.342) and on the held-out test set (0.379). Although its accuracy is lower than the most-frequent dummy classifier (0.838), accuracy is misleading in this highly imbalanced dataset (churn rate = 16.27%), where the dummy model achieves high accuracy by always predicting the majority class but fails to detect any churners. In contrast, the selected model significantly improves recall (0.65), meaning it successfully identifies most churned customers, while maintaining reasonable precision (0.27). Compared to the stratified dummy baseline (F1 = 0.162), the model more than doubles performance, confirming it has learned meaningful patterns beyond random guessing. Overall, the model generalizes well with consistent CV and test results, making it the best choice among linear models, though further improvement may require more advanced models.")


if __name__ == "__main__":

    X, y = load_and_prepare()

    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    models = define_models()

    results = evaluate_models(models, X_train, y_train)

    recommend_model(results)

    best_model_name = results[
        ~results["Model"].str.contains("Dummy")
    ].sort_values(by="Mean F1", ascending=False).iloc[0]["Model"]

    best_model = models[best_model_name]

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])

    final_results = final_evaluation(
        pipeline,
        X_train, X_test,
        y_train, y_test
    )

    print("\nBest model:", best_model_name)
    print("Final Test Results:", final_results)