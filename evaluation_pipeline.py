"""
Module 5 Week A — Integration: ML Evaluation Pipeline
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
import warnings

warnings.filterwarnings("ignore")



# LOAD DATA

def load_and_prepare(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.lower()

    # drop ID column (IMPORTANT FIX)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    y = df["churned"]
    X = df.drop(columns=["churned"])

    return X, y



# PREPROCESSOR

def build_preprocessor():
    numeric_features = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "num_support_calls",
        "senior_citizen",
        "has_partner",
        "has_dependents"
    ]

    categorical_features = [
        "gender",
        "contract_type",
        "internet_service",
        "payment_method"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ])

    return preprocessor


# MODELS (PIPELINES)

def define_models():
    preprocessor = build_preprocessor()

    models = {
        "LogReg_default": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=1.0, max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "LogReg_L1": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=0.1, penalty="l1", solver="saga",
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "Ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RidgeClassifier(alpha=1.0))
        ]),

        "Dummy_most_frequent": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "Dummy_stratified": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="stratified", random_state=42))
        ])
    }

    return models


# CROSS VALIDATION

def evaluate_models(models, X, y, cv=5):
    results = []

    for name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        results.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean()
        })

    return pd.DataFrame(results)


# FINAL EVALUATION

def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


# RECOMMENDATION 

def recommend_model(results_df):
    print("\n=== CV Results ===")
    print(results_df)

    print("\n=== Recommendation ===")
    print("Select model with highest F1 score among non-dummy models.")


# MAIN

if __name__ == "__main__":

    X, y = load_and_prepare()
    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    models = define_models()

    results_df = evaluate_models(models, X_train, y_train)
    print(results_df)

    recommend_model(results_df)

    best_model_name = results_df[
        ~results_df["model"].str.contains("Dummy")
    ].sort_values(by="f1_mean", ascending=False).iloc[0]["model"]

    print("Best model:", best_model_name)

    # FINAL EVALUATION
    best_pipeline = models[best_model_name]

    final_results = final_evaluation(
        best_pipeline,
        X_train, X_test,
        y_train, y_test
    )

    print("\n=== Final Test Results ===")
    print(final_results)