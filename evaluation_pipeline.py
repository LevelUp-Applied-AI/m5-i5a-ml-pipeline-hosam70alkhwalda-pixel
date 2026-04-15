
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")





NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []



# Data Loading

def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target."""
    df = pd.read_csv(filepath)

    X = df.drop(columns=['churned', 'customer_id'])
    y = df['churned']

   
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES

    NUMERIC_FEATURES = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    CATEGORICAL_FEATURES = X.select_dtypes(include=['object']).columns.tolist()

    return X, y



# Preprocessing

def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor



# Models

def define_models():
    """Define the 5 model configurations to compare."""
    preprocessor = build_preprocessor()

    models = {
        "LogReg_default": Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ]),

        "LogReg_L1": Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(
                C=0.1,
                penalty='l1',
                solver='saga',
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ]),

        "RidgeClassifier": Pipeline([
            ('preprocessor', preprocessor),
            ('model', RidgeClassifier(
                alpha=1.0,
                random_state=42,
                class_weight='balanced'
            ))
        ]),

        "Dummy_most_frequent": Pipeline([
            ('preprocessor', preprocessor),
            ('model', DummyClassifier(strategy='most_frequent'))
        ]),

        "Dummy_stratified": Pipeline([
            ('preprocessor', preprocessor),
            ('model', DummyClassifier(strategy='stratified', random_state=42))
        ])
    }

    return models



# Cross Validation

def evaluate_models(models, X, y, cv=5, random_state=42):
    """Run cross-validation on all models and return results."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    scoring = ["accuracy", "precision", "recall", "f1"]

    results = []

    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)

        results.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean()
        })

    return pd.DataFrame(results)


# Final Evaluation

def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate on test set."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }



# Recommendation

def recommend_model(results_df):
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))

    print("\n=== Recommendation ===")
    print("Write your recommendation in the PR description.")



# Main Execution

if __name__ == "__main__":

    X, y = load_and_prepare()

    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.2%}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    models = define_models()

    # Cross-validation
    results = evaluate_models(models, X_train, y_train)
    recommend_model(results)

    # Select best non-dummy model
    non_dummy = results[~results["model"].str.startswith("Dummy")]

    best_model_name = non_dummy.loc[
        non_dummy["f1_mean"].idxmax(), "model"
    ]

    best_pipeline = models[best_model_name]

    # Final evaluation
    print(f"\n=== Final Test-Set Evaluation: {best_model_name} ===")

    test_metrics = final_evaluation(
        best_pipeline,
        X_train, X_test,
        y_train, y_test
    )

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Compare CV vs Test
    cv_f1 = non_dummy.loc[
        non_dummy["model"] == best_model_name, "f1_mean"
    ].values[0]

    print(f"\nCV F1 estimate: {cv_f1:.4f} | Test F1: {test_metrics['f1']:.4f}")
    
    
    """
Recommendation:

Based on the cross-validation results, the RidgeClassifier is recommended as the best-performing model, achieving the highest F1 score (~0.34) among all non-dummy models, with a slightly better balance between precision and recall compared to Logistic Regression variants. Although accuracy is around 0.61, accuracy alone is misleading in this problem due to class imbalance, as demonstrated by the Dummy_most_frequent model achieving a high accuracy of ~0.84 while completely failing to identify churners (F1 = 0.0). This highlights that correctly identifying the minority class (churned customers) is more important than overall accuracy.

The RidgeClassifier achieves moderate recall (~0.62), meaning it successfully captures a reasonable portion of churners, but at the cost of lower precision (~0.24), indicating some false positives. This trade-off is acceptable in churn prediction, where missing a churner is typically more costly than incorrectly flagging a non-churner.

Compared to the baselines, the model significantly outperforms the Dummy_stratified classifier (F1 ~0.17), achieving roughly double its F1 score (~0.34). This shows that the model is learning meaningful patterns beyond random guessing, although the improvement is still modest, suggesting that the predictive signal in the features may be limited for linear models.

Finally, the test-set evaluation (F1 ~0.38) slightly exceeds the cross-validation estimate (~0.34), confirming that the model generalizes reasonably well and that the CV results were not overly optimistic. However, the overall performance indicates room for improvement, and more advanced models (e.g., tree-based methods) may be needed to achieve stronger predictive power.
"""