import os
import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TRAIN_PATH = r"C:\Users\Rachit Mittal\OneDrive\Documents\GDSC_Project_Database\MiNDAT.csv"
TEST_PATH = r"C:\Users\Rachit Mittal\OneDrive\Documents\GDSC_Project_Database\MiNDAT_UNK.csv"
MODELS_DICT_FILE = "all_models.pkl"
SUBMISSION_PATH = "submission_CatBoost.csv"

warnings.filterwarnings("ignore", category=UserWarning)


# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------
def load_data(train_path, test_path):
    """Load training and test datasets."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------
def build_preprocessor(df, target_col="CORRUCYSTIC_DENSITY"):
    """Build preprocessing pipeline for numerical and categorical features."""
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", "passthrough")
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )
    return preprocessor


# -------------------------------------------------------------------
# Model Training
# -------------------------------------------------------------------
def train_model(X_train, y_train, preprocessor):
    """Train CatBoost model with RandomizedSearchCV."""
    param_dist = {
        "regressor__iterations": [500, 1000],
        "regressor__depth": [6, 8, 10],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__l2_leaf_reg": [1, 3, 5, 7],
        "regressor__subsample": [0.6, 0.8, 1.0]
    }

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", CatBoostRegressor(
            random_state=42,
            verbose=0,
            task_type="CPU"
        ))
    ])

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, y_train):
    """Evaluate model against test set and return metrics."""
    y_pred = model.predict(X_test)
    y_baseline = np.full_like(y_test, y_train.mean(), dtype=float)

    baseline_rmse = root_mean_squared_error(y_test, y_baseline)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return baseline_rmse, rmse, mae


# -------------------------------------------------------------------
# Model Persistence
# -------------------------------------------------------------------
def save_model_to_dict(model, model_key, dict_path):
    """Save trained model into a joblib dictionary."""
    if os.path.exists(dict_path):
        all_models = joblib.load(dict_path)
    else:
        all_models = {}

    all_models[model_key] = model
    joblib.dump(all_models, dict_path)


# -------------------------------------------------------------------
# Submission
# -------------------------------------------------------------------
def create_submission(model, df_test, submission_path):
    """Generate submission file from test set predictions."""
    X_unk = df_test.drop(columns=["CORRUCYSTIC_DENSITY"], errors="ignore")
    y_unk_pred = model.predict(X_unk)

    submission = pd.DataFrame({
        "LOCAL_IDENTIFIER": df_test["LOCAL_IDENTIFIER"],
        "CORRUCYSTIC_DENSITY": y_unk_pred
    })

    submission.to_csv(submission_path, index=False, float_format="%.6f")

    # Assertions
    assert len(submission) == len(df_test), "Row count mismatch!"
    assert submission["LOCAL_IDENTIFIER"].is_unique, "Duplicate IDs found!"
    assert list(submission.columns) == ["LOCAL_IDENTIFIER", "CORRUCYSTIC_DENSITY"], "Wrong column order!"

    return submission


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
def main():
    # Load data
    df_train, df_test = load_data(TRAIN_PATH, TEST_PATH)

    # Prepare features and labels
    df_train.dropna(subset=["CORRUCYSTIC_DENSITY"], inplace=True)
    X = df_train.drop(columns=["CORRUCYSTIC_DENSITY"])
    y = df_train["CORRUCYSTIC_DENSITY"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessor
    preprocessor = build_preprocessor(df_train)

    # Train model
    best_model, best_params = train_model(X_train, y_train, preprocessor)
    print("Best parameters:", best_params)

    # Evaluate
    baseline_rmse, rmse, mae = evaluate_model(best_model, X_test, y_test, y_train)
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Fit full training data
    best_model.fit(X, y)

    # Save model into dictionary
    save_model_to_dict(best_model, "catboost", MODELS_DICT_FILE)
    print("CatBoost model saved to dictionary.")

    # Submission
    submission = create_submission(best_model, df_test, SUBMISSION_PATH)
    print("Submission file created.")
    print(submission.head())
    print("All checks passed.")


if __name__ == "__main__":
    main()
