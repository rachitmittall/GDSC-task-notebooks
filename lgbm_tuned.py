import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TRAIN_PATH = r"C:\Users\Rachit Mittal\OneDrive\Documents\GDSC_Project_Database\MiNDAT.csv"
TEST_PATH = r"C:\Users\Rachit Mittal\OneDrive\Documents\GDSC_Project_Database\MiNDAT_UNK.csv"
MODEL_PATH = "lgbm_model.pkl"
SUBMISSION_PATH = "submission_lgbm_tuned.csv"

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
    """Builds preprocessing pipeline for numerical and categorical features."""
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
    """Train LightGBM model with RandomizedSearchCV."""
    param_dist = {
        "regressor__n_estimators": [1500],
        "regressor__max_depth": [16],
        "regressor__learning_rate": [0.01],
        "regressor__num_leaves": [250],
        "regressor__subsample": [0.6],
        "regressor__colsample_bytree": [0.6],
        "regressor__min_child_samples": [3]
    }
    """
    Here in parameters search I used many different parameters using 
    RandomizedSearchCV (which is used to search for different combinations of parameters)
    so these may or may not be the best parameters 
    but this was the last that I tried with only one value for each parameter.
    In my previous attempts I added multiple values. 
    """

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(random_state=42))
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
    print("Best parameters:", search.best_params_)
    return search.best_estimator_


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, y_train):
    """Evaluate model against test set and print metrics."""
    y_pred = model.predict(X_test)

    # Baseline prediction (mean of training labels)
    y_baseline = np.full_like(y_test, y_train.mean(), dtype=float)

    baseline_rmse = root_mean_squared_error(y_test, y_baseline)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")


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

    # Save submission
    submission.to_csv(submission_path, index=False, float_format="%.6f")
    print("Submission file created!")
    print(submission.head())

    # Sanity checks
    assert len(submission) == len(df_test), "Row count mismatch!"
    assert submission["LOCAL_IDENTIFIER"].is_unique, "Duplicate IDs found!"
    assert list(submission.columns) == ["LOCAL_IDENTIFIER", "CORRUCYSTIC_DENSITY"], "Wrong column order!"

    return submission


# -------------------------------------------------------------------
# Feature Importance
# -------------------------------------------------------------------
def plot_feature_importance(model, top_n=20):
    """Plot LightGBM feature importance."""
    lgb.plot_importance(model["regressor"], max_num_features=top_n)
    plt.show()


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
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Preprocessor
    preprocessor = build_preprocessor(df_train)

    # Train model
    best_model = train_model(X_train, y_train, preprocessor)

    # Evaluate
    evaluate_model(best_model, X_test, y_test, y_train)

    # Fit on full training data
    best_model.fit(X, y)

    # Save model
    joblib.dump(best_model, MODEL_PATH)

    # Submission
    create_submission(best_model, df_test, SUBMISSION_PATH)

    # Plot feature importance
    plot_feature_importance(best_model)


if __name__ == "__main__":
    main()




