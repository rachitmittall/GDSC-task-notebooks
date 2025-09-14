import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load test dataset
# -------------------------------
TEST_PATH = r"C:\Users\Rachit Mittal\OneDrive\Documents\GDSC_Project_Database\MiNDAT_UNK.csv"
df_unk = pd.read_csv(TEST_PATH)
X_unk = df_unk.drop(columns=['CORRUCYSTIC_DENSITY'], errors='ignore')

# -------------------------------
# Load models
# -------------------------------
all_models = joblib.load("all_models.pkl")
new_lgbm = joblib.load("lgbm_model.pkl")  # LGBM loaded separately

# -------------------------------
# Generate predictions
# -------------------------------
preds = {
    "lgbm": new_lgbm.predict(X_unk),
    "catboost": all_models["catboost"].predict(X_unk),
    "xgb": all_models["xgb"].predict(X_unk),
}

# -------------------------------
# Weighted blending
# -------------------------------
y_blend = (
    0.50 * preds["lgbm"] +
    0.30 * preds["catboost"] +
    0.20 * preds["xgb"]
)

# -------------------------------
# Create submission file
# -------------------------------
submission = pd.DataFrame({
    "LOCAL_IDENTIFIER": df_unk["LOCAL_IDENTIFIER"],
    "CORRUCYSTIC_DENSITY": y_blend
})

submission.to_csv("submission_blend_safe2.csv", index=False, float_format="%.6f")
print("Submission file 'submission_blend_safe2.csv' created successfully!")

# -------------------------------
# Final sanity checks
# -------------------------------
assert len(submission) == len(df_unk), "Row count mismatch!"
assert submission['LOCAL_IDENTIFIER'].is_unique, "Duplicate IDs found!"
assert list(submission.columns) == ["LOCAL_IDENTIFIER", "CORRUCYSTIC_DENSITY"], "Wrong column order!"

print("All sanity checks passed.")
print(submission.head())

"""
In this I used many different combinations of weightage and every combination gave different results.
This may or may not be the weights correspomding to the best resluts.
I also tried model stacking instead of blending but that didn't give me the expected results so I  sticked to blending only.
In model hyperparameter tuning I also used optuna but that also didn't go as expected so this was the final path that I used.
"""
