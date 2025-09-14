# GDSC Task – Model Training and Blending

This repository contains my work for the GDSC task, where I trained multiple machine learning models and created a **weighted blend** of them for the final submission.

---

## Project Overview

- **Goal**: Predict the target variable from the provided dataset.  
- **Dataset**: [Add dataset source or description here – e.g., provided by GDSC task organizers].  
- **Metric**: Root Mean Squared Error (RMSE).  
- **Final Approach**: A weighted blend of **XGBoost**, **LightGBM**, and **CatBoost** models.

---

## Approaches Tried

I experimented with several methods before finalizing the blended model:

- Built baseline models to establish reference performance.  
- Tuned models using **RandomizedSearchCV** for hyperparameter optimization. (I also tried **optuna** for tuning but it didn't give me the expected results)
- Attempted **model stacking**, but it did not provide the expected improvements.  
- Final approach: **Model blending (weighted average of XGBoost, LightGBM, and CatBoost)** gave the best results.

---

## Repository Structure
GDSC-task-notebooks/
│
├── models/
│ ├── xgboost_model.py # XGBoost model training
│ ├── lgbm_tuned.py # Tuned LightGBM model (RandomizedSearchCV)
│ ├── catboost_model.py # CatBoost model
│ └── blending_allmodels.py # Weighted blend of all models (final)
│
├── results/
│ └── final_submission.csv # Output submission file
│
├── README.md

---

## How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/rachitmittall/GDSC-task-notebooks.git
   cd GDSC-task-notebooks
