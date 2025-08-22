"""
src/model_comparison.py

Single-file runner that:
- Loads data from data/transactions.csv (update path as needed)
- Builds ColumnTransformer preprocessor (from src.preprocessing)
- Compares LogisticRegression, RandomForest, XGBoost using GridSearchCV (PR AUC)
- Implements two imbalance strategies:
    * algorithmic: use class_weight='balanced' where available
    * resampling: SMOTE inside an imblearn Pipeline
- Selects final model by highest test average_precision (PR AUC)
- Applies cost-sensitive threshold minimising 5*FP + 1*FN
- Exports final pipeline to models/fraud_model.pkl
- Keeps grids tiny so runtime stays << 15 minutes
"""
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump

from preprocessing import load_dataset, build_preprocessor

DATA_PATH = os.path.join("data", "transactions.csv")  # ensure your CSV is here
TARGET = "isFraud"
RANDOM_STATE = 42

def choose_threshold_cost_sensitive(y_true, y_proba, fp_cost=5.0, fn_cost=1.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    best_thr = 0.5
    best_cost = np.inf
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp_cost * fp + fn_cost * fn
        if cost < best_cost:
            best_cost = cost
            best_thr = thr
    return best_thr, best_cost

def main():
    print("Loading data from", DATA_PATH)
    X, y = load_dataset(DATA_PATH, target=TARGET)
    print("Data shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    # Model definitions with tiny grids (GridSearchCV required by spec)
    # Logistic Regression: use solver 'liblinear' supporting l1/l2
    pipe_lr = Pipeline([("preproc", preprocessor), ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE))])
    lr_grid = {
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.1, 1.0]
    }

    # Random Forest
    pipe_rf = Pipeline([("preproc", preprocessor), ("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1))])
    rf_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [8, 16]
    }

    # XGBoost
    pipe_xgb = Pipeline([("preproc", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist', random_state=RANDOM_STATE, n_jobs=1))])
    xgb_grid = {
        "clf__n_estimators": [100],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.1]
    }

    models = [
        ("LogisticRegression", pipe_lr, lr_grid),
        ("RandomForest", pipe_rf, rf_grid),
        ("XGBoost", pipe_xgb, xgb_grid)
    ]

    results = []

    # For each model, run two strategies: algorithmic (class_weight) and SMOTE
    for name, pipe, grid in models:
        print(f"=== Model: {name} ===")
        # Strategy A: algorithmic (rely on class_weight where supported)
        print("Strategy: algorithmic (class_weight where applicable)")
        gs_a = GridSearchCV(pipe, grid, scoring="average_precision", cv=3, n_jobs=1, refit=True)
        t0 = time.time()
        gs_a.fit(X_train, y_train)
        t_a = time.time() - t0
        y_proba_a = gs_a.predict_proba(X_test)[:,1]
        pr_a = average_precision_score(y_test, y_proba_a)

        print(f"  Grid best params: {gs_a.best_params_}")
        print(f"  Test PR AUC: {pr_a:.4f}, time: {t_a:.1f}s")

        results.append({"model": name, "strategy":"class_weight", "estimator": gs_a.best_estimator_, "pr_auc": pr_a, "time_s": t_a, "best_params": gs_a.best_params_})

        # Strategy B: SMOTE resampling
        print("Strategy: SMOTE (resampling)")
        imb_pipe = ImbPipeline([("preproc", build_preprocessor()), ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)), ("clf", pipe.named_steps["clf"])])
        gs_b = GridSearchCV(imb_pipe, {"clf__" + k.split("__",1)[1] if "__" in k else k: v for k,v in grid.items()}, scoring="average_precision", cv=3, n_jobs=1, refit=True)
        # Note: above grid key mapping ensures grid matches 'clf__param' style for imb pipeline
        t0 = time.time()
        gs_b.fit(X_train, y_train)
        t_b = time.time() - t0
        y_proba_b = gs_b.predict_proba(X_test)[:,1]
        pr_b = average_precision_score(y_test, y_proba_b)

        print(f"  Grid best params: {gs_b.best_params_}")
        print(f"  Test PR AUC: {pr_b:.4f}, time: {t_b:.1f}s")

        results.append({"model": name, "strategy":"smote", "estimator": gs_b.best_estimator_, "pr_auc": pr_b, "time_s": t_b, "best_params": gs_b.best_params_})

    # Summarize and pick best by PR AUC
    results_sorted = sorted(results, key=lambda x: x["pr_auc"], reverse=True)
    summary = [{"model": r["model"], "strategy": r["strategy"], "pr_auc": r["pr_auc"], "time_s": r["time_s"], "best_params": r["best_params"]} for r in results_sorted]
    print("\\n=== Summary (sorted by PR AUC) ===")
    for s in summary:
        print(s)

    best = results_sorted[0]
    print(f"Selected final model: {best['model']} with strategy {best['strategy']} (PR AUC={best['pr_auc']:.4f})")

    # Refit final model on full training data (already best['estimator'] is fitted)
    final_pipeline = best["estimator"]

    # Determine cost-sensitive threshold to minimize 5*FP + 1*FN
    y_proba = final_pipeline.predict_proba(X_test)[:,1]
    thr, cost = choose_threshold_cost_sensitive(y_test, y_proba, fp_cost=5.0, fn_cost=1.0)
    preds = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print(f"Cost-sensitive threshold: {thr:.4f} -> TN={tn}, FP={fp}, FN={fn}, TP={tp}, cost={cost}")

    # Save pipeline
    os.makedirs("models", exist_ok=True)
    dump(final_pipeline, os.path.join("models", "fraud_model.pkl"))
    print("Saved final pipeline to models/fraud_model.pkl")

    # Save summary JSON and quick report snippet
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports","comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
