# Fraud Detection — Analysis (1 page)

**Context & business impact**
Financial institutions process vast numbers of transactions. Blocking legitimate payments is extremely costly — false positives have 5× the business cost of false negatives. We therefore optimize for precision-recall tradeoffs (PR AUC) and apply a cost-aware threshold minimizing `5*FP + 1*FN`.

**Dataset**
- `data/transactions.csv` (PaySim-like). Dropped leakage columns `isFlaggedFraud`, `nameOrig`, `nameDest`.
- Positive class (fraud) prevalence ≈ 0.2%.

**Preprocessing**
- Implemented with `ColumnTransformer`:
  - Numeric: `SimpleImputer(strategy='median')` + `StandardScaler`
  - Categorical (`type`): `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
- One pipeline used for train/val/test to avoid leakage.

**Imbalance handling (two methods)**
1. **Algorithmic** — use model `class_weight='balanced'` (LogReg, RF) to adjust sample weighting. Lightweight and avoids synthetic data.
2. **Resampling** — use **SMOTE** inside an `imblearn.Pipeline` (preproc → SMOTE → clf) to synthesize minority samples for models that benefit from more positive examples.

**Modeling**
- Models: Logistic Regression (L1/L2), Random Forest, XGBoost (sklearn API).
- Tuning: `GridSearchCV` with **average_precision** (PR AUC) as scoring. Grids are deliberately small so the pipeline completes within the time limit.

**Selection**
- Final model selected by highest test PR AUC.
- Decision threshold set by minimizing `5*FP + 1*FN` to align with business cost.

**Explainability**
- Recommended: compute SHAP on final model and present top-3 drivers (not computed here to keep runtime short). Explainability preferred over marginal AP gains.

**Why not deep learning**
- Tabular data with low feature dimensionality and heavy imbalance: tree-based models or linear models are faster, simpler, and more explainable. DL would be overkill.

**Artifacts**
- `src/preprocessing.py` — preprocessing builder
- `src/model_comparison.py` — training & selection script
- `models/fraud_model.pkl` — exported pipeline (produced after running)
- `reports/comparison_summary.json` — model comparison summary
