"""
src/preprocessing.py

Preprocessing pipeline using ColumnTransformer + Pipeline.
Provides:
- build_preprocessor(): ColumnTransformer for numeric + categorical handling
- load_dataset(): loads CSV, drops leakage cols
- build_smote_pipeline(): helper to build imblearn pipeline with SMOTE (if desired)
"""
from typing import List, Tuple, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

LEAKAGE_COLS = ["isFlaggedFraud", "nameOrig", "nameDest"]

NUMERIC_FEATURES = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CATEGORICAL_FEATURES = ["type"]

def load_dataset(path: str, target: str = "isFraud"):
    df = pd.read_csv(path)
    # Drop leakage columns if present
    for c in LEAKAGE_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    assert target in df.columns, f"Target column '{target}' not found"
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y

def build_preprocessor(numeric_features: Optional[List[str]] = None,
                       categorical_features: Optional[List[str]] = None,
                       sparse_ohe: bool = True) -> ColumnTransformer:
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=not sparse_ohe))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_ohe))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ], remainder="drop", sparse_threshold=0.3)
    return preprocessor
