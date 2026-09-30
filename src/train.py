"""
Model training and hyperparameter tuning module for Experiment 1 and Experiment 2.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.config import RANDOM_STATE, MODELS_DIR

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def get_model_pipelines(preprocessor, class_weight_mode: str = "balanced") -> Dict[str, Pipeline]:
    """
    Constructs sklearn Pipelines combining the ColumnTransformer preprocessor and classifiers.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced" if class_weight_mode == "balanced" else None,
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced" if class_weight_mode == "balanced" else None,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    if HAS_XGBOOST:
        # scale_pos_weight for imbalance ~ 10422 / 1908 ≈ 5.46
        scale_pos = 5.46 if class_weight_mode == "balanced" else 1.0
        pipelines["XGBoost"] = Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

    return pipelines

def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """
    Defines parameter search distributions for hyperparameter tuning.
    """
    grids = {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs", "liblinear"]
        },
        "Random Forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 15, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4]
        }
    }

    if HAS_XGBOOST:
        grids["XGBoost"] = {
            "clf__n_estimators": [100, 150, 200],
            "clf__max_depth": [3, 4, 6],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0]
        }

    return grids

def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    tune: bool = True,
    n_iter: int = 8,
    cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Trains all candidate models, optionally tuning hyperparameters via RandomizedSearchCV.
    """
    pipelines = get_model_pipelines(preprocessor, class_weight_mode="balanced")
    grids = get_hyperparameter_grids()
    trained_models = {}

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, pipe in pipelines.items():
        print(f"\n[Training] Starting {name}...")
        if tune and name in grids:
            print(f"[Tuning] Running RandomizedSearchCV for {name} ({n_iter} iterations)...")
            search = RandomizedSearchCV(
                pipe,
                param_distributions=grids[name],
                n_iter=n_iter,
                scoring="average_precision",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            print(f"[Tuning] Best CV PR-AUC for {name}: {search.best_score_:.4f}")
            print(f"[Tuning] Best Params: {search.best_params_}")
            trained_models[name] = {
                "pipeline": best_pipe,
                "best_params": search.best_params_,
                "cv_score": search.best_score_
            }
        else:
            pipe.fit(X_train, y_train)
            trained_models[name] = {
                "pipeline": pipe,
                "best_params": "default",
                "cv_score": None
            }

    return trained_models

def save_model_artifacts(pipeline, model_name: str = "best_model.joblib"):
    """
    Saves trained pipeline to models directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, model_name)
    joblib.dump(pipeline, out_path)
    print(f"[ModelSaver] Artifact saved successfully to: {out_path}")
