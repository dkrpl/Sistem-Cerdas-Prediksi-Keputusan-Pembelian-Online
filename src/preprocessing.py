"""
Preprocessing module implementing CRISP-DM Phase 2 with strict anti-data-leakage guarantees.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE
)

def create_preprocessor(numerical_cols: List[str] = None, categorical_cols: List[str] = None) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def split_and_prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    drop_page_values: bool = False,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Splits dataset into stratified train and test sets, ensuring no data leakage.
    Returns:
        X_train, X_test, y_train, y_test, num_features, cat_features
    """
    df_clean = df.copy()
    num_features = NUMERICAL_FEATURES.copy()
    cat_features = CATEGORICAL_FEATURES.copy()

    if drop_page_values and "PageValues" in num_features:
        num_features.remove("PageValues")
        df_clean = df_clean.drop(columns=["PageValues"])

    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[DataSplit] Train instances: {len(X_train)} (Positive: {y_train.sum()}, Rate: {y_train.mean():.2%})")
    print(f"[DataSplit] Test instances : {len(X_test)} (Positive: {y_test.sum()}, Rate: {y_test.mean():.2%})")

    return X_train, X_test, y_train, y_test, num_features, cat_features

def apply_smote(
    X_train_proc: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE strictly to training data only. Test set is NEVER oversampled.
    """
    if not HAS_SMOTE:
        print("[SMOTE] Notice: imbalanced-learn is not installed in local environment. Returning original train data.")
        return X_train_proc, y_train.values

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_proc, y_train)
    print(f"[SMOTE] Resampled train size from {len(y_train)} to {len(y_resampled)} (Positive: {y_resampled.sum()})")
    return X_resampled, y_resampled
