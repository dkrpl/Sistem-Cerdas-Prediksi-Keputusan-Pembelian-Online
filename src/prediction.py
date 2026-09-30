"""
Prediction Engine module for Online Shoppers Purchase Decision Prediction.
Sistem Inferensi Prediktif Keputusan Pembelian Online Berbasis Machine Learning.
Implements input validation, preprocessor execution, and probability inference.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    ALL_PREDICTORS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_BOUNDS
)

class TupleValidation:
    """Helper container for input validation status."""
    def __init__(self, is_valid: bool, missing: list, warnings: list):
        self.is_valid = is_valid
        self.missing = missing
        self.warnings = warnings

class PurchasePredictionEngine:
    """
    Inference engine that encapsulates the preprocessing pipeline and trained classifier.
    Supports single-session and batch inference with rigorous input validation.
    """
    def __init__(self, model_path: Optional[str] = None, pipeline=None):
        self.pipeline = pipeline
        self.model_path = model_path or os.path.join(MODELS_DIR, "best_model.joblib")

        if self.pipeline is None and os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                print(f"[PredictionEngine] Successfully loaded model pipeline from: {self.model_path}")
            except Exception as e:
                print(f"[PredictionEngine] Warning: Failed to load pipeline ({e}). Fallback heuristic will be used.")
                self.pipeline = None

    def validate_input(self, input_df: pd.DataFrame) -> TupleValidation:
        """
        Validates whether input dataframe contains expected predictor features and checks ranges.
        """
        missing_cols = [c for c in ALL_PREDICTORS if c not in input_df.columns]
        range_warnings = []

        for col, (min_v, max_v) in FEATURE_BOUNDS.items():
            if col in input_df.columns:
                val = float(input_df[col].iloc[0])
                if val < min_v or val > max_v:
                    range_warnings.append(f"Nilai '{col}' ({val}) di luar batas wajar [{min_v}, {max_v}].")

        return len(missing_cols) == 0, missing_cols, range_warnings

    def predict_session(self, session_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predicts purchase outcome for a single e-commerce user session.
        Returns:
            Dictionary containing prediction label, probabilities, and confidence level.
        """
        if isinstance(session_data, dict):
            df_input = pd.DataFrame([session_data])
        else:
            df_input = session_data.copy()

        # Ensure all columns exist with defaults if missing
        defaults = {
            "Administrative": 0, "Administrative_Duration": 0.0,
            "Informational": 0, "Informational_Duration": 0.0,
            "ProductRelated": 1, "ProductRelated_Duration": 20.0,
            "BounceRates": 0.0, "ExitRates": 0.02, "PageValues": 0.0,
            "SpecialDay": 0.0, "Month": "May", "OperatingSystems": 2,
            "Browser": 2, "Region": 1, "TrafficType": 2,
            "VisitorType": "Returning_Visitor", "Weekend": False
        }
        for col, default_val in defaults.items():
            if col not in df_input.columns:
                df_input[col] = default_val

        # Inference
        if self.pipeline is not None:
            prob_raw = self.pipeline.predict_proba(df_input[ALL_PREDICTORS])[0, 1]
            prob_purchase = float(prob_raw)
        else:
            # Domain-informed rational heuristic fallback if model artifact not yet generated
            pv = float(df_input["PageValues"].iloc[0])
            pr = float(df_input["ProductRelated"].iloc[0])
            ex = float(df_input["ExitRates"].iloc[0])
            score = 0.08 + (pv * 0.025) + (min(pr, 60) * 0.005) - (ex * 2.0)
            prob_purchase = float(np.clip(score, 0.02, 0.98))

        prob_non_purchase = float(1.0 - prob_purchase)
        predicted_class = 1 if prob_purchase >= 0.5 else 0
        label = "AKAN MEMBELI (PURCHASE)" if predicted_class == 1 else "TIDAK MEMBELI (NON-PURCHASE)"

        # Confidence calculation
        distance_from_threshold = abs(prob_purchase - 0.5)
        if distance_from_threshold >= 0.30:
            confidence = "Tinggi"
        elif distance_from_threshold >= 0.15:
            confidence = "Sedang"
        else:
            confidence = "Marjinal (Di Batas Keputusan)"

        return {
            "prediction_label": label,
            "predicted_class": predicted_class,
            "purchase_probability": prob_purchase,
            "non_purchase_probability": prob_non_purchase,
            "confidence_level": confidence,
            "input_features": df_input[ALL_PREDICTORS].iloc[0].to_dict()
        }

    def predict_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes prediction on a batch of sessions.
        """
        result_df = batch_df.copy()
        if self.pipeline is not None:
            probs = self.pipeline.predict_proba(batch_df[ALL_PREDICTORS])[:, 1]
            result_df["Purchase_Probability"] = [float(p) for p in probs]
            result_df["Predicted_Class"] = (result_df["Purchase_Probability"] >= 0.5).astype(int)
            result_df["Predicted_Outcome"] = np.where(
                result_df["Predicted_Class"] == 1,
                "AKAN MEMBELI (PURCHASE)",
                "TIDAK MEMBELI (NON-PURCHASE)"
            )
        else:
            probs = []
            for _, row in batch_df.iterrows():
                res = self.predict_session(pd.DataFrame([row]))
                probs.append(res["purchase_probability"])
            result_df["Purchase_Probability"] = probs
            result_df["Predicted_Class"] = (np.array(probs) >= 0.5).astype(int)
            result_df["Predicted_Outcome"] = np.where(
                result_df["Predicted_Class"] == 1,
                "AKAN MEMBELI (PURCHASE)",
                "TIDAK MEMBELI (NON-PURCHASE)"
            )
        return result_df

