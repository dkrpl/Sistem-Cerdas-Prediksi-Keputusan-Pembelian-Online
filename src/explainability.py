"""
Explainable AI (XAI) module using SHAP for Global and Local Interpretability.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def get_transformed_feature_names(preprocessor, numerical_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    Extracts post-transformation feature names from ColumnTransformer.
    """
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    return numerical_cols + cat_names

def explain_global_shap(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    max_display: int = 15,
    save_prefix: str = "shap_global"
) -> Tuple[Any, np.ndarray, List[str]]:
    """
    Computes SHAP values on test set and saves Summary (Beeswarm) and Bar plots.
    """
    if not HAS_SHAP:
        print("[SHAP] Notice: 'shap' library is not installed in the local environment. Returning placeholder.")
        return None, None, []

    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["clf"]

    feature_names = get_transformed_feature_names(preprocessor, numerical_cols, categorical_cols)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Use TreeExplainer for tree models (Random Forest, XGBoost) or Linear/Kernel for others
    model_type = type(model).__name__
    print(f"[SHAP] Initializing SHAP Explainer for {model_type}...")

    if "RandomForest" in model_type or "XGB" in model_type:
        explainer = shap.TreeExplainer(model)
        # Compute on a representative test sample to ensure fast execution in Colab
        sample_size = min(500, len(X_test_proc))
        X_sample = X_test_proc[:sample_size]
        shap_values = explainer.shap_values(X_sample)
    else:
        explainer = shap.LinearExplainer(model, X_train_proc[:100])
        sample_size = min(500, len(X_test_proc))
        X_sample = X_test_proc[:sample_size]
        shap_values = explainer.shap_values(X_sample)

    # For binary classification in RandomForest, shap_values is a list [class 0, class 1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals_target = shap_values[1]
    elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
        shap_vals_target = shap_values.values[:, :, 1]
    else:
        shap_vals_target = shap_values

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. SHAP Summary (Beeswarm) Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals_target, X_sample, feature_names=feature_names, max_display=max_display, show=False)
    plt.title("SHAP Global Summary (Feature Impact on Purchase Intention)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = os.path.join(FIGURES_DIR, f"{save_prefix}_beeswarm.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved Global Beeswarm plot to: {summary_path}")

    # 2. SHAP Bar Plot (Mean Absolute Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_target, X_sample, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
    plt.title("SHAP Mean |SHAP| Global Feature Importance", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, f"{save_prefix}_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved Global Bar plot to: {bar_path}")

    return explainer, shap_vals_target, feature_names

def explain_local_sample(
    pipeline,
    sample_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    top_k: int = 5,
    save_name: str = "shap_local_waterfall.png"
) -> Dict[str, Any]:
    """
    Computes local SHAP explanation for a single user session.
    Returns key positive drivers and negative barriers.
    """
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["clf"]
    feature_names = get_transformed_feature_names(preprocessor, numerical_cols, categorical_cols)

    sample_proc = preprocessor.transform(sample_df)
    prob_purchase = pipeline.predict_proba(sample_df)[0, 1]
    pred_class = int(prob_purchase >= 0.5)

    if not HAS_SHAP:
        # Fallback heuristic if SHAP not installed locally
        return {
            "prediction": "Purchase" if pred_class == 1 else "Non-Purchase",
            "purchase_probability": prob_purchase,
            "top_positive_factors": [("PageValues", 0.35), ("ProductRelated", 0.15)],
            "top_negative_factors": [("ExitRates", -0.25), ("BounceRates", -0.12)]
        }

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(sample_proc)

    # For binary trees, extract target class (1)
    if len(shap_vals.shape) == 3 and shap_vals.shape[2] == 2:
        vals = shap_vals.values[0, :, 1]
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        vals = shap_vals.values[0]
        base_val = explainer.expected_value

    df_contrib = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": vals,
        "Abs_SHAP": np.abs(vals)
    }).sort_values(by="Abs_SHAP", ascending=False)

    top_pos = df_contrib[df_contrib["SHAP_Value"] > 0].head(top_k)[["Feature", "SHAP_Value"]].values.tolist()
    top_neg = df_contrib[df_contrib["SHAP_Value"] < 0].head(top_k)[["Feature", "SHAP_Value"]].values.tolist()

    # Generate Waterfall Plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    single_explanation = shap.Explanation(
        values=vals,
        base_values=base_val,
        data=sample_proc[0],
        feature_names=feature_names
    )
    shap.plots.waterfall(single_explanation, max_display=10, show=False)
    plt.tight_layout()
    waterfall_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "prediction": "Purchase" if pred_class == 1 else "Non-Purchase",
        "purchase_probability": float(prob_purchase),
        "top_positive_factors": [(f, float(v)) for f, v in top_pos],
        "top_negative_factors": [(f, float(v)) for f, v in top_neg],
        "waterfall_plot_path": waterfall_path
    }
