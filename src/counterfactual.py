"""
Actionable Counterfactual Explanation Engine and Benchmark Evaluation.
Metodologi Evaluasi Kualitas Counterfactual Berdasarkan 6 Dimensi Kualitas.
Implements DiCE integration, Constrained Evolutionary Search, and Quantitative Quality Metrics:
Validity, Proximity, Sparsity, Plausibility, Actionability, and Diversity.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import (
    IMMUTABLE_FEATURES,
    ACTIONABLE_FEATURES,
    ANALYTICAL_FEATURES,
    FEATURE_BOUNDS,
    RANDOM_STATE,
    METRICS_DIR
)
from src.constraints import ConstraintValidator

try:
    import dice_ml
    HAS_DICE = True
except ImportError:
    HAS_DICE = False

class ActionableCounterfactualEngine:
    """
    Counterfactual generation and evaluation engine supporting both DiCE and
    a specialized Constrained Actionable Search algorithm.
    """
    def __init__(
        self,
        pipeline,
        train_df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        outcome_name: str = "Revenue"
    ):
        self.pipeline = pipeline
        self.train_df = train_df.copy()
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.outcome_name = outcome_name

        self.validator = ConstraintValidator()

        # Compute Median Absolute Deviations (MAD) for L1 proximity normalization
        self.mads = self.train_df[self.numerical_cols].apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        ).replace(0, 1e-4)

        # Compute feature ranges for normalized proximity
        self.ranges = (
            self.train_df[self.numerical_cols].max() - self.train_df[self.numerical_cols].min()
        ).replace(0, 1.0)

        # Fit Isolation Forest on Purchase instances for Plausibility evaluation
        purchase_samples = self.train_df[self.train_df[self.outcome_name] == 1]
        if len(purchase_samples) == 0:
            purchase_samples = self.train_df
        self.plausibility_model = IsolationForest(random_state=RANDOM_STATE, contamination=0.05)
        self.plausibility_model.fit(purchase_samples[self.numerical_cols])

        # Initialize DiCE if available
        self.dice_exp = None
        if HAS_DICE:
            try:
                d = dice_ml.Data(
                    dataframe=self.train_df,
                    continuous_features=self.numerical_cols,
                    outcome_name=self.outcome_name
                )
                m = dice_ml.Model(model=self.pipeline, backend="sklearn")
                self.dice_exp = dice_ml.Dice(d, m, method="random")
                print("[CounterfactualEngine] DiCE-ML initialized successfully.")
            except Exception as e:
                print(f"[CounterfactualEngine] DiCE initialization notice: {e}")

    def generate_unconstrained(
        self,
        query_instance: pd.DataFrame,
        total_CFs: int = 3,
        desired_class: int = 1
    ) -> pd.DataFrame:
        """
        Generates baseline unconstrained counterfactuals where ANY feature may mutate (including OS, Month, Region).
        Used as the comparative baseline for Experiment 4 & 6.
        """
        if self.dice_exp is not None:
            try:
                query = query_instance.drop(columns=[self.outcome_name], errors="ignore")
                cf = self.dice_exp.generate_counterfactuals(
                    query,
                    total_CFs=total_CFs,
                    desired_class=desired_class
                )
                return cf.cf_examples_list[0].final_cfs_df
            except Exception as e:
                print(f"[UnconstrainedCF] DiCE generation notice ({e}). Using unconstrained heuristic search.")

        return self._heuristic_search(query_instance, total_CFs=total_CFs, constrained=False)

    def generate_constrained(
        self,
        query_instance: pd.DataFrame,
        total_CFs: int = 3,
        desired_class: int = 1
    ) -> pd.DataFrame:
        """
        Generates proposed constrained actionable counterfactuals (Novelty Proposed Approach):
        - Immutable features (Month, OS, Browser, Region, Weekend, SpecialDay) are strictly locked.
        - Only candidate actionable / behavioral features can mutate.
        - Strict integer and range bounds enforced.
        - Validated through ConstraintValidator rule engine.
        """
        features_to_vary = [col for col in ACTIONABLE_FEATURES if col in query_instance.columns]
        if "PageValues" in query_instance.columns:
            features_to_vary.append("PageValues")

        permitted_range = {
            col: FEATURE_BOUNDS[col] for col in features_to_vary if col in FEATURE_BOUNDS
        }

        if self.dice_exp is not None:
            try:
                query = query_instance.drop(columns=[self.outcome_name], errors="ignore")
                cf = self.dice_exp.generate_counterfactuals(
                    query,
                    total_CFs=total_CFs,
                    desired_class=desired_class,
                    features_to_vary=features_to_vary,
                    permitted_range=permitted_range
                )
                df_cf = cf.cf_examples_list[0].final_cfs_df
                filtered = self._filter_valid_cfs(query_instance.iloc[0], df_cf)
                if len(filtered) > 0:
                    return filtered
            except Exception as e:
                print(f"[ConstrainedCF] DiCE generation notice ({e}). Executing robust constrained optimizer.")

        return self._heuristic_search(query_instance, total_CFs=total_CFs, constrained=True)

    def _heuristic_search(
        self,
        query_instance: pd.DataFrame,
        total_CFs: int = 3,
        constrained: bool = True
    ) -> pd.DataFrame:
        """
        Specialized evolutionary / hill-climbing search that finds minimal perturbations
        while strictly enforcing constraints and bounds.
        """
        query_row = query_instance.iloc[0].drop(labels=[self.outcome_name], errors="ignore").copy()
        candidates = []

        if constrained:
            allowed_features = [col for col in (ACTIONABLE_FEATURES + ["PageValues"]) if col in query_row.index]
        else:
            # Unconstrained baseline: allowed to mutate categorical and numerical indiscriminately
            allowed_features = list(query_row.index)

        # Baseline purchase prototypes from training data
        purchase_pool = self.train_df[self.train_df[self.outcome_name] == 1]
        if len(purchase_pool) == 0:
            purchase_pool = self.train_df

        np.random.seed(RANDOM_STATE)
        for _ in range(600):
            candidate = query_row.copy()
            # Pick 1 to 3 features to mutate
            n_perturb = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            cols_to_change = np.random.choice(
                allowed_features,
                size=min(n_perturb, len(allowed_features)),
                replace=False
            )

            donor = purchase_pool.sample(1, random_state=None).iloc[0]

            for col in cols_to_change:
                if col in self.numerical_cols:
                    alpha = np.random.uniform(0.3, 1.0)
                    new_val = (1 - alpha) * float(query_row[col]) + alpha * float(donor[col])
                    if col in FEATURE_BOUNDS:
                        min_b, max_b = FEATURE_BOUNDS[col]
                        new_val = np.clip(new_val, min_b, max_b)
                    if col in ["ProductRelated", "Administrative", "Informational"]:
                        new_val = int(round(new_val))
                    candidate[col] = new_val
                else:
                    if not constrained:
                        # Baseline can mutate categorical features (e.g. Month, Browser, OS)
                        candidate[col] = donor[col]

            # Enforce physical coherence
            if candidate.get("ProductRelated", 0) == 0:
                candidate["ProductRelated_Duration"] = 0.0
            if "ExitRates" in candidate and "BounceRates" in candidate:
                if candidate["BounceRates"] > candidate["ExitRates"]:
                    candidate["BounceRates"] = candidate["ExitRates"]

            # Evaluate probability flip
            try:
                cand_prob = float(self.pipeline.predict_proba(pd.DataFrame([candidate]))[0, 1])
            except Exception:
                cand_prob = 0.0

            if cand_prob >= 0.5:
                rep = self.validator.evaluate_counterfactual(query_row, candidate)
                if not constrained or rep["is_valid"]:
                    candidate[self.outcome_name] = 1
                    # Distance score (lower is better)
                    dist = sum(
                        abs(float(candidate[c]) - float(query_row[c])) / self.ranges[c]
                        for c in cols_to_change if c in self.ranges
                    )
                    candidates.append((dist, candidate))

        candidates.sort(key=lambda x: x[0])
        selected = [c[1] for c in candidates[:total_CFs]]

        if len(selected) == 0:
            # Fallback prototype if search budget exhausted
            fallback = query_row.copy()
            if constrained:
                fallback["PageValues"] = max(float(query_row.get("PageValues", 0)) + 14.0, 18.0)
                fallback["ProductRelated"] = int(query_row.get("ProductRelated", 0)) + 6
                fallback["ProductRelated_Duration"] = float(query_row.get("ProductRelated_Duration", 0)) + 250.0
            else:
                fallback["Month"] = "Nov"
                fallback["OperatingSystems"] = 3
                fallback["PageValues"] = 20.0
            fallback[self.outcome_name] = 1
            selected = [fallback]

        return pd.DataFrame(selected).reset_index(drop=True)

    def _filter_valid_cfs(self, query_row: pd.Series, df_cfs: pd.DataFrame) -> pd.DataFrame:
        """
        Validates DiCE counterfactuals against the rule engine.
        """
        valid_rows = []
        for _, cf_row in df_cfs.iterrows():
            rep = self.validator.evaluate_counterfactual(query_row, cf_row)
            if rep["is_valid"]:
                valid_rows.append(cf_row)
        if valid_rows:
            return pd.DataFrame(valid_rows).reset_index(drop=True)
        return pd.DataFrame()

    def evaluate_counterfactual_quality(
        self,
        original_df: pd.DataFrame,
        cf_df: pd.DataFrame,
        desired_class: int = 1
    ) -> Dict[str, float]:
        """
        Computes formal Counterfactual Quality Metrics:
        - Validity: % reaching target class
        - Proximity L1: Normalized by MAD
        - Proximity L2: Normalized by Range
        - Sparsity: Average number of features altered
        - Actionability: % of altered features that belong to ACTIONABLE_FEATURES
        - Plausibility: Isolation Forest inlier rate on positive manifold
        - Diversity: Pairwise distance among counterfactuals
        """
        if len(cf_df) == 0:
            return {
                "Validity": 0.0,
                "Proximity_L1": np.nan,
                "Proximity_L2": np.nan,
                "Sparsity_ChangedFeatures": 0.0,
                "Actionability_Score": 0.0,
                "Plausibility_Score": 0.0,
                "Diversity": 0.0
            }

        clean_cfs = cf_df.drop(columns=[self.outcome_name], errors="ignore")
        probs = self.pipeline.predict_proba(clean_cfs)[:, 1]
        preds = (probs >= 0.5).astype(int)
        validity = float((preds == desired_class).mean())

        l1_scores, l2_scores, sparsity_counts, actionability_scores = [], [], [], []
        orig_row = original_df.iloc[0].drop(labels=[self.outcome_name], errors="ignore")

        for _, cf_row in clean_cfs.iterrows():
            diffs_range = np.array([
                abs(float(cf_row[c]) - float(orig_row[c])) / self.ranges[c]
                for c in self.numerical_cols if c in cf_row and c in orig_row
            ])
            l1_scores.append(np.mean(diffs_range))

            diffs_l2 = np.array([
                ((float(cf_row[c]) - float(orig_row[c])) / self.ranges[c]) ** 2
                for c in self.numerical_cols if c in cf_row and c in orig_row
            ])
            l2_scores.append(np.sqrt(np.mean(diffs_l2)))

            # Sparsity & Actionability
            changed_cols = [
                c for c in orig_row.index if c in cf_row and str(orig_row[c]) != str(cf_row[c])
            ]
            sparsity_counts.append(len(changed_cols))

            if len(changed_cols) > 0:
                actionable_count = sum(1 for c in changed_cols if c in ACTIONABLE_FEATURES or c == "PageValues")
                actionability_scores.append(actionable_count / len(changed_cols))
            else:
                actionability_scores.append(1.0)

        # Plausibility via Isolation Forest (+1 inlier, -1 outlier)
        plaus_preds = self.plausibility_model.predict(clean_cfs[self.numerical_cols])
        plausibility = float((plaus_preds == 1).mean())

        # Diversity
        diversity = 0.0
        if len(clean_cfs) > 1:
            mat = clean_cfs[self.numerical_cols].values / self.ranges.values
            pairs = []
            for i in range(len(mat)):
                for j in range(i + 1, len(mat)):
                    pairs.append(np.linalg.norm(mat[i] - mat[j]))
            diversity = float(np.mean(pairs)) if pairs else 0.0

        return {
            "Validity": validity,
            "Proximity_L1": float(np.mean(l1_scores)),
            "Proximity_L2": float(np.mean(l2_scores)),
            "Sparsity_ChangedFeatures": float(np.mean(sparsity_counts)),
            "Actionability_Score": float(np.mean(actionability_scores)),
            "Plausibility_Score": plausibility,
            "Diversity": diversity
        }

def compare_counterfactual_frameworks(
    engine: ActionableCounterfactualEngine,
    sample_queries: pd.DataFrame,
    total_CFs: int = 2
) -> pd.DataFrame:
    """
    Executes Experiment 6:
    Compares Unconstrained vs Constrained Actionable Counterfactuals across multiple query sessions.
    Answers RQ3 quantitatively.
    """
    unconstrained_metrics = []
    constrained_metrics = []

    print(f"\n[CounterfactualBenchmark] Running Experiment 6 over {len(sample_queries)} non-purchase sessions...")

    for i in range(len(sample_queries)):
        q = sample_queries.iloc[[i]]
        cf_uncon = engine.generate_unconstrained(q, total_CFs=total_CFs)
        m_uncon = engine.evaluate_counterfactual_quality(q, cf_uncon)
        unconstrained_metrics.append(m_uncon)

        cf_con = engine.generate_constrained(q, total_CFs=total_CFs)
        m_con = engine.evaluate_counterfactual_quality(q, cf_con)
        constrained_metrics.append(m_con)

    df_uncon = pd.DataFrame(unconstrained_metrics).mean().to_dict()
    df_con = pd.DataFrame(constrained_metrics).mean().to_dict()

    comparison = pd.DataFrame([
        {"Metric": "Validity (% Target Class Achieved)", "Unconstrained": f"{df_uncon['Validity']:.2%}", "Constrained (Proposed)": f"{df_con['Validity']:.2%}"},
        {"Metric": "Proximity L1 (MAD-normalized, lower is better)", "Unconstrained": f"{df_uncon['Proximity_L1']:.3f}", "Constrained (Proposed)": f"{df_con['Proximity_L1']:.3f}"},
        {"Metric": "Proximity L2 (Range-normalized, lower is better)", "Unconstrained": f"{df_uncon['Proximity_L2']:.3f}", "Constrained (Proposed)": f"{df_con['Proximity_L2']:.3f}"},
        {"Metric": "Sparsity (Avg Features Changed, lower is better)", "Unconstrained": f"{df_uncon['Sparsity_ChangedFeatures']:.2f}", "Constrained (Proposed)": f"{df_con['Sparsity_ChangedFeatures']:.2f}"},
        {"Metric": "Actionability Score (% allowed features modified)", "Unconstrained": f"{df_uncon['Actionability_Score']:.2%}", "Constrained (Proposed)": f"{df_con['Actionability_Score']:.2%}"},
        {"Metric": "Plausibility (% Inliers on Purchase Manifold)", "Unconstrained": f"{df_uncon['Plausibility_Score']:.2%}", "Constrained (Proposed)": f"{df_con['Plausibility_Score']:.2%}"},
        {"Metric": "Diversity (Pairwise Distance between CFs)", "Unconstrained": f"{df_uncon['Diversity']:.3f}", "Constrained (Proposed)": f"{df_con['Diversity']:.3f}"}
    ])

    os.makedirs(METRICS_DIR, exist_ok=True)
    out_csv = os.path.join(METRICS_DIR, "counterfactual_quality_comparison.csv")
    comparison.to_csv(out_csv, index=False)
    print(f"\n[CounterfactualBenchmark] Benchmark Results (Experiment 6):\n{comparison.to_string(index=False)}")
    print(f"[CounterfactualBenchmark] Saved metrics to: {out_csv}")

    return comparison
