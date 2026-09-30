"""
Constraint Validator and Rule Engine for Actionable Counterfactual Explanations.
Kaidah Domain E-Commerce dan Integritas Sesi Pengunjung.
Guarantees immutability, integer constraints, range feasibility, physical coherence, and actionability.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from src.config import (
    IMMUTABLE_FEATURES,
    ACTIONABLE_FEATURES,
    ANALYTICAL_FEATURES,
    FEATURE_BOUNDS
)

class ConstraintValidator:
    """
    Rule engine verifying that generated counterfactuals are feasible,
    respect immutable features, satisfy feature bounds, and follow domain rules.
    """
    def __init__(
        self,
        immutable_cols: List[str] = None,
        actionable_cols: List[str] = None,
        bounds: Dict[str, Tuple[float, float]] = None,
        max_changed_features: int = 4
    ):
        self.immutable_cols = immutable_cols if immutable_cols is not None else IMMUTABLE_FEATURES
        self.actionable_cols = actionable_cols if actionable_cols is not None else ACTIONABLE_FEATURES
        self.bounds = bounds if bounds is not None else FEATURE_BOUNDS
        self.max_changed_features = max_changed_features
        self.integer_features = ["ProductRelated", "Administrative", "Informational"]

    def validate_immutability(self, original_row: pd.Series, cf_row: pd.Series) -> Tuple[bool, List[str]]:
        """
        Ensures none of the immutable features were modified.
        """
        violations = []
        for col in self.immutable_cols:
            if col in original_row.index and col in cf_row.index:
                val_orig = original_row[col]
                val_cf = cf_row[col]
                if str(val_orig) != str(val_cf):
                    violations.append(f"Pelanggaran Fitur Tetap: '{col}' diubah dari {val_orig} menjadi {val_cf}.")
        return len(violations) == 0, violations

    def validate_ranges(self, cf_row: pd.Series) -> Tuple[bool, List[str]]:
        """
        Ensures all numerical values stay within realistic minimum and maximum boundaries.
        """
        violations = []
        for col, (min_val, max_val) in self.bounds.items():
            if col in cf_row.index:
                val = float(cf_row[col])
                if val < min_val or val > max_val:
                    violations.append(f"Batas Nilai Terlanggar: '{col}' bernilai {val:.2f}, di luar rentang [{min_val}, {max_val}].")
        return len(violations) == 0, violations

    def validate_integers(self, cf_row: pd.Series) -> Tuple[bool, List[str]]:
        """
        Ensures count features (e.g. number of pages visited) remain valid integers.
        """
        violations = []
        for col in self.integer_features:
            if col in cf_row.index:
                val = float(cf_row[col])
                if abs(val - round(val)) > 1e-4:
                    violations.append(f"Tipe Data Terlanggar: '{col}' bernilai pecahan ({val:.2f}), harus bilangan bulat.")
        return len(violations) == 0, violations

    def validate_coherence(self, cf_row: pd.Series) -> Tuple[bool, List[str]]:
        """
        Domain coherence rules:
        - Counts cannot have zero duration if count > 0, nor positive duration if count == 0.
        - BounceRates <= ExitRates in web analytics.
        """
        violations = []
        for page_col, dur_col in [
            ("ProductRelated", "ProductRelated_Duration"),
            ("Administrative", "Administrative_Duration"),
            ("Informational", "Informational_Duration")
        ]:
            if page_col in cf_row.index and dur_col in cf_row.index:
                count = float(cf_row[page_col])
                dur = float(cf_row[dur_col])
                if count == 0 and dur > 0:
                    violations.append(f"Inkoherensi Sesi: {page_col} = 0 halaman tetapi durasi {dur_col} tercatat {dur:.1f} detik.")
                elif count > 3 and dur == 0:
                    violations.append(f"Inkoherensi Sesi: {page_col} = {count:.0f} halaman tetapi durasi {dur_col} adalah 0 detik.")

        # Web analytics relationship: BounceRates <= ExitRates
        if "BounceRates" in cf_row.index and "ExitRates" in cf_row.index:
            bounce = float(cf_row["BounceRates"])
            exit_r = float(cf_row["ExitRates"])
            if bounce > (exit_r + 1e-4):
                violations.append(f"Inkoherensi Web: BounceRates ({bounce:.4f}) lebih besar dari ExitRates ({exit_r:.4f}).")

        return len(violations) == 0, violations

    def validate_sparsity(self, original_row: pd.Series, cf_row: pd.Series) -> Tuple[bool, List[str], int]:
        """
        Ensures the number of altered features does not exceed max_changed_features.
        """
        changed = []
        for col in original_row.index:
            if col in cf_row.index and str(original_row[col]) != str(cf_row[col]):
                changed.append(col)
        n_changed = len(changed)
        violations = []
        if n_changed > self.max_changed_features:
            violations.append(f"Sparsity Terlampaui: {n_changed} fitur diubah (maksimal {self.max_changed_features}).")
        return len(violations) == 0, violations, n_changed

    def evaluate_counterfactual(self, original_row: pd.Series, cf_row: pd.Series) -> Dict[str, Any]:
        """
        Full diagnostic evaluation report of a single candidate counterfactual.
        """
        valid_imm, imm_violations = self.validate_immutability(original_row, cf_row)
        valid_rng, rng_violations = self.validate_ranges(cf_row)
        valid_int, int_violations = self.validate_integers(cf_row)
        valid_coh, coh_violations = self.validate_coherence(cf_row)
        valid_spr, spr_violations, n_changed = self.validate_sparsity(original_row, cf_row)

        all_violations = imm_violations + rng_violations + int_violations + coh_violations + spr_violations
        is_fully_valid = valid_imm and valid_rng and valid_int and valid_coh and valid_spr

        # Calculate Actionability score: % of modified features that belong to actionable set
        changed_cols = [
            c for c in original_row.index if c in cf_row.index and str(original_row[c]) != str(cf_row[c])
        ]
        if len(changed_cols) == 0:
            actionability_score = 1.0
        else:
            actionable_changes = sum(1 for c in changed_cols if c in self.actionable_cols or c == "PageValues")
            actionability_score = actionable_changes / len(changed_cols)

        return {
            "is_valid": is_fully_valid,
            "immutable_valid": valid_imm,
            "range_valid": valid_rng,
            "integer_valid": valid_int,
            "coherence_valid": valid_coh,
            "sparsity_valid": valid_spr,
            "n_features_changed": n_changed,
            "actionability_score": float(actionability_score),
            "violations": all_violations
        }
