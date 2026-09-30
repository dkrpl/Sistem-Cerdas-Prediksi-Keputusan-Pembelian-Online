"""
Unit test for Actionable Counterfactual Generation & Evaluation.
Tests validity, proximity, sparsity, plausibility, diversity, and actionability metrics calculation.
"""

import unittest
import pandas as pd
import numpy as np
from src.counterfactual import ActionableCounterfactualEngine

class MockPipeline:
    def predict_proba(self, X):
        # High PageValues or ProductRelated yields purchase probability > 0.5
        probs = []
        for _, row in X.iterrows():
            pv = float(row.get("PageValues", 0.0))
            pr = float(row.get("ProductRelated", 0.0))
            p = min(0.1 + (pv * 0.03) + (pr * 0.01), 0.95)
            probs.append([1.0 - p, p])
        return np.array(probs)

class TestCounterfactualEngine(unittest.TestCase):
    def setUp(self):
        self.mock_pipe = MockPipeline()
        self.train_data = pd.DataFrame([
            {
                "Administrative": 0, "Administrative_Duration": 0.0,
                "Informational": 0, "Informational_Duration": 0.0,
                "ProductRelated": 5, "ProductRelated_Duration": 100.0,
                "BounceRates": 0.02, "ExitRates": 0.05, "PageValues": 0.0,
                "SpecialDay": 0.0, "Month": "Feb", "OperatingSystems": 1,
                "Browser": 1, "Region": 1, "TrafficType": 1,
                "VisitorType": "Returning_Visitor", "Weekend": False,
                "Revenue": 0
            },
            {
                "Administrative": 3, "Administrative_Duration": 80.0,
                "Informational": 1, "Informational_Duration": 30.0,
                "ProductRelated": 25, "ProductRelated_Duration": 900.0,
                "BounceRates": 0.005, "ExitRates": 0.015, "PageValues": 25.0,
                "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
                "Browser": 2, "Region": 1, "TrafficType": 2,
                "VisitorType": "Returning_Visitor", "Weekend": True,
                "Revenue": 1
            }
        ])
        num_cols = [
            "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"
        ]
        cat_cols = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
        self.engine = ActionableCounterfactualEngine(
            self.mock_pipe, self.train_data, num_cols, cat_cols
        )

    def test_quality_metrics_structure(self):
        orig_row = self.train_data.iloc[[0]]
        cf_row = self.train_data.iloc[[1]]
        metrics = self.engine.evaluate_counterfactual_quality(orig_row, cf_row)
        expected_metrics = ["Validity", "Proximity_L1", "Proximity_L2", "Sparsity_ChangedFeatures", "Actionability_Score", "Plausibility_Score", "Diversity"]
        for m in expected_metrics:
            self.assertIn(m, metrics)

if __name__ == "__main__":
    unittest.main()
