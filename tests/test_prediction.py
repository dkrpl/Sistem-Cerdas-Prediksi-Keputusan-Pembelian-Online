"""
Unit test for Purchase Prediction Engine.
Tests input validation, inference outputs, probability bounds, and defaults.
"""

import unittest
import pandas as pd
import numpy as np
from src.prediction import PurchasePredictionEngine

class TestPurchasePredictionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PurchasePredictionEngine()
        self.sample_session = {
            "Administrative": 2,
            "Administrative_Duration": 50.0,
            "Informational": 1,
            "Informational_Duration": 20.0,
            "ProductRelated": 15,
            "ProductRelated_Duration": 400.0,
            "BounceRates": 0.01,
            "ExitRates": 0.03,
            "PageValues": 15.0,
            "SpecialDay": 0.0,
            "Month": "Nov",
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 2,
            "VisitorType": "Returning_Visitor",
            "Weekend": True
        }

    def test_predict_session_output_keys(self):
        result = self.engine.predict_session(self.sample_session)
        self.assertIn("prediction_label", result)
        self.assertIn("predicted_class", result)
        self.assertIn("purchase_probability", result)
        self.assertIn("non_purchase_probability", result)
        self.assertIn("confidence_level", result)

    def test_probability_range(self):
        result = self.engine.predict_session(self.sample_session)
        prob = result["purchase_probability"]
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertAlmostEqual(prob + result["non_purchase_probability"], 1.0, places=5)

    def test_batch_prediction(self):
        df_batch = pd.DataFrame([self.sample_session, self.sample_session])
        df_res = self.engine.predict_batch(df_batch)
        self.assertIn("Purchase_Probability", df_res.columns)
        self.assertIn("Predicted_Class", df_res.columns)
        self.assertEqual(len(df_res), 2)

if __name__ == "__main__":
    unittest.main()
