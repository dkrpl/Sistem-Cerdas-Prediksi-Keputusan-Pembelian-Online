"""
Unit test for Constraint Validator and Rule Engine.
Tests immutable lock, integer checking, range verification, and physical coherence.
"""

import unittest
import pandas as pd
from src.constraints import ConstraintValidator

class TestConstraintValidator(unittest.TestCase):
    def setUp(self):
        self.validator = ConstraintValidator()
        self.orig = pd.Series({
            "Administrative": 0, "Administrative_Duration": 0.0,
            "Informational": 0, "Informational_Duration": 0.0,
            "ProductRelated": 6, "ProductRelated_Duration": 150.0,
            "BounceRates": 0.02, "ExitRates": 0.04, "PageValues": 0.0,
            "SpecialDay": 0.0, "Month": "May", "OperatingSystems": 2,
            "Browser": 2, "Region": 1, "TrafficType": 2,
            "VisitorType": "Returning_Visitor", "Weekend": False
        })

    def test_immutable_feature_violation(self):
        cf_bad = self.orig.copy()
        cf_bad["Month"] = "Nov"  # Month is immutable
        valid, violations = self.validator.validate_immutability(self.orig, cf_bad)
        self.assertFalse(valid)
        self.assertGreater(len(violations), 0)

    def test_immutable_feature_pass(self):
        cf_good = self.orig.copy()
        cf_good["ProductRelated"] = 12
        cf_good["ProductRelated_Duration"] = 350.0
        valid, violations = self.validator.validate_immutability(self.orig, cf_good)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

    def test_range_validation(self):
        cf_out_of_bounds = self.orig.copy()
        cf_out_of_bounds["BounceRates"] = 1.5  # Max is 1.0
        valid, violations = self.validator.validate_ranges(cf_out_of_bounds)
        self.assertFalse(valid)

    def test_integer_validation(self):
        cf_fractional = self.orig.copy()
        cf_fractional["ProductRelated"] = 8.5  # Must be integer
        valid, violations = self.validator.validate_integers(cf_fractional)
        self.assertFalse(valid)

    def test_coherence_validation(self):
        cf_incoherent = self.orig.copy()
        cf_incoherent["BounceRates"] = 0.50
        cf_incoherent["ExitRates"] = 0.10  # BounceRates cannot exceed ExitRates
        valid, violations = self.validator.validate_coherence(cf_incoherent)
        self.assertFalse(valid)

if __name__ == "__main__":
    unittest.main()
