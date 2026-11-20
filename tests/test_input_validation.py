"""Unit tests for the Lab 05 input validator and inference gate."""
import unittest
from src.input_validation import validate_request


def valid_request(**updates):
    request = {
        "input_ref": "data/inference_demo_sample/leaf.JPG",
        "format": "JPEG",
        "width": 256,
        "height": 256,
        "source": "plantvillage_controlled",
        "requires_review": False,
    }
    request.update(updates)
    return request


class InputValidationTests(unittest.TestCase):
    def test_error_overrides_warning(self):
        result = validate_request(valid_request(format="GIF", source="grower_phone"))
        self.assertEqual(result["overall_status"], "error")
        self.assertEqual(result["action"], "reject_input")
        self.assertFalse(result["prediction_allowed"])

    def test_warning_does_not_allow_prediction(self):
        result = validate_request(valid_request(source="grower_phone"))
        self.assertEqual(result["overall_status"], "warning")
        self.assertEqual(result["action"], "route_for_review")
        self.assertFalse(result["prediction_allowed"])

    def test_pass_allows_prediction(self):
        result = validate_request(valid_request())
        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["action"], "run_prediction")
        self.assertTrue(result["prediction_allowed"])


if __name__ == "__main__":
    unittest.main()
