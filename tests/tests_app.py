
"""API routing tests. The model is mocked so tests do not download training data."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app as service


class DummyModel:
    def __init__(self):
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        return ["Tomato___Early_blight"]


def make_rgb_png():
    image = Image.new("RGB", (96, 96), (120, 150, 80))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class TomatoApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_log_path = service.LOG_PATH
        service.LOG_PATH = Path(self.temp_dir.name) / "requests.jsonl"
        service._model = None
        self.client = TestClient(service.app)

    def tearDown(self):
        service.LOG_PATH = self.old_log_path
        service._model = None
        self.temp_dir.cleanup()

    def read_records(self):
        return [
            json.loads(line)
            for line in service.LOG_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def test_health_identifies_service_and_lazy_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["service"], service.SERVICE_NAME
        )
        self.assertFalse(response.json()["model_loaded"])

    def test_valid_rgb_png_reaches_prediction_and_is_logged(self):
        dummy = DummyModel()

        with patch.object(service, "get_model", return_value=dummy):
            response = self.client.post(
                "/predict",
                data={"request_id": "lab06-valid-001"},
                files={
                    "file": (
                        "leaf.png",
                        make_rgb_png(),
                        "image/png",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["status"], "review")
        self.assertTrue(body["prediction_called"])
        self.assertEqual(
            body["prediction"]["suggested_category"],
            "Tomato___Early_blight",
        )
        self.assertEqual(dummy.calls, 1)

        record = self.read_records()[0]
        self.assertEqual(record["request_id"], "lab06-valid-001")
        self.assertTrue(record["prediction_called"])
        self.assertEqual(
            record["action"], "human_review_required"
        )

    def test_invalid_text_file_is_rejected_before_prediction(self):
        dummy = DummyModel()

        with patch.object(service, "get_model", return_value=dummy):
            response = self.client.post(
                "/predict",
                data={"request_id": "lab06-invalid-001"},
                files={
                    "file": (
                        "notes.txt",
                        b"this is not an image",
                        "text/plain",
                    )
                },
            )

        self.assertEqual(response.status_code, 400)

        body = response.json()
        self.assertEqual(body["action"], "reject_input")
        self.assertFalse(body["prediction_called"])
        self.assertEqual(dummy.calls, 0)

        record = self.read_records()[0]
        self.assertEqual(record["request_id"], "lab06-invalid-001")
        self.assertFalse(record["prediction_called"])
        self.assertEqual(record["action"], "reject_input")

    def test_missing_file_is_rejected_and_logged(self):
        dummy = DummyModel()

        with patch.object(service, "get_model", return_value=dummy):
            response = self.client.post(
                "/predict",
                data={"request_id": "lab06-missing-001"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["action"], "reject_input"
        )
        self.assertFalse(response.json()["prediction_called"])
        self.assertEqual(dummy.calls, 0)

        record = self.read_records()[0]
        self.assertEqual(record["request_id"], "lab06-missing-001")
        self.assertFalse(record["prediction_called"])

    def test_grayscale_image_is_rejected_before_prediction(self):
        image = Image.new("L", (96, 96), 120)
        output = io.BytesIO()
        image.save(output, format="PNG")

        dummy = DummyModel()

        with patch.object(service, "get_model", return_value=dummy):
            response = self.client.post(
                "/predict",
                data={"request_id": "lab06-grayscale-001"},
                files={
                    "file": (
                        "gray.png",
                        output.getvalue(),
                        "image/png",
                    )
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["prediction_called"])
        self.assertEqual(dummy.calls, 0)


if __name__ == "__main__":
    unittest.main()
