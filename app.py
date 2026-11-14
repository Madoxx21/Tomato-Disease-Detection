
"""FastAPI service for the Tomato Disease Detection Lab 06 deployment."""

from __future__ import annotations

import io
import json
import threading
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "reports" / "service_requests.jsonl"
MODEL_VERSION = "tomato-rgb-hist-logreg-v1"
SERVICE_NAME = "tomato-disease-detection-api"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_SIDE = 32

app = FastAPI(
    title="Tomato Disease Detection API",
    description=(
        "A classroom inference service using the team's fixed RGB-histogram "
        "Logistic Regression baseline. Predictions are preliminary and require "
        "human review."
    ),
    version="1.0.0",
)

_model = None
_model_lock = threading.Lock()
_log_lock = threading.Lock()


class InputValidationError(Exception):
    """An input error that must be rejected before model inference."""

    def __init__(self, message: str, checks: dict[str, Any]):
        super().__init__(message)
        self.message = message
        self.checks = checks


def _write_jsonl(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with _log_lock:
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _request_id(value: str | None) -> str:
    value = (value or "").strip()
    return value[:100] if value else str(uuid.uuid4())


def _safe_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    return Path(filename.replace("\\", "/")).name[:150]


def _validate_and_extract(content: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate a JPEG/PNG RGB image, then extract the Lab 02 features."""

    checks: dict[str, Any] = {
        "non_empty": bool(content),
        "within_size_limit": len(content) <= MAX_IMAGE_BYTES,
        "readable_image": False,
        "supported_format": False,
        "rgb_mode": False,
        "minimum_dimensions": False,
    }

    if not content:
        raise InputValidationError("The uploaded file is empty.", checks)

    if len(content) > MAX_IMAGE_BYTES:
        raise InputValidationError("Image exceeds the 10 MB limit.", checks)

    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()

            checks["readable_image"] = True
            checks["supported_format"] = image.format in ("JPEG", "PNG")
            checks["rgb_mode"] = image.mode == "RGB"
            checks["minimum_dimensions"] = (
                image.width >= MIN_IMAGE_SIDE
                and image.height >= MIN_IMAGE_SIDE
            )

            checks["format"] = image.format
            checks["mode"] = image.mode
            checks["width"] = image.width
            checks["height"] = image.height

            if not checks["supported_format"]:
                raise InputValidationError(
                    "Only JPEG and PNG images are supported.", checks
                )

            if not checks["rgb_mode"]:
                raise InputValidationError(
                    "The image must use RGB color mode. Convert it to RGB and retry.",
                    checks,
                )

            if not checks["minimum_dimensions"]:
                raise InputValidationError(
                    "Image dimensions must be at least 32 by 32 pixels.",
                    checks,
                )

            pixels = np.asarray(
                image.resize((64, 64), Image.Resampling.BILINEAR)
            )

    except InputValidationError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise InputValidationError(
            "The uploaded file is not a readable image.", checks
        ) from exc

    features = np.concatenate(
        [
            np.histogram(
                pixels[:, :, channel], bins=16, range=(0, 256)
            )[0] / (64 * 64)
            for channel in range(3)
        ]
    ).reshape(1, -1)

    checks["feature_vector_valid"] = (
        features.shape == (1, 48)
        and bool(np.isfinite(features).all())
    )

    if not checks["feature_vector_valid"]:
        raise InputValidationError(
            "Image features could not be extracted.", checks
        )

    return features, checks


def load_baseline_model():
    """Rebuild the unchanged Lab 02 estimator using only training records."""

    from src.baseline_pipeline import LABELS, load_image, select_sample

    records = select_sample()
    train_records = [
        record for record in records if record["split"] == "train"
    ]

    loaded = [load_image(record)[0] for record in train_records]
    X = np.asarray(loaded)
    y = np.asarray([record["label"] for record in train_records])

    if len(X) != 240 or set(y) != set(LABELS):
        raise RuntimeError(
            "The fixed baseline training sample is incomplete. "
            "Run src/baseline_pipeline.py once or check VM internet access."
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        ),
    )

    model.fit(X, y)
    return model


def get_model():
    """Load the model once, only after a valid image has passed validation."""

    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_baseline_model()

    return _model


@app.get("/health")
def health():
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "model": MODEL_VERSION,
        "model_loaded": _model is not None,
    }


@app.post("/predict")
def predict(
    file: UploadFile | None = File(default=None),
    request_id: str = Form(default=""),
):
    rid = _request_id(request_id)
    filename = _safe_filename(file.filename) if file is not None else None

    content = (
        file.file.read(MAX_IMAGE_BYTES + 1)
        if file is not None
        else b""
    )

    base_record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": rid,
        "filename": filename,
        "content_type": file.content_type if file is not None else None,
        "file_size_bytes": len(content),
        "model_version": MODEL_VERSION,
        "checks": {},
        "prediction_called": False,
        "prediction": None,
        "latency_ms": None,
    }

    started = time.perf_counter()

    try:
        features, checks = _validate_and_extract(content)

    except InputValidationError as exc:
        base_record.update(
            {
                "status": "error",
                "action": "reject_input",
                "checks": exc.checks,
                "error": exc.message,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
            }
        )

        _write_jsonl(base_record)

        return JSONResponse(
            status_code=400,
            content={
                "request_id": rid,
                "status": "error",
                "action": "reject_input",
                "message": exc.message,
                "checks": exc.checks,
                "prediction_called": False,
            },
        )

    base_record["checks"] = checks

    try:
        model = get_model()
        base_record["prediction_called"] = True

        suggested_category = str(model.predict(features)[0])

        result = {
            "suggested_category": suggested_category,
            "preliminary": True,
            "message": (
                "Preliminary suggestion only. A healthy suggestion does not "
                "rule out disease. Inspect the plant or consult an expert "
                "before deciding on treatment."
            ),
        }

        base_record.update(
            {
                "status": "review",
                "action": "human_review_required",
                "prediction": result,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
            }
        )

        _write_jsonl(base_record)

        return {
            "request_id": rid,
            "status": "review",
            "action": "human_review_required",
            "prediction_called": True,
            "prediction": result,
            "checks": checks,
            "latency_ms": base_record["latency_ms"],
        }

    except Exception as exc:
        base_record.update(
            {
                "status": "error",
                "action": "inference_failed",
                "error": type(exc).__name__,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
            }
        )

        _write_jsonl(base_record)

        return JSONResponse(
            status_code=500,
            content={
                "request_id": rid,
                "status": "error",
                "action": "inference_failed",
                "message": "The model could not process this valid image.",
                "prediction_called": base_record["prediction_called"],
            },
        )
