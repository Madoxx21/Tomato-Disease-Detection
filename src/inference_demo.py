"""Lab 05 input assurance demo using the unchanged Lab 04 tomato estimator.

Validation checks request metadata. It does not prove that a file exists, decode
its bytes, inspect the decoded pixels, or validate the final model tensor.
"""
import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "validation_demo.jsonl"
MODEL_VERSION = "tomato-rgb-hist-logreg-v1"
RULE_VERSION = "tomato-input-contract-v1"
PREPROCESSING_ID = "tomato-rgb-resize64-bilinear-hist16x3-v1"
REVIEW_ACTION = "human review required; use a controlled single-leaf color photo or consult an expert"
REJECT_ACTION = "input rejected; provide a JPEG or PNG with declared dimensions at least 32x32"
LABELS = (
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___healthy",
)


def image_features(path):
    """Lab 02 deterministic inference transform: RGB, 64x64, 16-bin histograms."""
    with Image.open(path) as image:
        if image.format not in ("JPEG", "PNG") or image.mode != "RGB" or min(image.size) < 32:
            raise ValueError("Expected readable RGB JPEG/PNG at least 32x32 pixels.")
        pixels = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR))
    return np.concatenate([
        np.histogram(pixels[:, :, channel], bins=16, range=(0, 256))[0] / (64 * 64)
        for channel in range(3)
    ]).reshape(1, -1)


def load_baseline_model():
    """Rebuild the fixed Lab 02 estimator on its training partition only."""
    from baseline_pipeline import LABELS as BASELINE_LABELS, load_image, select_sample
    records = select_sample()
    train_records = [r for r in records if r["split"] == "train"]
    X = np.asarray([load_image(r)[0] for r in train_records])
    y = np.asarray([r["label"] for r in train_records])
    if len(X) != 240 or set(y) != set(BASELINE_LABELS):
        raise RuntimeError("Baseline training sample incomplete; run src/baseline_pipeline.py first.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
    model.fit(X, y)
    return model


def project_predict(input_ref, model):
    """Existing Lab 04 prediction behavior; called only after validation passes."""
    features = image_features(input_ref)
    prediction = str(model.predict(features)[0])
    return {
        "suggested_category": prediction,
        "preliminary": True,
        "message": (
            "Preliminary suggestion only. A healthy suggestion does not rule out disease. "
            "Unsupported conditions may be present."
        ),
    }


def predict_from_request(request, model):
    return project_predict(request["input_ref"], model)


def _safe_ref(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return Path(path).name


def _request_metadata(input_path, source, overrides=None):
    # Metadata is measured from the supplied sample for the demo. The validator
    # itself remains request-level and does not open/decode the reference.
    with Image.open(input_path) as image:
        width, height = image.size
        fmt = image.format
    request = {
        "input_ref": _safe_ref(input_path),
        "format": fmt,
        "width": width,
        "height": height,
        "source": source,
        "requires_review": False,
    }
    if overrides:
        request.update(overrides)
    return request


def route_request(request, model, validator):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    validation = validator(request)
    prediction_ran = False
    result = None

    if validation["overall_status"] == "pass":
        try:
            result = predict_from_request(request, model)
            prediction_ran = True
            action = "human review required; inspect plant or consult an expert before treatment"
            outcome = "prediction_completed"
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            # A request-level pass is not proof that the file exists or decodes.
            validation["overall_status"] = "error"
            validation["action"] = "reject_input"
            validation["prediction_allowed"] = False
            validation["checks"].append({
                "rule": "runtime_file_read",
                "observed": str(exc),
                "status": "error",
                "action": "reject_input",
                "explanation": "Runtime image decoding failed after request-level validation."
            })
            action = f"{REJECT_ACTION}: {exc}"
            outcome = "prediction_not_completed"
    elif validation["overall_status"] == "warning":
        action = REVIEW_ACTION
        outcome = "prediction_skipped_for_review"
    else:
        action = REJECT_ACTION
        outcome = "prediction_blocked"

    return {
        "request_id": request_id,
        "input_ref": request.get("input_ref"),
        "validation": validation,
        "prediction": {
            "ran": prediction_ran,
            "outcome": outcome,
            "result": result,
        },
        "model_version": MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "preprocessing_id": PREPROCESSING_ID,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "action": action,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ref", default="data/inference_demo_sample/leaf.JPG")
    args = parser.parse_args()
    input_path = Path(args.input_ref)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    # Import validator after script path is established.
    from input_validation import validate_request

    # Same fixed Lab 04 model, reconstructed from the fixed training partition.
    model = load_baseline_model()

    normal_request = _request_metadata(input_path, "plantvillage_controlled")
    warning_request = _request_metadata(input_path, "grower_phone_photo")
    # Deliberately invalid declared metadata: error must outrank the source warning.
    error_request = _request_metadata(input_path, "grower_phone_photo", {"format": "GIF"})

    records = [
        route_request(normal_request, model, validate_request),
        route_request(warning_request, model, validate_request),
        route_request(error_request, model, validate_request),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({
        "statuses": [r["validation"]["overall_status"] for r in records],
        "prediction_ran": [r["prediction"]["ran"] for r in records],
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
