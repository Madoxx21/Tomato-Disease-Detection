"""Lab 04 inference demo for the existing tomato RGB-histogram baseline.

The repository does not contain a serialized estimator. This script reconstructs
the same fixed baseline from the cached, pinned training sample, without tuning.
The first run requires the internet if the baseline data cache is absent.
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
DATA = ROOT / "data" / "plantvillage_sample"
REPORT = ROOT / "reports" / "inference_demo.jsonl"
MODEL_VERSION = "tomato-rgb-hist-logreg-v1"
RULE_VERSION = "lab04-review-all-v1"
NORMAL_ACTION = "inspect plant; retake photo or consult an expert before treatment"
REVIEW_ACTION = "human review required; inspect plant or consult an expert"
UNAVAILABLE_ACTION = "prediction unavailable; check input/data and retry"
LABELS = (
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___healthy",
)

def image_features(path):
    """Apply the baseline's 64x64 resize and three 16-bin RGB histograms."""
    with Image.open(path) as image:
        if image.format not in ("JPEG", "PNG") or image.mode != "RGB" or min(image.size) < 32:
            raise ValueError("Expected a readable RGB JPEG/PNG at least 32x32 pixels.")
        pixels = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR))
    return np.concatenate([
        np.histogram(pixels[:, :, channel], bins=16, range=(0, 256))[0] / (64 * 64)
        for channel in range(3)
    ]).reshape(1, -1)

def load_baseline_model():
    """Fit the unchanged Lab 02 estimator on its fixed training partition."""
    # Importing the baseline pipeline reuses its pinned sample selection,
    # metadata verification, image loading, and exact feature extraction.
    from baseline_pipeline import COUNTS, DATA as BASELINE_DATA, LABELS as BASELINE_LABELS
    from baseline_pipeline import load_image, select_sample

    records = select_sample()
    train_records = [r for r in records if r["split"] == "train"]
    if not (BASELINE_DATA / "manifest.json").exists():
        # The selected records are loaded below; this does not tune the model.
        pass
    loaded = [load_image(r)[0] for r in train_records]
    X = np.asarray(loaded)
    y = np.asarray([r["label"] for r in train_records])
    if len(X) != 240 or set(y) != set(BASELINE_LABELS):
        raise RuntimeError("Baseline training sample is incomplete; run src/baseline_pipeline.py first.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
    model.fit(X, y)
    return model

def safe_input_ref(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return Path(path).name

def write_record(record):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def attempt(input_path, model, simulated=False):
    request_id = str(uuid.uuid4())
    input_ref = safe_input_ref(input_path)
    start = time.perf_counter()
    if simulated:
        status, result, action = "unavailable", None, UNAVAILABLE_ACTION
    else:
        try:
            features = image_features(input_path)
            prediction = str(model.predict(features)[0])
            # Lab 03 has no validated uncertainty threshold. For safety, the
            # demo routes every output to human review instead of pretending
            # an arbitrary confidence cutoff is validated.
            status = "review"
            result = {
                "suggested_category": prediction,
                "preliminary": True,
                "message": (
                    "Preliminary suggestion only. A healthy suggestion does not rule out disease. "
                    "Unsupported conditions may be present."
                ),
            }
            action = REVIEW_ACTION
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            status, result = "unavailable", None
            action = f"input unavailable: {exc} Retake a clear, single-leaf color photo and retry."
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    record = {
        "request_id": request_id,
        "input_ref": input_ref,
        "status": status,
        "result": result,
        "model_version": MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "latency_ms": latency_ms,
        "action": action,
        "simulated": simulated,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_record(record)
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ref", required=True, help="Path to one RGB tomato-leaf photo")
    args = parser.parse_args()
    input_path = Path(args.input_ref)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    # Fit the existing, fixed baseline once for this demo invocation. No
    # hyperparameter search or held-out evaluation data is used for fitting.
    model = load_baseline_model()
    # Start a fresh evidence file so each documented run contains exactly two lines.
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("", encoding="utf-8")
    normal = attempt(input_path, model, simulated=False)
    unavailable = attempt(input_path, model, simulated=True)
    print(json.dumps({"normal_attempt": normal, "simulated_unavailable_attempt": unavailable}, indent=2))

if __name__ == "__main__":
    main()
