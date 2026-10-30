"""Download a fixed tomato sample, fit one baseline, and save held-out accuracy."""

import hashlib
import io
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import PIL
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plantvillage_sample"
HF_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
IMAGE_REVISION = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
HF_BASE = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{HF_REVISION}/"
IMAGE_BASE = (
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/"
    f"{IMAGE_REVISION}/"
)
SEED = 42
COUNTS = {"train": 30, "test": 10}
LABELS = (
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
)
METADATA_HASHES = {
    "splits/color_train.txt": "c43e0205b321d1c929116f5df1842348ac1e5c4861306e081416785d3c439dbd",
    "splits/color_test.txt": "7a7257dcb5f9456feae6ae81248d29c28f45dc4cd41a00c13a5c5c198c5f2f8b",
    "leaf_grouping/leaf-map.json": "3b4b253683c6911744959a3870a92509b1faa1ee34e9f8585e21d0fe6337a25b",
}
LIMITATION = (
    "Only 80 held-out PlantVillage images (10 per category) across eight categories; "
    "Target Spot and mosaic virus are excluded because leaf mappings are unavailable. Controlled leaf "
    "photographs do not establish performance on grower phone photos, unknown "
    "conditions, or real-world disease prevalence. Color features may learn "
    "background or lighting differences instead of disease symptoms."
)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def download(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def metadata(relative_path):
    path = DATA / "metadata" / relative_path
    content = path.read_bytes() if path.exists() else download(HF_BASE + relative_path)
    if sha256(content) != METADATA_HASHES[relative_path]:
        raise ValueError(f"Metadata checksum mismatch: {path}. Remove it and rerun.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content.decode("utf-8")


def leaf_id(path, leaf_map):
    """Use the upstream lookup, excluding missing or ambiguous mappings."""
    identifier = path.name.rsplit("___", 1)[-1].split("copy", 1)[0]
    identifier = identifier.replace(".jpg", "").replace(".JPG", "")
    matches = [
        item for item in leaf_map.get(identifier.strip().lower(), [])
        if item.startswith(path.parent.name + ":::")
    ]
    return matches[0] if len(matches) == 1 else None


def select_sample():
    leaf_map = json.loads(metadata("leaf_grouping/leaf-map.json"))
    candidates = {}
    for split in COUNTS:
        candidates[split] = {}
        for line in metadata(f"splits/color_{split}.txt").splitlines():
            path = PurePosixPath(line.strip())
            if len(path.parts) != 4 or path.parts[:2] != ("raw", "color"):
                raise ValueError(f"Unexpected source path: {line}")
            if path.parent.name not in LABELS:
                continue
            group = leaf_id(path, leaf_map)
            if group:
                candidates[split][str(path)] = group

    # Conservatively exclude any leaf listed in both upstream partitions.
    overlap = set(candidates["train"].values()) & set(candidates["test"].values())
    records = []
    for split, count in COUNTS.items():
        for label in LABELS:
            paths = sorted(
                (p for p, group in candidates[split].items()
                 if PurePosixPath(p).parent.name == label and group not in overlap),
                key=lambda p: (sha256(f"{SEED}:{p}".encode()), p),
            )
            selected_groups = set()
            for path in paths:
                group = candidates[split][path]
                if group in selected_groups:
                    continue
                selected_groups.add(group)
                records.append({"path": path, "label": label, "split": split, "leaf_id": group})
                if len(selected_groups) == count:
                    break
            if len(selected_groups) != count:
                raise ValueError(f"Not enough uniquely mapped leaves for {split}/{label}")
    return records


def load_image(record):
    path = DATA / "images" / record["path"]
    content = path.read_bytes() if path.exists() else download(IMAGE_BASE + quote(record["path"]))
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "JPEG" or image.mode != "RGB" or min(image.size) < 32:
            raise ValueError(f"Expected an RGB JPEG at least 32x32 pixels: {path}")
        pixels = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR))
    # Three normalized 16-bin histograms: 48 features, no learned extractor.
    features = np.concatenate([
        np.histogram(pixels[:, :, channel], bins=16, range=(0, 256))[0] / (64 * 64)
        for channel in range(3)
    ])
    if features.shape != (48,) or not np.isfinite(features).all():
        raise ValueError(f"Invalid features for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return features, sha256(content)


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    print("Selecting 240 training and 80 held-out tomato images...", flush=True)
    records = select_sample()
    print("Loading selected images (first run requires internet)...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(load_image, records))
    features = np.asarray([item[0] for item in loaded])
    labels = np.asarray([record["label"] for record in records])
    train = np.asarray([record["split"] == "train" for record in records])
    groups = [{record["leaf_id"] for record in records if record["split"] == split}
              for split in COUNTS]
    if groups[0] & groups[1]:
        raise ValueError("Leaf overlap between training and evaluation")
    hashes = [set(item[1] for record, item in zip(records, loaded) if record["split"] == split)
              for split in COUNTS]
    if hashes[0] & hashes[1]:
        raise ValueError("Identical image content in training and evaluation")
    for record, (_, digest) in zip(records, loaded):
        record["sha256"] = digest
    manifest = {"hf_revision": HF_REVISION, "image_revision": IMAGE_REVISION, "records": records}
    write_json(DATA / "manifest.json", manifest)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
    model.fit(features[train], labels[train])
    predictions = model.predict(features[~train])
    accuracy = float(accuracy_score(labels[~train], predictions))
    result = {
        "experiment": "tomato_color_histogram_logistic_regression_v1",
        "data": "320 original-color PlantVillage tomato images; 40 per category across 8 categories",
        "evaluation": "80 held-out images, 10 per category; 240 training images, 30 per category; disjoint leaf groups",
        "model": "StandardScaler + logistic regression (C=1.0, max_iter=1000) on 48 RGB histogram features",
        "metric": {"name": "accuracy", "value": accuracy},
        "correct": int(np.sum(predictions == labels[~train])),
        "train_images": int(train.sum()),
        "evaluation_images": int((~train).sum()),
        "labels": list(LABELS),
        "seed": SEED,
        "source": "https://huggingface.co/datasets/mohanty/PlantVillage",
        "hf_revision": HF_REVISION,
        "image_source": "https://github.com/spMohanty/PlantVillage-Dataset",
        "image_revision": IMAGE_REVISION,
        "sample_sha256": sha256(json.dumps(manifest, sort_keys=True).encode()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "Pillow": PIL.__version__, "scikit-learn": sklearn.__version__},
        "limitation": LIMITATION,
    }
    output = ROOT / "reports" / "baseline_metrics.json"
    write_json(output, result)
    print(f"Accuracy: {accuracy:.3f} ({result['correct']}/{result['evaluation_images']}). Saved {output}")


if __name__ == "__main__":
    main()
