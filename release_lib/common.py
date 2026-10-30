"""Release format, cropped SceneFun3D indices, and the historical Fun3DU metrics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mAP", "AP50", "AP25", "mAR", "AR50", "AR25", "mIoU")
# Historical Fun3DU thresholds from torch.linspace(0.5, 0.95, 10).
THRESHOLDS = np.array([0.5, 0.550000011920929, 0.6000000238418579,
                       0.6499999761581421, 0.699999988079071,
                       0.75, 0.800000011920929, 0.8500000238418579,
                       0.8999999761581421, 0.949999988079071], dtype=np.float32)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value, *, pretty=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, allow_nan=False,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":"))
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visits_for_split(dataset_root, split):
    with (Path(dataset_root) / "benchmark_file_lists" / f"{split}_set.csv").open() as f:
        return list(dict.fromkeys(row["visit_id"] for row in csv.DictReader(f)))


class Scene:
    """Load GT without loading a laser scan or constructing dense masks per query."""

    def __init__(self, dataset_root, split, visit_id):
        self.visit_id = str(visit_id)
        self.directory = Path(dataset_root) / split / self.visit_id
        self.crop_path = self.directory / f"{visit_id}_crop_mask.npy"
        self.crop = np.load(self.crop_path, allow_pickle=False)
        if self.crop.ndim != 1 or not np.isin(self.crop, [0, 1]).all():
            raise ValueError(f"Invalid binary crop mask: {self.crop_path}")
        self.full_indices = np.flatnonzero(self.crop == 1)
        self.n_points = len(self.full_indices)
        self.descriptions = read_json(self.directory / f"{visit_id}_descriptions.json")["descriptions"]
        self.by_id = {desc["desc_id"]: desc for desc in self.descriptions}
        if len(self.by_id) != len(self.descriptions):
            raise ValueError(f"Duplicate descriptions in {visit_id}")
        annotations = read_json(self.directory / f"{visit_id}_annotations.json")["annotations"]
        self.annotations = {annot["annot_id"]: annot for annot in annotations}
        if len(self.annotations) != len(annotations):
            raise ValueError(f"Duplicate annotations in {visit_id}")

    def gt_indices(self, desc_id):
        chunks = []
        for annot_id in self.by_id[desc_id]["annot_id"]:
            # Some original descriptions reference absent annotation IDs. The
            # dense parser iterates existing annotations, so those contribute
            # no points; retain that behavior for exact benchmark parity.
            annot = self.annotations.get(annot_id)
            if annot is None or annot["label"] == "exclude":
                continue
            full = np.asarray(annot["indices"], dtype=np.int64)
            if full.size and (full.min() < 0 or full.max() >= len(self.crop)):
                raise ValueError(f"GT indices out of bounds: {self.visit_id}/{desc_id}")
            full = full[self.crop[full] == 1]
            chunks.append(np.searchsorted(self.full_indices, full))
        return np.unique(np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.int64)

    def metadata(self):
        return {"n_full_points": len(self.crop), "n_points": self.n_points,
                "crop_sha256": sha256(self.crop_path),
                "descriptions_sha256": sha256(self.directory / f"{self.visit_id}_descriptions.json"),
                "annotations_sha256": sha256(self.directory / f"{self.visit_id}_annotations.json"),
                "desc_ids": sorted(self.by_id)}


def validate_indices(values, n_points, context="prediction"):
    if not isinstance(values, list) or any(type(x) is not int for x in values):
        raise ValueError(f"{context}: pred_mask must be a list of integers")
    indices = np.asarray(values, dtype=np.int64)
    if indices.size and (indices[0] < 0 or indices[-1] >= n_points
                         or np.any(indices[1:] <= indices[:-1])):
        raise ValueError(f"{context}: indices must be sorted, unique and in [0, {n_points})")
    return indices


def read_predictions(path, scene_meta):
    entries = read_json(path)
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list in {path}")
    predictions = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"desc_id", "pred_mask"}:
            raise ValueError(f"Unexpected record fields in {path}")
        desc_id = entry["desc_id"]
        if not isinstance(desc_id, str) or desc_id in predictions:
            raise ValueError(f"Duplicate or invalid description in {path}")
        predictions[desc_id] = validate_indices(entry["pred_mask"], scene_meta["n_points"], str(path))
    if set(predictions) != set(scene_meta["desc_ids"]):
        raise ValueError(f"Description coverage mismatch in {path}")
    return predictions



def scores_for_indices(gt, pred):
    intersection = np.intersect1d(gt, pred, assume_unique=True).size
    # torch division of int64 counts produces float32, while the historical
    # dense GT mask is float64 and therefore IoU is float64.
    precision = np.float32(intersection / len(pred)) if len(pred) else np.float32(0)
    recall = np.float32(intersection / len(gt)) if len(gt) else np.float32(0)
    union = len(gt) + len(pred) - intersection
    return {"pred_count": len(pred), "Prc": float(precision),
            "mAP": float(np.mean(precision >= THRESHOLDS, dtype=np.float32)),
            "AP50": float(precision >= np.float32(.5)),
            "AP25": float(precision >= np.float32(.25)),
            "Rec": float(recall),
            "mAR": float(np.mean(recall >= THRESHOLDS, dtype=np.float32)),
            "AR50": float(recall >= np.float32(.5)),
            "AR25": float(recall >= np.float32(.25)),
            "mIoU": intersection / union if union else 0.0}


def reference_rows(reference):
    keys = list(zip(reference["visit_id"], reference["desc_id"]))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate reference query IDs")
    if any(len(v) != len(keys) for v in reference.values()):
        raise ValueError("Inconsistent reference lengths")
    return {key: {name: values[i] for name, values in reference.items()
                  if name not in {"visit_id", "desc_id"}} for i, key in enumerate(keys)}


def compare_row(actual, expected, context):
    if set(actual) != set(expected):
        raise ValueError(f"Metric schema mismatch: {context}")
    for metric, value in actual.items():
        # Same counts and discrete scores; tolerate only 1e-7 for float scores.
        tolerance = 0 if metric in {"pred_count", "mAP", "AP50", "AP25", "mAR", "AR50", "AR25"} else 1e-7
        if not np.isfinite(value) or abs(value - expected[metric]) > tolerance:
            raise ValueError(f"{context}: {metric}: {value} != {expected[metric]}")
