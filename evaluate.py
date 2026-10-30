#!/usr/bin/env python3
"""Evaluate released sparse predictions against external SceneFun3D GT."""
from __future__ import annotations

import argparse

from release_lib.experiments import EXPERIMENTS
from pathlib import Path

import numpy as np

from release_lib.common import (ROOT, METRICS, Scene, compare_row,
                                read_json, read_predictions, reference_rows,
                                scores_for_indices, visits_for_split, write_json)


def evaluate(key, experiment, dataset_root, check_reference):
    scenes = read_json(ROOT / "datasets" / f"{experiment['split']}.json")
    if set(visits_for_split(dataset_root, experiment["split"])) != set(scenes):
        raise ValueError(f"Split visit coverage mismatch: {key}")
    directory = ROOT / experiment["predictions"]
    if {p.stem for p in directory.glob('*.json')} != set(scenes):
        raise ValueError(f"Prediction file coverage mismatch: {key}")
    reference = reference_rows(read_json(ROOT / experiment["reference_results"]))
    expected_pairs = {(v, d) for v, m in scenes.items() for d in m["desc_ids"]}
    if set(reference) != expected_pairs:
        raise ValueError(f"Reference coverage mismatch: {key}")
    values, empty, max_error = [], 0, 0.0
    for visit_id, meta in scenes.items():
        scene = Scene(dataset_root, experiment["split"], visit_id)
        if scene.metadata() != meta:
            raise ValueError(f"Dataset fingerprint mismatch: {visit_id}")
        predictions = read_predictions(directory / f"{visit_id}.json", meta)
        for desc_id, pred in predictions.items():
            row = scores_for_indices(scene.gt_indices(desc_id), pred)
            if check_reference:
                expected = reference[(visit_id, desc_id)]
                compare_row(row, expected, f"{key}/{visit_id}/{desc_id}")
                max_error = max(max_error, *(abs(row[m] - expected[m]) for m in row))
            values.append(row)
            empty += int(len(pred) == 0)
    means = {m: float(np.mean([v[m] for v in values])) for m in METRICS}
    percentages = {m: f"{means[m] * 100:.2f}" for m in METRICS}
    if check_reference and percentages != experiment["expected_percent"]:
        raise ValueError(f"Published table mismatch: {key}: {percentages}")
    print(f"{key}: " + " & ".join(percentages.values()), flush=True)
    return {"experiment": key, "split": experiment["split"], "visits": len(scenes),
            "queries": len(values), "empty_predictions": empty, "means": means,
            "percent": percentages, "reference_checked": check_reference,
            "max_per_query_error": max_error if check_reference else None,
            "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--experiment", default="all")
    parser.add_argument("--check-reference", action="store_true",
                        help="Fail if any query metric/count or the published table differs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiments = EXPERIMENTS
    if args.experiment != "all" and args.experiment not in experiments:
        parser.error(f"Choose all or {', '.join(experiments)}")
    results = [evaluate(k, v, args.dataset_root, args.check_reference)
               for k, v in experiments.items() if args.experiment in {"all", k}]
    if args.output:
        write_json(args.output, {"experiments": results}, pretty=True)


if __name__ == "__main__":
    main()
