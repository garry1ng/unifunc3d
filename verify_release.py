#!/usr/bin/env python3
"""Check file hashes, prediction schemas, split coverage and index bounds."""
import argparse

from release_lib.experiments import EXPERIMENTS

from release_lib.common import ROOT, read_json, read_predictions, sha256


def verify(checksums=True):
    files = read_json(ROOT / "checksums.json") if checksums else {}
    if checksums:
        for relative, expected in files.items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Checksum mismatch: {relative}")
    summary = []
    for key, experiment in EXPERIMENTS.items():
        scenes = read_json(ROOT / "datasets" / f"{experiment['split']}.json")
        directory = ROOT / experiment["predictions"]
        if {p.stem for p in directory.glob("*.json")} != set(scenes):
            raise ValueError(f"Unexpected/missing visit files: {key}")
        count = empty = 0
        for visit_id, meta in scenes.items():
            predictions = read_predictions(directory / f"{visit_id}.json", meta)
            count += len(predictions)
            empty += sum(len(mask) == 0 for mask in predictions.values())
        if len(scenes) != experiment["visits"] or count != experiment["queries"]:
            raise ValueError(f"Query/visit count mismatch: {key}")
        summary.append({"experiment": key, "visits": len(scenes), "queries": count,
                        "empty_predictions": empty})
        print(f"{key}: {len(scenes)} visits, {count} queries, {empty} empty masks; passed")
    if checksums:
        required = {str(p.relative_to(ROOT)) for p in (ROOT / "predictions").rglob("*.json")}
        if not required <= set(files):
            raise ValueError("Some prediction files have no checksum")
        print(f"Verified {len(files)} SHA-256 hashes")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checksums", action="store_true",
                        help="Maintainers only: validate before generating checksums")
    args = parser.parse_args()
    verify(not args.skip_checksums)
