#!/usr/bin/env python3
"""Regenerate the release's content checksums after an intentional edit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_lib.common import ROOT, sha256, write_json

EXCLUDE_DIRS = {"__pycache__", "outputs", ".git", ".venv"}

if __name__ == "__main__":
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not EXCLUDE_DIRS.intersection(p.relative_to(ROOT).parts)
                   and p.name not in {"checksums.json", ".DS_Store"} and p.suffix not in {".log", ".pyc"})
    write_json(ROOT / "checksums.json", {str(p.relative_to(ROOT)): sha256(p) for p in paths}, pretty=True)
    print(f"Recorded {len(paths)} SHA-256 hashes")
