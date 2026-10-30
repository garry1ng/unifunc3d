"""Regression checks for index alignment, historical metrics and release failures."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from release_lib.common import (Scene, read_predictions,
                                scores_for_indices, validate_indices)


class IndexSpaceTests(unittest.TestCase):
    def test_crop_union_and_exclude_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "val" / "123"
            directory.mkdir(parents=True)
            np.save(directory / "123_crop_mask.npy", [0, 1, 1, 0, 1, 1])
            (directory / "123_descriptions.json").write_text(json.dumps({"descriptions": [
                {"desc_id": "task", "annot_id": ["a", "b", "exclude", "absent"]}]}))
            (directory / "123_annotations.json").write_text(json.dumps({"annotations": [
                {"annot_id": "a", "label": "pull", "indices": [0, 1, 4]},
                {"annot_id": "b", "label": "pull", "indices": [4, 5]},
                {"annot_id": "exclude", "label": "exclude", "indices": [2]}]}))
            scene = Scene(temp, "val", "123")
            np.testing.assert_array_equal(scene.gt_indices("task"), [0, 2, 3])
            self.assertEqual(scene.n_points, 4)

    def test_reject_invalid_indices(self):
        for values in [[-1], [5], [2, 1], [1, 1], [True], [1.0]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_indices(values, 5)
        self.assertEqual(len(validate_indices([], 5)), 0)

    def test_missing_extra_or_duplicate_descriptions_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scene.json"
            meta = {"n_points": 5, "desc_ids": ["a", "b"]}
            for entries in [[{"desc_id": "a", "pred_mask": []}],
                            [{"desc_id": "a", "pred_mask": []}] * 2,
                            [{"desc_id": "a", "pred_mask": [], "gt_mask": []}],
                            [{"desc_id": x, "pred_mask": []} for x in ["a", "b", "c"]]]:
                path.write_text(json.dumps(entries))
                with self.assertRaises(ValueError):
                    read_predictions(path, meta)


class MetricTests(unittest.TestCase):
    def test_precision_recall_protocol_is_distinct_from_iou(self):
        row = scores_for_indices(np.arange(4), np.array([0, 1, 7]))
        self.assertEqual(row["AP50"], 1)
        self.assertEqual(row["AR50"], 1)
        self.assertEqual(row["mAP"], float(np.float32(.4)))
        self.assertEqual(row["mAR"], float(np.float32(.1)))
        self.assertEqual(row["mIoU"], .4)

    def test_empty_masks(self):
        empty = np.array([], dtype=np.int64)
        self.assertTrue(all(value == 0 for value in scores_for_indices(empty, empty).values()))
        self.assertEqual(scores_for_indices(np.array([1]), empty)["Rec"], 0)



if __name__ == "__main__":
    unittest.main()
