"""Experiment settings for the released UniFunc3D predictions."""

EXPERIMENTS = {
    "8b_val": {
        "split": "val",
        "predictions": "predictions/8b_val",
        "reference_results": "reference_results/8b_val.json",
        "threshold": 0.7,
        "queries": 445,
        "visits": 30,
        "expected_percent": {
            "mAP": "10.65",
            "AP50": "23.82",
            "AP25": "44.04",
            "mAR": "31.98",
            "AR50": "46.07",
            "AR25": "55.51",
            "mIoU": "20.92"
        }
    },
    "30b_val": {
        "split": "val",
        "predictions": "predictions/30b_val",
        "reference_results": "reference_results/30b_val.json",
        "threshold": 0.7,
        "queries": 445,
        "visits": 30,
        "expected_percent": {
            "mAP": "14.47",
            "AP50": "31.24",
            "AP25": "51.01",
            "mAR": "32.76",
            "AR50": "46.97",
            "AR25": "58.88",
            "mIoU": "24.30"
        }
    },
    "8b_train": {
        "split": "train",
        "predictions": "predictions/8b_train",
        "reference_results": "reference_results/8b_train.json",
        "threshold": 0.7,
        "queries": 2598,
        "visits": 200,
        "expected_percent": {
            "mAP": "7.25",
            "AP50": "16.24",
            "AP25": "29.02",
            "mAR": "28.32",
            "AR50": "38.91",
            "AR25": "48.15",
            "mIoU": "14.23"
        }
    },
    "30b_train": {
        "split": "train",
        "predictions": "predictions/30b_train",
        "reference_results": "reference_results/30b_train.json",
        "threshold": 0.7,
        "queries": 2598,
        "visits": 200,
        "expected_percent": {
            "mAP": "10.02",
            "AP50": "21.32",
            "AP25": "35.76",
            "mAR": "27.67",
            "AR50": "40.03",
            "AR25": "51.00",
            "mIoU": "17.09"
        }
    }
}
