import os.path as osp
import numpy as np
import pathlib
from typing import Dict, Optional, Union
from glob import glob
import cv2
from dataclasses import dataclass
import pandas as pd


@dataclass
class Results:
    true_positive: Optional[int] = None
    false_positive: Optional[int] = None
    false_negative: Optional[int] = None

    @property
    def recall(self) -> Optional[float]:
        if self.valid and (self.true_positive + self.false_negative) > 0:
            return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def precision(self) -> Optional[float]:
        if self.valid and (self.true_positive + self.false_positive) > 0:
            return self.true_positive / (self.true_positive + self.false_positive)

    @property
    def iou(self) -> Optional[float]:
        if (
            self.valid
            and (self.true_positive + self.false_positive + self.false_negative) > 0
        ):
            return self.true_positive / (
                self.true_positive + self.false_positive + self.false_negative
            )

    @property
    def valid(self) -> bool:
        resp = True
        resp = resp and self.true_positive is not None
        resp = resp and self.false_negative is not None
        resp = resp and self.false_positive is not None

        return resp

    @property
    def dict(self) -> Dict[str, Optional[float]]:

        return {
            "recall": round(self.recall, 2) if self.recall is not None else self.recall,
            "precision": round(self.precision, 2)
            if self.precision is not None
            else self.precision,
            "IoU": round(self.iou, 2) if self.iou is not None else self.iou,
        }


def evaluate_results(
    predicted_masks_dir: Union[str, pathlib.PosixPath],
    ground_truth_mask_dir: Union[str, pathlib.PosixPath],
    output_dir: Union[str, pathlib.PosixPath],
    output_file: str = "test",
):
    out_csv = osp.join(output_dir, output_file + ".csv")
    data_csv: Dict[str, Dict[str, float]] = {}
    gt_files = glob(osp.join(ground_truth_mask_dir, "*.jpg"))
    for gt_file in gt_files:
        file_name = osp.basename(gt_file)
        pred_file = osp.join(predicted_masks_dir, file_name)
        subject = file_name.split("_")[1]
        if osp.isfile(pred_file):
            pred_mask = cv2.imread(pred_file)
            gt_mask = cv2.imread(gt_file)
            data_csv[subject] = calculate_metrics(pred_mask, gt_mask).dict

    df = pd.DataFrame.from_dict(data_csv, orient="index")
    mean_dict = {
        "mean": {
            "recall": round(df["recall"].mean(), 2),
            "precision": round(df["precision"].mean(), 2),
            "IoU": round(df["IoU"].mean(), 2),
        }
    }
    mean_df = pd.DataFrame.from_dict(mean_dict, orient="index")

    df = pd.concat([df, mean_df])
    df.to_csv(out_csv)

def calculate_metrics(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Results:
    true_positive = np.count_nonzero((mask_pred == 255) & (mask_pred == mask_gt))
    false_positive = np.count_nonzero((mask_pred == 255) & (mask_pred != mask_gt))
    false_negative = np.count_nonzero((mask_pred == 0) & (mask_pred != mask_gt))
    return Results(true_positive, false_positive, false_negative)


if __name__ == "__main__":
    pred_dir = "out/mask"
    gt_dir = "gt/annotations"
    output_dir = "out"
    evaluate_results(pred_dir, gt_dir, output_dir)
