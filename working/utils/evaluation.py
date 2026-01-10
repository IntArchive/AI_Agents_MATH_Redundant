"""Utility helpers for basic binary classification evaluation."""

import numpy as np
import pandas as pd
from typing import Mapping, Sequence
from sklearn.metrics import confusion_matrix    


COMMAND_TO_RUN = \
"""
python ./utils/evaluation.py \
--data_path "./data/test_for_WITH_RA.xlsx" \
--evaluation_for_problem "Problem_with_redundant_assumption"

python ./utils/evaluation.py \
--data_path "./data/test_for_WITHOUT_RA.xlsx" \
--evaluation_for_problem "Original_Problem_with_numerical_assumption"
"""

def binary_classification_metrics_FOR_PROBLEM_WITH_RA(
    y_true: Sequence[int] | Sequence[bool],
    y_pred: Sequence[int] | Sequence[bool],
) -> Mapping[str, float]:
    """
    Compute confusion counts and derived metrics for binary classification.

    The positive class is assumed to be represented by truthy values (1/True).

    Returns a mapping with keys: TP, FN, FP, TN, accuracy, precision, recall.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    # Ensure deterministic order of labels: 0 = negative, 1 = positive.
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1], normalize=None
    ).ravel()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "TP": tp,
        "FN": fn,
    }

# We need to filter the problem which review as false and true

def evaluation_metrics_for_PROBLEM_WITH_RA(data):
    yesno_list = []
    detect_list = []
    review_list = []
    for a, b, proof_review in zip(data['Groundtruth_redundant_assumption_number'], data['redundant_assumption_number'], data['proof_review']):
        if pd.isna(b):
            yesno_list.append(0)
            detect_list.append(0)
        elif a == b:
            yesno_list.append(1)
            detect_list.append(1)
            review_list.append(1 if proof_review else 0)
        else:
            yesno_list.append(1)
            detect_list.append(0)
            review_list.append(0 if proof_review else 1)

    return binary_classification_metrics_FOR_PROBLEM_WITH_RA([1]*len(yesno_list), yesno_list), binary_classification_metrics_FOR_PROBLEM_WITH_RA([1]*len(review_list), review_list)


def print_evaluation_metrics(evaluation_metrics):
    for name, value in evaluation_metrics.items():
        print(f"{name}: {value}")



########## For Problem without redundant assumption ##########


def binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA(
    y_true: Sequence[int] | Sequence[bool],
    y_pred: Sequence[int] | Sequence[bool],
) -> Mapping[str, float]:
    """
    Compute confusion counts and derived metrics for binary classification.

    The positive class is assumed to be represented by truthy values (1/True).

    Returns a mapping with keys: TP, FN, FP, TN, accuracy, precision, recall.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    # Ensure deterministic order of labels: 0 = negative, 1 = positive.
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1], normalize=None
    ).ravel()

    return {
        "TN": tp,
        "FP": fn,
    }

def evaluation_metrics_for_PROBLEM_WITHOUT_RA(data):
    yesno_list = []
    review_list = []
    for b, proof_review in zip(data['redundant_assumption_number'], data['proof_review']):
        if pd.isna(b):
            yesno_list.append(1)
        else:
            yesno_list.append(0)
            review_list.append(0 if proof_review else 1)

    return binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA([1]*len(yesno_list), yesno_list), binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA([1]*len(review_list), review_list)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument(
        "--evaluation_for_problem",
        type=str,
        default="Problem_with_redundant_assumption",
        choices=[
            "Problem_with_redundant_assumption",
            "Original_Problem_with_numerical_assumption"
        ],
        help="Choose which evaluation type to use: Problem_with_redundant_assumption or Original_Problem_with_numerical_assumption"
    )

    args = parser.parse_args()
    data = pd.read_excel(args.data_path)
    if args.evaluation_for_problem == "Problem_with_redundant_assumption":
        print("Evaluate for the whole dataset on Problem with redundant assumption")
        evaluation_metrics, evaluation_metrics_reviews = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        print_evaluation_metrics(evaluation_metrics)
        print("==================END_FOR_WHOLE_DATASET=================")
        print("\n")
        print("Evaluate for the predicted positive dataset")
        data_predicted_positive = data[data["redundant_assumption_number"].notna()]
        evaluation_metrics, evaluation_metrics_review = evaluation_metrics_for_PROBLEM_WITH_RA(data_predicted_positive)
        print("\n")
        print("The following will print the truely predicted problem with redundant assumption (TP)")
        print("Then calculate the TP_withREVIEW_TRUE and TP_withREVIEW_FALSE")
        print(f"TP_withREVIEW_TRUE = {evaluation_metrics_review["TP"]}")
        print(f"TP_withREVIEW_FALSE = {evaluation_metrics_review["FN"]}")
        print("==================END_FOR_PREDICTED_POSITIVE_DATASET=================")
    elif args.evaluation_for_problem == "Original_Problem_with_numerical_assumption":
        print("Evaluate for the whole dataset on Problem without redundant assumption")
        evaluation_metrics, evaluation_metrics_review = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        print_evaluation_metrics(evaluation_metrics)
        print("==================END_FOR_WHOLE_DATASET=================")
        
        print("\n")

        print("Evaluate for negative proof review true/false")
        data_predicted_positive = data[data["redundant_assumption_number"].notna()]
        evaluation_metrics, evaluation_metrics_review = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data_predicted_positive)
        print_evaluation_metrics(evaluation_metrics)
        print("\n")
        print("The following will print the wrongly predicted problem without redundant assumption (FP)")
        print("Then calculate the FP_withREVIEW_TRUE and FP_withREVIEW_FALSE")
        print(f"FP_withREVIEW_FALSE = {evaluation_metrics_review["TN"]}")
        print(f"FP_withREVIEW_TRUE = {evaluation_metrics_review["FP"]}")
        print("==================END_FOR_PREDICTED_POSITIVE_DATASET=================")
    else:
        raise ValueError(f"Invalid evaluation type: {args.evaluation_for_problem}")
    