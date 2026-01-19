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
    for a, b, c in zip(data["llm_answer_yesno_redundant_assumption"], data['Groundtruth_redundant_assumption_number'], data["llm_ordinal_number_of_redundant_assumption"]):
        try:
            b = int(b)
            c = int(c)
        except:
            c = -1
        if a == "yes" and int(b) == int(c):
            yesno_list.append(1)
            detect_list.append(1)
        elif a == "yes" and int(b)!=int(c):
            yesno_list.append(1)
            detect_list.append(0)
        elif a == "no":
            yesno_list.append(0)
            detect_list.append(0)

    return binary_classification_metrics_FOR_PROBLEM_WITH_RA([1]*len(yesno_list), yesno_list), binary_classification_metrics_FOR_PROBLEM_WITH_RA([1]*len(detect_list), detect_list)


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
    for b in zip(data["llm_answer_yesno_redundant_assumption"]):
        if "no" in str(b).lower() and "yes" not in str(b).lower():
            yesno_list.append(1)
        elif "yes" in str(b).lower() and "no" not in str(b).lower():
            yesno_list.append(0)
        elif pd.isna(b[0]):
            
            yesno_list.append(1)
        else:
            raise ValueError(f"Invalid answer: {b}")

    return binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA([1]*len(yesno_list), yesno_list)

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
        evaluation_metrics, evaluation_metrics_detect = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        print_evaluation_metrics(evaluation_metrics)
        print("==================END_FOR_WHOLE_DATASET=================")
        print("\n")
        print("Evaluate for the predicted positive dataset")
        data_predicted_positive = data[data["llm_ordinal_number_of_redundant_assumption"].notna()]
        evaluation_metrics, evaluation_metrics_detect = evaluation_metrics_for_PROBLEM_WITH_RA(data_predicted_positive)
        print("\n")
        print("The following will print the truely predicted problem with redundant assumption (TP)")
        print(f"TP_withDETECT_TRUE = {evaluation_metrics_detect["TP"]}")
        print(f"TP_withDETECT_FALSE = {evaluation_metrics_detect["FN"]}")
        print("==================END_FOR_PREDICTED_POSITIVE_DATASET=================")
    elif args.evaluation_for_problem == "Original_Problem_with_numerical_assumption":
        print("Evaluate for the whole dataset on Problem without redundant assumption")
        evaluation_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        print_evaluation_metrics(evaluation_metrics)
        print("==================END_FOR_WHOLE_DATASET=================")
        
        print("\n")

        
    else:
        raise ValueError(f"Invalid evaluation type: {args.evaluation_for_problem}")
    