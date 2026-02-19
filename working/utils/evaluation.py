"""Utility helpers for basic binary classification evaluation."""

import numpy as np
import pandas as pd
from typing import Mapping, Sequence
from sklearn.metrics import confusion_matrix    


COMMAND_TO_RUN = \
"""
python utils/evaluation.py \
--file_benchmark_on_datawithredundantassumption "data/Task1_PRWITHRA_deepseek-chat.xlsx" \
--file_benchmark_on_datawithoutredundantassumption "data/Task2_PRWITHOUTRA_deepseek-chat.xlsx" \
--task "classification"

python utils/evaluation.py --file_benchmark_on_datawithredundantassumption "data/Task1_PRWITHRA_deepseek-chat.xlsx" --file_benchmark_on_datawithoutredundantassumption "data/Task2_PRWITHOUTRA_deepseek-chat.xlsx" --task "detection"

python utils/evaluation.py \
--file_benchmark_on_datawithredundantassumption "data/Task1_PRWITHRA_gemini-25-flash.xlsx" \
--file_benchmark_on_datawithoutredundantassumption "data/Task2_PRWITHOUTRA_gemini-25-flash.xlsx" \
--task "detection"


python utils/evaluation.py \
--file_benchmark_on_datawithredundantassumption "data/Task1_PRWITHRA_deepseek-reasoner.xlsx" \
--file_benchmark_on_datawithoutredundantassumption "data/Task2_PRWITHOUTRA_deepseek-reasoner.xlsx" \
--task "detection"


python utils/evaluation.py \
--file_benchmark_on_datawithredundantassumption "data/Task1_PRWITHRA_Qwen3-Next-80B-A3B-Instruct.xlsx" \
--file_benchmark_on_datawithoutredundantassumption "data/Task2_PRWITHOUTRA_Qwen3-Next-80B-A3B-Instruct.xlsx" \
--task "detection"


python utils/evaluation.py \
--file_benchmark_on_datawithredundantassumption "data/Task1_PRWITHRA_Pipeline1.xlsx" \
--file_benchmark_on_datawithoutredundantassumption "data/Task2_PRWITHOUTRA_Pipeline1.xlsx" \
--task "detection"
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
    for a, b, proof_review in zip(data['Groundtruth_redundant_assumption_number'], data['llm_ordinal_number_of_redundant_assumption'], data['llm_answer_proof_review']):
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
    for yes_no, ordinal_number, proof_review in zip(data['llm_answer_yesno_redundant_assumption'], data['llm_ordinal_number_of_redundant_assumption'], data['llm_answer_proof_review']):
        if "yes" in str(yes_no).lower() and "no" not in str(yes_no).lower():
            yesno_list.append(1)
        else:
            yesno_list.append(0)
            review_list.append(0 if bool(proof_review) else 1)

    return binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA([1]*len(yesno_list), yesno_list), binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA([1]*len(review_list), review_list)




if __name__ == "__main__":
    import argparse
    from evaluation_ver2 import RedundantHypothesisEvaluator
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_benchmark_on_datawithredundantassumption", type=str, default="")
    parser.add_argument("--file_benchmark_on_datawithoutredundantassumption", type=str, default="")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "detection"])

    args = parser.parse_args()
    data_with_redundant_assumption = pd.read_excel(args.file_benchmark_on_datawithredundantassumption)
    print(data_with_redundant_assumption.describe())
    data_without_redundant_assumption = pd.read_excel(args.file_benchmark_on_datawithoutredundantassumption)
    print(data_without_redundant_assumption.describe())
    if args.task == "detection":
        evaluator = RedundantHypothesisEvaluator()
        # coverage_score = coverage_metrics(data_with_redundant_assumption, data_without_redundant_assumption)
        for gt, pred, num_hypotheses in zip(data_with_redundant_assumption['Groundtruth_redundant_assumption_number'], data_with_redundant_assumption['llm_ordinal_number_of_redundant_assumption'], data_with_redundant_assumption['Number_of_Assumption']):
            evaluator.add_prediction(gt_label=str(gt), pred_label=str(pred), num_hypotheses=num_hypotheses + 1)
        
        ct = 0
        for gt, pred, num_hypotheses in zip([str(-1)]*len(data_without_redundant_assumption), data_without_redundant_assumption['llm_ordinal_number_of_redundant_assumption'], data_without_redundant_assumption['Number_of_Assumption']):
            evaluator.add_prediction(gt_label='NONE', pred_label='NONE' if str(pred) == '-1' else str(pred), num_hypotheses=num_hypotheses)
        evaluator.compute_metrics()
        evaluator.print_report()
    elif args.task == "classification":
        print("Evaluate for the whole dataset on Problem with redundant assumption")
        evaluation_metrics, evaluation_metrics_reviews = evaluation_metrics_for_PROBLEM_WITH_RA(data_with_redundant_assumption)
        print_evaluation_metrics(evaluation_metrics)
        print("==================END_FOR_WHOLE_DATASET=================")
        print("\n")
        print("Evaluate for the whole dataset on Problem without redundant assumption")
        evaluation_metrics, evaluation_metrics_review = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data_without_redundant_assumption)
        print_evaluation_metrics(evaluation_metrics)
        print("==================END_FOR_WHOLE_DATASET=================")
    else:
        raise ValueError(f"Invalid task: {args.task}")
    