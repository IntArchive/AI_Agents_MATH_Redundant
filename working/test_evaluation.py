"""Test suite for evaluation.py utility functions."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

# Import the functions to test
# Assuming evaluation.py is in utils/ directory
try:
    from utils.evaluation import (
        binary_classification_metrics_FOR_PROBLEM_WITH_RA,
        binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA,
        evaluation_metrics_for_PROBLEM_WITH_RA,
        evaluation_metrics_for_PROBLEM_WITHOUT_RA,
        print_evaluation_metrics
    )
except ImportError:
    # If running from the same directory
    from evaluation import (
        binary_classification_metrics_FOR_PROBLEM_WITH_RA,
        binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA,
        evaluation_metrics_for_PROBLEM_WITH_RA,
        evaluation_metrics_for_PROBLEM_WITHOUT_RA,
        print_evaluation_metrics
    )


class TestBinaryClassificationMetricsForProblemWithRA:
    """Test binary_classification_metrics_FOR_PROBLEM_WITH_RA function."""
    
    def test_perfect_predictions(self):
        """Test with all correct predictions."""
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        result = binary_classification_metrics_FOR_PROBLEM_WITH_RA(y_true, y_pred)
        
        assert result["TP"] == 4
        assert result["FN"] == 0
    
    def test_all_false_negatives(self):
        """Test with all false negatives."""
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]
        result = binary_classification_metrics_FOR_PROBLEM_WITH_RA(y_true, y_pred)
        
        assert result["TP"] == 0
        assert result["FN"] == 4
    
    def test_mixed_predictions(self):
        """Test with mixed predictions."""
        y_true = [1, 1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 1]
        result = binary_classification_metrics_FOR_PROBLEM_WITH_RA(y_true, y_pred)
        
        assert result["TP"] == 2
        assert result["FN"] == 2
    
    def test_boolean_inputs(self):
        """Test with boolean inputs instead of integers."""
        y_true = [True, True, False, False]
        y_pred = [True, False, False, True]
        result = binary_classification_metrics_FOR_PROBLEM_WITH_RA(y_true, y_pred)
        
        assert result["TP"] == 1
        assert result["FN"] == 1
    
    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = [1, 1, 1]
        y_pred = [1, 1]
        
        with pytest.raises(ValueError, match="y_true and y_pred must be the same length"):
            binary_classification_metrics_FOR_PROBLEM_WITH_RA(y_true, y_pred)
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        y_true = []
        y_pred = []
        result = binary_classification_metrics_FOR_PROBLEM_WITH_RA(y_true, y_pred)
        
        # With empty inputs, confusion matrix should handle gracefully
        assert "TP" in result
        assert "FN" in result


class TestBinaryClassificationMetricsForProblemWithoutRA:
    """Test binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA function."""
    
    def test_perfect_predictions(self):
        """Test with all correct predictions."""
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        result = binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA(y_true, y_pred)
        
        assert result["TN"] == 4
        assert result["FP"] == 0
    
    def test_all_false_positives(self):
        """Test with all false positives."""
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]
        result = binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA(y_true, y_pred)
        
        assert result["TN"] == 0
        assert result["FP"] == 4
    
    def test_mixed_predictions(self):
        """Test with mixed predictions."""
        y_true = [1, 1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 1]
        result = binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA(y_true, y_pred)
        
        assert result["TN"] == 2
        assert result["FP"] == 2
    
    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = [1, 1, 1]
        y_pred = [1, 1]
        
        with pytest.raises(ValueError, match="y_true and y_pred must be the same length"):
            binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA(y_true, y_pred)


class TestEvaluationMetricsForProblemWithRA:
    """Test evaluation_metrics_for_PROBLEM_WITH_RA function."""
    
    def test_all_correct_detections(self):
        """Test when all detections are correct."""
        data = pd.DataFrame({
            "Groundtruth_redundant_assumption_number": [1, 2, 3],
            "llm_answer_ordinal_number_of_redundant_assumption": [1, 2, 3],
            "llm_answer_proof_review": [True, True, True]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # All groundtruth == predicted, so yesno_list = [1,1,1], all TP
        assert yesno_metrics["TP"] == 3
        assert yesno_metrics["FN"] == 0
        # review_list = [1,1,1] (proof_review is True), all TP
        assert review_metrics["TP"] == 3
        assert review_metrics["FN"] == 0
    
    def test_incorrect_detections(self):
        """Test when detections are incorrect."""
        data = pd.DataFrame({
            "Groundtruth_redundant_assumption_number": [1, 2, 3],
            "llm_answer_ordinal_number_of_redundant_assumption": [2, 3, 1],
            "llm_answer_proof_review": [True, False, True]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # All groundtruth != predicted, so yesno_list = [1,1,1] (still yes), all TP
        assert yesno_metrics["TP"] == 3
        assert yesno_metrics["FN"] == 0
        # review_list = [0,1,0] (when proof_review is True, review=0; when False, review=1)
        # So review_list = [0,1,0], comparing with [1,1,1]: TP=1 (index 1), FN=2
        assert review_metrics["TP"] == 1
        assert review_metrics["FN"] == 2
    
    def test_mixed_correct_incorrect_detections(self):
        """Test with mixed correct and incorrect detections."""
        data = pd.DataFrame({
            "Groundtruth_redundant_assumption_number": [1, 2, 3, 4],
            "llm_answer_ordinal_number_of_redundant_assumption": [1, 2, 4, 5],
            "llm_answer_proof_review": [True, False, True, False]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # yesno_list = [1,1,1,1] (all yes), all TP
        assert yesno_metrics["TP"] == 4
        assert yesno_metrics["FN"] == 0
        # review_list = [1,0,0,1] (when a==b and proof_review=True: 1; when a==b and proof_review=False: 0; when a!=b and proof_review=True: 0; when a!=b and proof_review=False: 1)
        # Comparing [1,1,1,1] with [1,0,0,1]: TP=2 (indices 0,3), FN=2 (indices 1,2)
        assert review_metrics["TP"] == 2
        assert review_metrics["FN"] == 2
    
    def test_with_nan_values(self):
        """Test with NaN values in ordinal number."""
        data = pd.DataFrame({
            "Groundtruth_redundant_assumption_number": [1, 2, 3],
            "llm_answer_ordinal_number_of_redundant_assumption": [1, pd.NA, 3],
            "llm_answer_proof_review": [True, True, True]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # yesno_list = [1,0,1] (NaN becomes 0), comparing with [1,1,1]: TP=2, FN=1
        assert yesno_metrics["TP"] == 2
        assert yesno_metrics["FN"] == 1
        # review_list only includes entries where a==b, so [1] (only index 0 and 2, but index 1 is NaN so not added)
        # Actually wait, let me re-read the logic...
        # If pd.isna(b): yesno=0, detect=0 (no review added)
        # If a==b: yesno=1, detect=1, review=1 if proof_review else 0
        # Else: yesno=1, detect=0, review=0 if proof_review else 1
        # So review_list = [1, 1] (indices 0 and 2), comparing with [1,1]: TP=2, FN=0
        assert review_metrics["TP"] == 2
        assert review_metrics["FN"] == 0
    
    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        data = pd.DataFrame({
            "Groundtruth_redundant_assumption_number": [],
            "llm_answer_ordinal_number_of_redundant_assumption": [],
            "llm_answer_proof_review": []
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # Should handle empty data gracefully
        assert "TP" in yesno_metrics
        assert "FN" in yesno_metrics
        assert "TP" in review_metrics
        assert "FN" in review_metrics


class TestEvaluationMetricsForProblemWithoutRA:
    """Test evaluation_metrics_for_PROBLEM_WITHOUT_RA function."""
    
    def test_all_no_answers(self):
        """Test when all answers are 'no'."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["no", "no", "no"],
            "llm_answer_ordinal_number_of_redundant_assumption": [None, None, None],
            "llm_answer_proof_review": [True, False, True]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # yesno_list = [0,0,0] (all "no"), comparing with [1,1,1]: TP=0, FN=3
        # But wait, binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA returns TN=tp, FP=fn
        # So TN=0, FP=3
        assert yesno_metrics["TN"] == 0
        assert yesno_metrics["FP"] == 3
        # review_list = [0,1,0] (when yesno=0, review=0 if proof_review else 1)
        # Comparing [1,1,1] with [0,1,0]: TP=1, FN=2, so TN=1, FP=2
        assert review_metrics["TN"] == 1
        assert review_metrics["FP"] == 2
    
    def test_all_yes_answers(self):
        """Test when all answers are 'yes'."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes", "yes", "yes"],
            "llm_answer_ordinal_number_of_redundant_assumption": [1, 2, 3],
            "llm_answer_proof_review": [True, False, True]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # yesno_list = [1,1,1] (all "yes"), comparing with [1,1,1]: TP=3, FN=0
        # So TN=3, FP=0
        assert yesno_metrics["TN"] == 3
        assert yesno_metrics["FP"] == 0
        # review_list is empty (only populated when yesno=0)
        # So comparing [1,1,1] with []: this would cause an error... wait, let me check
        # Actually review_list is only appended when yesno=0, so if all yesno=1, review_list=[]
        # Comparing [1]*0 with []: TP=0, FN=0, so TN=0, FP=0
        assert review_metrics["TN"] == 0
        assert review_metrics["FP"] == 0
    
    def test_mixed_answers(self):
        """Test with mixed yes and no answers."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["no", "yes", "no", "yes"],
            "llm_answer_ordinal_number_of_redundant_assumption": [None, 1, None, 2],
            "llm_answer_proof_review": [True, False, False, True]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # yesno_list = [0,1,0,1], comparing with [1,1,1,1]: TP=2, FN=2, so TN=2, FP=2
        assert yesno_metrics["TN"] == 2
        assert yesno_metrics["FP"] == 2
        # review_list = [0,1] (indices 0 and 2, where yesno=0)
        # review_list[0] = 0 (proof_review=True), review_list[1] = 1 (proof_review=False)
        # Comparing [1,1] with [0,1]: TP=1, FN=1, so TN=1, FP=1
        assert review_metrics["TN"] == 1
        assert review_metrics["FP"] == 1
    
    def test_case_insensitive(self):
        """Test that the function handles different cases."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["No", "YES", "nO", "Yes"],
            "llm_answer_ordinal_number_of_redundant_assumption": [None, 1, None, 2],
            "llm_answer_proof_review": [True, False, True, False]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # yesno_list = [0,1,0,1], comparing with [1,1,1,1]: TP=2, FN=2, so TN=2, FP=2
        assert yesno_metrics["TN"] == 2
        assert yesno_metrics["FP"] == 2
        # review_list = [0,1] (indices 0 and 2, where yesno=0)
        # review_list[0] = 0 (proof_review=True), review_list[1] = 1 (proof_review=False)
        # Comparing [1,1] with [0,1]: TP=1, FN=1, so TN=1, FP=1
        assert review_metrics["TN"] == 1
        assert review_metrics["FP"] == 1
    
    def test_ambiguous_answer_with_no(self):
        """Test that answers containing both yes and no are treated as no."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes and no", "no"],
            "llm_answer_ordinal_number_of_redundant_assumption": [None, None],
            "llm_answer_proof_review": [True, False]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # "yes and no" contains "no", so yesno_list = [0,0]
        # Comparing [1,1] with [0,0]: TP=0, FN=2, so TN=0, FP=2
        assert yesno_metrics["TN"] == 0
        assert yesno_metrics["FP"] == 2
        # review_list = [0,1] (proof_review=True gives 0, proof_review=False gives 1)
        # Comparing [1,1] with [0,1]: TP=1, FN=1, so TN=1, FP=1
        assert review_metrics["TN"] == 1
        assert review_metrics["FP"] == 1
    
    def test_string_with_yes_but_not_no(self):
        """Test that strings containing yes but not no are treated as yes."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes, definitely", "maybe yes"],
            "llm_answer_ordinal_number_of_redundant_assumption": [1, 2],
            "llm_answer_proof_review": [True, False]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # Both contain "yes" and not "no", so yesno_list = [1,1]
        # Comparing [1,1] with [1,1]: TP=2, FN=0, so TN=2, FP=0
        assert yesno_metrics["TN"] == 2
        assert yesno_metrics["FP"] == 0
        # review_list is empty (only populated when yesno=0)
        # Comparing [1]*0 with []: TP=0, FN=0, so TN=0, FP=0
        assert review_metrics["TN"] == 0
        assert review_metrics["FP"] == 0


class TestPrintEvaluationMetrics:
    """Test print_evaluation_metrics function."""
    
    def test_print_metrics(self, capsys):
        """Test that metrics are printed correctly."""
        metrics = {
            "TP": 10,
            "FN": 5,
            "TN": 20,
            "FP": 3
        }
        
        print_evaluation_metrics(metrics)
        
        captured = capsys.readouterr()
        assert "TP: 10" in captured.out
        assert "FN: 5" in captured.out
        assert "TN: 20" in captured.out
        assert "FP: 3" in captured.out
    
    def test_print_empty_metrics(self, capsys):
        """Test printing empty metrics dictionary."""
        metrics = {}
        
        print_evaluation_metrics(metrics)
        
        captured = capsys.readouterr()
        assert captured.out == ""


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_nan_values_in_dataframe_with_ra(self):
        """Test handling of NaN values in dataframe for WITH_RA."""
        data = pd.DataFrame({
            "Groundtruth_redundant_assumption_number": [1, 2, 3],
            "llm_answer_ordinal_number_of_redundant_assumption": [1, pd.NA, 3],
            "llm_answer_proof_review": [True, True, False]
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # Should handle NaN values gracefully
        assert "TP" in yesno_metrics
        assert "FN" in yesno_metrics
        assert "TP" in review_metrics
        assert "FN" in review_metrics
    
    def test_nan_values_in_dataframe_without_ra(self):
        """Test handling of NaN values in dataframe for WITHOUT_RA."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes", "no", pd.NA],
            "llm_answer_ordinal_number_of_redundant_assumption": [1, None, None],
            "llm_answer_proof_review": [True, False, True]
        })
        
        # pd.NA will be converted to string "nan" or similar, which doesn't contain "yes" or "no"
        # So it will be treated as "no" (yesno=0)
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # Should handle NaN values gracefully
        assert "TN" in yesno_metrics
        assert "FP" in yesno_metrics
        assert "TN" in review_metrics
        assert "FP" in review_metrics
    
    def test_large_dataset_with_ra(self):
        """Test with a larger dataset for WITH_RA."""
        n = 1000
        data = pd.DataFrame({
            "Groundtruth_redundant_assumption_number": list(range(1, n + 1)),
            "llm_answer_ordinal_number_of_redundant_assumption": list(range(1, n + 1)),
            "llm_answer_proof_review": [True] * n
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # All correct detections, so yesno_list = [1]*n, all TP
        assert yesno_metrics["TP"] == n
        assert yesno_metrics["FN"] == 0
        # review_list = [1]*n (all proof_review=True), all TP
        assert review_metrics["TP"] == n
        assert review_metrics["FN"] == 0
    
    def test_large_dataset_without_ra(self):
        """Test with a larger dataset for WITHOUT_RA."""
        n = 1000
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes"] * (n // 2) + ["no"] * (n // 2),
            "llm_answer_ordinal_number_of_redundant_assumption": list(range(1, n + 1)),
            "llm_answer_proof_review": [True] * n
        })
        
        yesno_metrics, review_metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        # yesno_list = [1]*(n//2) + [0]*(n//2), comparing with [1]*n: TP=n//2, FN=n//2
        # So TN=n//2, FP=n//2
        assert yesno_metrics["TN"] == n // 2
        assert yesno_metrics["FP"] == n // 2
        # review_list = [0]*(n//2) (only for "no" answers), comparing with [1]*(n//2): TP=0, FN=n//2
        # So TN=0, FP=n//2
        assert review_metrics["TN"] == 0
        assert review_metrics["FP"] == n // 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])