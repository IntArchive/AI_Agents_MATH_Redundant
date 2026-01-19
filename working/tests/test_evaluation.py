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
    from utils.evaluation_api import (
        binary_classification_metrics_FOR_PROBLEM_WITH_RA,
        binary_classification_metrics_FOR_PROBLEM_WITHOUT_RA,
        evaluation_metrics_for_PROBLEM_WITH_RA,
        evaluation_metrics_for_PROBLEM_WITHOUT_RA,
        print_evaluation_metrics
    )
except ImportError:
    # If running from the same directory
    from evaluation_api import (
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
            "llm_answer_yesno_redundant_assumption": ["yes", "yes", "yes"],
            "Groundtruth_redundant_assumption_number": [1, 2, 3],
            "llm_ordinal_number_of_redundant_assumption": [1, 2, 3]
        })
        
        yesno_metrics, detect_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        assert yesno_metrics["TP"] == 3
        assert yesno_metrics["FN"] == 0
        assert detect_metrics["TP"] == 3
        assert detect_metrics["FN"] == 0
    
    def test_incorrect_detections(self):
        """Test when detections are incorrect."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes", "yes", "yes"],
            "Groundtruth_redundant_assumption_number": [1, 2, 3],
            "llm_ordinal_number_of_redundant_assumption": [2, 3, 1]
        })
        
        yesno_metrics, detect_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        assert yesno_metrics["TP"] == 3
        assert yesno_metrics["FN"] == 0
        assert detect_metrics["TP"] == 0
        assert detect_metrics["FN"] == 3
    
    def test_mixed_yes_no_answers(self):
        """Test with mixed yes and no answers."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes", "no", "yes", "no"],
            "Groundtruth_redundant_assumption_number": [1, 2, 3, 4],
            "llm_ordinal_number_of_redundant_assumption": [1, 2, 4, 4]
        })
        
        yesno_metrics, detect_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        assert yesno_metrics["TP"] == 2
        assert yesno_metrics["FN"] == 2
        assert detect_metrics["TP"] == 1
        assert detect_metrics["FN"] == 3
    
    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": [],
            "Groundtruth_redundant_assumption_number": [],
            "llm_ordinal_number_of_redundant_assumption": []
        })
        
        yesno_metrics, detect_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        # Should handle empty data gracefully
        assert "TP" in yesno_metrics
        assert "FN" in yesno_metrics


class TestEvaluationMetricsForProblemWithoutRA:
    """Test evaluation_metrics_for_PROBLEM_WITHOUT_RA function."""
    
    def test_all_no_answers(self):
        """Test when all answers are 'no'."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["no", "no", "no"]
        })
        
        metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        assert metrics["TN"] == 3
        assert metrics["FP"] == 0
    
    def test_all_yes_answers(self):
        """Test when all answers are 'yes'."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes", "yes", "yes"]
        })
        
        metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        assert metrics["TN"] == 0
        assert metrics["FP"] == 3
    
    def test_mixed_answers(self):
        """Test with mixed yes and no answers."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["no", "yes", "no", "yes"]
        })
        
        metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        assert metrics["TN"] == 2
        assert metrics["FP"] == 2
    
    def test_case_insensitive(self):
        """Test that the function handles different cases."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["No", "YES", "nO", "Yes"]
        })
        
        metrics = evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
        
        assert metrics["TN"] == 2
        assert metrics["FP"] == 2
    
    def test_invalid_answer(self):
        """Test that invalid answers raise ValueError."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["maybe", "yes", "no"]
        })
        
        with pytest.raises(ValueError, match="Invalid answer"):
            evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)
    
    def test_ambiguous_answer(self):
        """Test that ambiguous answers raise ValueError."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes and no"]
        })
        
        with pytest.raises(ValueError, match="Invalid answer"):
            evaluation_metrics_for_PROBLEM_WITHOUT_RA(data)


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
    
    def test_nan_values_in_dataframe(self):
        """Test handling of NaN values in dataframe."""
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes", "no", np.nan],
            "Groundtruth_redundant_assumption_number": [1, 2, 3],
            "llm_ordinal_number_of_redundant_assumption": [1, 2, 3]
        })
        
        # This might raise an error or handle NaN - test the actual behavior
        try:
            yesno_metrics, detect_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
            # If it doesn't raise an error, check that it produces some result
            assert "TP" in yesno_metrics
        except (ValueError, TypeError):
            # Expected if NaN values are not handled
            pass
    
    def test_large_dataset(self):
        """Test with a larger dataset."""
        n = 1000
        data = pd.DataFrame({
            "llm_answer_yesno_redundant_assumption": ["yes"] * (n // 2) + ["no"] * (n // 2),
            "Groundtruth_redundant_assumption_number": list(range(1, n + 1)),
            "llm_ordinal_number_of_redundant_assumption": list(range(1, n + 1))
        })
        
        yesno_metrics, detect_metrics = evaluation_metrics_for_PROBLEM_WITH_RA(data)
        
        assert yesno_metrics["TP"] == n // 2
        assert yesno_metrics["FN"] == n // 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])