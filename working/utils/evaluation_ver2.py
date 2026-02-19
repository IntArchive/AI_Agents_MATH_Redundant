import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

class RedundantHypothesisEvaluator:
    def __init__(self):
        self.results = []
    
    def add_prediction(self, gt_label, pred_label, num_hypotheses, confidence=None):
        """
        gt_label: "NONE" or index/name of redundant hypothesis
        pred_label: same format as gt_label
        num_hypotheses: number of hypotheses in this problem
        confidence: probability/confidence score (optional)
        """
        self.results.append({
            'gt': gt_label,
            'pred': pred_label,
            'num_hyp': num_hypotheses,
            'conf': confidence
        })
    
    def compute_metrics(self):
        n = len(self.results)
        
        # 1. Exact Match Accuracy
        # Exact Match Accuracy is easy to understand and usually misunderstood as the dectection accuracy. But they are not the same because the detection accuracy is 
        exact_match = sum(1 for r in self.results if r['gt'] == r['pred'])
        exact_match_acc = exact_match / n
        
        # 2. Detection metrics (binary)
        y_true_binary = ['HAS' if r['gt'] != 'NONE' else 'NONE' 
                        for r in self.results]
        y_pred_binary = ['HAS' if r['pred'] != 'NONE' else 'NONE' 
                        for r in self.results]
        
        cm = confusion_matrix(y_true_binary, y_pred_binary, 
                            labels=['NONE', 'HAS'])
        tn, fp, fn, tp = cm.ravel()
        
        detection_acc = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 3. Identification accuracy (conditional)
        has_redundant = [r for r in self.results if r['gt'] != 'NONE']
        if has_redundant:
            correct_id = sum(1 for r in has_redundant if r['gt'] == r['pred'])
            identification_acc = correct_id / len(has_redundant)
        else:
            identification_acc = 0
        # 3.1 Identification accuracy (unconditional)
        all_redundant = [r for r in self.results if r['pred'] != 'NONE']
        if all_redundant:
            correct_id = sum(1 for r in all_redundant if r['gt'] == r['pred'])
            identification_acc_unconditional = correct_id / len(all_redundant)
        else:
            identification_acc_unconditional = 0
        # 4. Stratified accuracy
        by_num_hyp = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in self.results:
            num = r['num_hyp']
            by_num_hyp[num]['total'] += 1
            if r['gt'] == r['pred']:
                by_num_hyp[num]['correct'] += 1
        
        stratified_acc = {
            num: stats['correct'] / stats['total']
            for num, stats in by_num_hyp.items()
        }
        
        # 5. Error type breakdown
        errors = {
            'FP': 0,  # GT=NONE, Pred≠NONE
            'FN': 0,  # GT≠NONE, Pred=NONE
            'WH': 0   # GT≠NONE, Pred≠NONE, GT≠Pred
        }
        
        for r in self.results:
            if r['gt'] == r['pred']:
                continue
            elif r['gt'] == 'NONE' and r['pred'] != 'NONE':
                errors['FP'] += 1
            elif r['gt'] != 'NONE' and r['pred'] == 'NONE':
                errors['FN'] += 1
            elif r['gt'] != 'NONE' and r['pred'] != 'NONE':
                errors['WH'] += 1
        
        # 6. Random baseline
        random_baseline = np.mean([1/(r['num_hyp']+1) for r in self.results])
        
        # 7. Top-k accuracy (if confidence scores available)
        top_k_metrics = {}
        if all(r['conf'] is not None for r in self.results):
            # This requires ranking info - simplified here
            pass
        
        # 8. MRR (if ranking available)
        mrr = None
        
        return {
            'exact_match_accuracy': exact_match_acc,
            'detection_accuracy': detection_acc,
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1,
            'specificity': specificity,
            'identification_accuracy': identification_acc,
            'identification_accuracy_unconditional': identification_acc_unconditional,
            'stratified_accuracy': stratified_acc,
            'error_breakdown': errors,
            'random_baseline': random_baseline,
            'confusion_matrix': cm,
            'total_samples': n,
            'total_with_redundancy': len(has_redundant),
            'mrr': mrr
        }
    
    def print_report(self):
        metrics = self.compute_metrics()
        
        print("="*70)
        print("Redundant Hypothesis Detection & Identification Evaluation")
        print("="*70)
        print(f"\nDataset Statistics:")
        print(f"- Total problems: {metrics['total_samples']}")
        print(f"- With redundancy: {metrics['total_with_redundancy']} "
              f"({metrics['total_with_redundancy']/metrics['total_samples']*100:.1f}%)")
        print(f"- Without redundancy: {metrics['total_samples']-metrics['total_with_redundancy']} "
              f"({(1-metrics['total_with_redundancy']/metrics['total_samples'])*100:.1f}%)")
        print(f"\nRandom Baseline: {metrics['random_baseline']:.1%}")
        
        print(f"\n{'='*70}")
        print("OVERALL PERFORMANCE:")
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.1%}")
        print(f"Improvement over baseline: {metrics['exact_match_accuracy']/metrics['random_baseline']:.1f}x")
        
        print(f"\n{'='*70}")
        print("DETECTION PERFORMANCE (Binary: Has/No Redundancy):")
        print(f"Detection Accuracy: {metrics['detection_accuracy']:.1%}")
        print(f"Precision: {metrics['detection_precision']:.1%}")
        print(f"Recall: {metrics['detection_recall']:.1%}")
        print(f"F1-Score: {metrics['detection_f1']:.1%}")
        print(f"Specificity: {metrics['specificity']:.1%}")
        
        print(f"\n{'='*70}")
        print("IDENTIFICATION PERFORMANCE (Which Hypothesis):")
        print(f"Identification Accuracy: {metrics['identification_accuracy']:.1%}")
        print(f"Identification Accuracy (unconditional): {metrics['identification_accuracy_unconditional']:.1%}")
        print(f"(among problems with redundant hypothesis)")
        
        print(f"\n{'='*70}")
        print("STRATIFIED ACCURACY (by # of hypotheses):")
        for num in sorted(metrics['stratified_accuracy'].keys()):
            acc = metrics['stratified_accuracy'][num]
            print(f"  {num} hypotheses: {acc:.1%}")
        
        print(f"\n{'='*70}")
        print("ERROR BREAKDOWN:")
        total_errors = sum(metrics['error_breakdown'].values())
        if total_errors > 0:
            for error_type, count in metrics['error_breakdown'].items():
                pct = count / total_errors * 100
                print(f"  {error_type}: {count} ({pct:.1f}%)")
                if error_type == 'FP':
                    print(f"       (False Positive: claimed redundancy when none exists)")
                elif error_type == 'FN':
                    print(f"       (False Negative: missed actual redundancy)")
                elif error_type == 'WH':
                    print(f"       (Wrong Hypothesis: detected but identified wrong one)")
        
        print("="*70)

# # Usage Example
# evaluator = RedundantHypothesisEvaluator()

# # Add predictions
# evaluator.add_prediction(gt_label='H2', pred_label='H2', num_hypotheses=3, confidence=0.95)
# evaluator.add_prediction(gt_label='H5', pred_label='H5', num_hypotheses=8, confidence=0.82)
# evaluator.add_prediction(gt_label='NONE', pred_label='NONE', num_hypotheses=5, confidence=0.88)
# evaluator.add_prediction(gt_label='H3', pred_label='H2', num_hypotheses=6, confidence=0.65)
# evaluator.add_prediction(gt_label='NONE', pred_label='H1', num_hypotheses=4, confidence=0.55)
# evaluator.add_prediction(gt_label='H7', pred_label='NONE', num_hypotheses=10, confidence=0.40)

# # ... add more predictions

# # Print comprehensive report
# evaluator.print_report()