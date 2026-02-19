

COMMAND_TO_RUN = \
"""
python ./utils/eval_whole.py \
--TP 113 \
--FN 87 \
--FP 135 \
--TN 0 \
--TPreviewFalse 0 \
--FPreviewTrue 0
"""

def evaluate_whole(TP, FN, FP, TN, TPreviewTrue, TPreviewFalse, FPreviewTrue, FPreviewFalse):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    far = FP / (FP + TN)
    tnr = TN / (TN + FP)
    false_alarm_rate_rv = FPreviewTrue / (FPreviewTrue + FPreviewFalse)
    TNR_rv = FPreviewFalse / (FPreviewTrue + FPreviewFalse)
    return {"accuracy": accuracy, \
           "precision": precision, \
           "recall": recall, \
           "far": far, \
           "tnr": tnr, \
           "false_alarm_rate_rv": false_alarm_rate_rv, \
           "false_alarm_rate_ra": TNR_rv}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--TP", type=int, default=0)
    parser.add_argument("--FN", type=int, default=0)
    parser.add_argument("--FP", type=int, default=0)
    parser.add_argument("--TN", type=int, default=0)
    parser.add_argument("--TPreviewFalse", type=int, default=0)
    parser.add_argument("--FPreviewTrue", type=int, default=0)
    args = parser.parse_args()

    TP = args.TP
    FN = args.FN
    FP = args.FP
    TN = args.TN
    TPreviewFalse = args.TPreviewFalse
    FPreviewTrue = args.FPreviewTrue
    TPreviewTrue = TP - TPreviewFalse
    FPreviewFalse = FP - FPreviewTrue

    print(evaluate_whole(TP, FN, FP, TN, TPreviewTrue, TPreviewFalse, FPreviewTrue, FPreviewFalse))