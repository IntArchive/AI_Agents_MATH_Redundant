

COMMAND_TO_RUN = \
"""
python ./utils/eval_whole.py \
--TP 76 \
--FN 4 \
--FP 49 \
--TN 31 \
--TPreviewFalse 22 \
--FPreviewTrue 33
"""

def evaluate_whole(TP, FN, FP, TN, TPreviewTrue, TPreviewFalse, FPreviewTrue, FPreviewFalse):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    false_alarm_rate_rv = FPreviewTrue / (FPreviewTrue + FPreviewFalse)
    TNR_rv = FPreviewFalse / (FPreviewTrue + FPreviewFalse)
    return {"accuracy": accuracy, \
           "precision": precision, \
           "recall": recall, \
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