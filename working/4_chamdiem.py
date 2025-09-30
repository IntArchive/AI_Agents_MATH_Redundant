# Command to run this file: python chamdiem.py --filechuacham <path_to_file_chuacham> --filedacham <path_to_file_dacham>
COMMAND = """
python 4_chamdiem.py \
--filechuacham "/Users/macos/Documents/Research/NCKH_2025/Redundant_Assumption_Project/Archive_File/Finalresult_somefile_not_chamdiem/deepseekr1/data2200to3600cham_diem715_deepseekR1.xlsx" \
--filedacham "/Users/macos/Documents/Research/NCKH_2025/Redundant_Assumption_Project/Task/4_Statistics_Depth_of_RA/deepseekR1_715samples_dachamdiem.xlsx" \
--task "detect_redundant_assumption"
"""
# python 4_chamdiem.py --filechuacham "/Users/macos/Documents/Research/pythonserver/python_run_prompt/Chamdiem/data2200to3600Original_Problem_created_redundant_assumption_them_gtdu_remain715samples_0to715.xlsx" --filedacham "/Users/macos/Documents/Research/pythonserver/python_run_prompt/Chamdiem/data2200to3600Original_Problem_created_redundant_assumption_them_gtdu_remain715samples_0to71_dachamdiem.xlsx"
# Data file must have column: llm_answers_Question_redundant_assumption, Redundant_assumption

import math
import pandas as pd
import re
import argparse
num = 0
RATE_OF_SIMILARITY = 0.5

PATTERN_YES_NO_ANSWER = r"answer:\s*(.*)"
# PATTERN_REDUNDANT_ASSUMPTION = r"redundant assumption:\s*(.*)"
PATTERN_REDUNDANT_ASSUMPTION = r"redundant assumption:\s*([\S\s]+)your explanation:"

def answer_YesNo(row, input_string_column: str = 'llm_answers_Question_redundant_assumption'):
    try:
        string = row[input_string_column].lower()
        pattern = PATTERN_YES_NO_ANSWER
        prog = re.compile(pattern)
        result = prog.search(string)
    
        return result.group(1).strip()
    except:
        print(f"LinkAPI is: ",row["Link_API"])
        return ""


def YesNoQA(row, input_from_column: str = 'llm_answers_Question_redundant_assumption', output_to_column: str = 'Yes-No_QuestionQA', true_value: str = 'yes', false_value: str = 'no'):
    if answer_YesNo(row, input_string_column = input_from_column).strip() == true_value:
        row[output_to_column] = 1
    else:
        row[output_to_column] = 0
    return row




###############################################################################################################################################
def longest_common_substring(ground_truth: str, answer: str) -> str:
    # Khởi tạo bảng DP và các biến lưu trữ kết quả
    m, n = len(ground_truth), len(answer)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0  # Độ dài chuỗi con dài nhất
    end_index = 0   # Vị trí kết thúc chuỗi con trong `ground_truth`

    # Xây dựng bảng DP
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ground_truth[i-1] == answer[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                # Cập nhật độ dài lớn nhất và vị trí kết thúc
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i  # Lưu vị trí cuối trong `ground_truth`
            else:
                dp[i][j] = 0  # Không khớp, reset về 0

    # Trích xuất chuỗi con từ kết quả
    if max_length == 0:
        return ""
    else:
        start_index = end_index - max_length
        return ground_truth[start_index:end_index]

def chamdiem(ground_truth, answer):
    longest_string = longest_common_substring(ground_truth, answer)
    if len(longest_string)/len(ground_truth) >= RATE_OF_SIMILARITY:
        return 1
    else:
        return 0    

def answer_redundantAssumption(row, answer_of_llm_column: str = 'llm_answers_Question_redundant_assumption'):
    try:
        text = row[answer_of_llm_column].lower()
        pattern = PATTERN_REDUNDANT_ASSUMPTION
        
        prog = re.compile(pattern)
        result = prog.search(text.lower())
        return result.group(1)
    except:
        print(f"Error in answer_redundantAssumption LinkAPI : ",row["Link_API"])
        return ""

def chamdiem_redundantAssumption(row, \
                                 input_from_column: str = 'Redundant_assumption', \
                                 answer_of_llm_column: str = 'llm_answers_Question_redundant_assumption', \
                                 output_to_column: str = 'Detect_RedundantAssumptionQA'):
    global num
    answer = answer_redundantAssumption(row, answer_of_llm_column)
    print(f"Ques: {num}\n")
    print("answer: ", answer)
    ground_truth = row[input_from_column]
    print("ground_truth: ", ground_truth)
    with open("log.txt", "a") as f:
        f.write(f"Ques: {num}\n")
        f.write(f"answer: {answer}\n")
        f.write(f"ground_truth: {ground_truth}\n")
    num += 1
    row[output_to_column] = chamdiem(ground_truth.lower(), answer.lower())
    return row



# load_unicode_to_latex_map
def load_unicode_to_latex_map(file_path):
    unicode_to_latex = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("^")
            if len(parts) >= 3:
                char = parts[1]
                latex_cmd = parts[2]
                if char and latex_cmd:
                    unicode_to_latex[char] = latex_cmd
    print("len(unicode_to_latex): ", len(unicode_to_latex))
    print("type(unicode_to_latex): ", type(unicode_to_latex))
    print("unicode_to_latex: ", unicode_to_latex)
    return unicode_to_latex




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filechuacham', type=str, default='')
    parser.add_argument('--filedacham', type=str, default='')
    parser.add_argument('--task', type=str, default="detect_redundant_assumption")
    args = parser.parse_args()
    
    file_chuacham = args.filechuacham
    file_dacham = args.filedacham
    task = args.task

    df = pd.read_excel(file_chuacham)
    df = df[~df['llm_answers_Question_redundant_assumption'].isna()]
    
    if task == "detect_redundant_assumption":
        df = df.apply(lambda row: YesNoQA(row, 
                                          input_from_column = 'llm_answers_Question_redundant_assumption',
                                          output_to_column = 'Yes-No_QuestionQA', 
                                          true_value = 'yes', 
                                          false_value = 'no'), axis=1)
        df = df.apply(lambda row: chamdiem_redundantAssumption(row, 
                                                               input_from_column = 'Redundant_assumption', 
                                                               output_to_column = 'Detect_RedundantAssumptionQA'), axis=1)
    elif task == "detect_redundant_assumption_from_question_without_redundant_assumption":
        print("task: ", task)
        df = df.apply(lambda row: YesNoQA(row, 
                                          input_from_column = 'llm_answers_Question_groundtruth_without_redundant_assumption',
                                          output_to_column = 'Yes-No_QuestionQA', 
                                          true_value = 'no', 
                                          false_value = 'yes'), axis=1)
        
        
    if task == "detect_redundant_assumption":
        print(df['Yes-No_QuestionQA'].sum()/len(df))
        print(df['Detect_RedundantAssumptionQA'].sum()/len(df))
    elif task == "detect_redundant_assumption_from_question_without_redundant_assumption":
        print(df['Yes-No_QuestionQA'].sum()/len(df))
    df.to_excel(args.filedacham, index=False)

# python chamdiem.py --filechuacham "/Users/macos/Documents/Research/pythonserver/python_run_prompt/file_ipynb/data1000to1400Original_Problem_created_redundant_assumption_proof205_output_redundant_assumption.xlsx" --filedacham "/Users/macos/Documents/Research/pythonserver/python_run_prompt/file_ipynb/data1000to1400Original_Problem_created_redundant_assumption_proof205_output_redundant_assumption_dachamdiem.xlsx"

if __name__ == "__main__":
    main()