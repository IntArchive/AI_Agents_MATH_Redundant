
import pandas as pd
import argparse
import re
import json
import os
from pathlib import Path

#####
# Principle to set file name
# 1. filechuakiemtra_originalproblem: filename should be <filename_to_check>_originalproblem.xlsx
# 2. filedatthem_gtdu: filename should be <filename_to_check>_them_gtdu.xlsx
# command to run this file: 
COMMAND = """
python ./utils/2_Nm_order.py \
--filechuakiemtra_originalproblem "./data/donedatasetdanhsogiathietplusredundant_assumption_with_numerical_order.xlsx" \
--folder_to_save_json "./data/donedataset" \
--file_json_without_extension "test_json_Nm_order"
"""

# Example: python ./pipeline/2_kiemtra_OriginalProblem_va_Them_GT_du.py --filechuakiemtra_originalproblem "./data1600to2000Original_Problem_created_redundant_assumption.xlsx" --filedatthem_gtdu_without_extension "./data1600to2000Original_Problem_created_redundant_assumption_them_gtdu"
# Example: python ./pipeline/2_kiemtra_OriginalProblem_va_Them_GT_du.py --filechuakiemtra_originalproblem "/Users/macos/Documents/Research/pythonserver/python_run_prompt/Run_deepseek_create_structured_problem/math_stack_exchange_11k_filter_noContent_noAcceptedAnswer_2k2to3k6.xlsx" --filedatthem_gtdu_without_extension "./data2200to3600Original_Problem_created_redundant_assumption_them_gtdu"
###########################################################################################################
# const Pattern
PATTERN_DETECT_ORIGINAL_PROBLEM = r"###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*###"
PATTERN_DETECT_ASSUMPTION = r"Assumption:([\S\s])+Problem:|Assumption:([\S\s])+problem:|assumption:([\S\s])+Problem:|assumption:([\S\s])+problem:" #r"[Aa]ssumption:([\S\s])+[Pp]roblem:"
PATTERN_SEPERATE_ASSUMPTION = r'[\S ]+'
###########################################################################################################
# Step 1: Detect the original problem

def numerical_order_of_assumption(row, input_original_problem_col: str = 'Original_Problem'):
    """
    Detect the numerical order of assumption from the original problem.
    input_original_problem_col: str = 'Original_Problem'
    """
    assumption = re.search(PATTERN_DETECT_ASSUMPTION, row[input_original_problem_col])
    if assumption:
        assumptions = assumption.group(0).replace("Assumption:", "").replace("Problem:", "").replace("problem:", "").replace("assumption:", "").strip()
    else:
        assumptions = ""
    
    # Find the first number in the original problem
    number_pattern = PATTERN_SEPERATE_ASSUMPTION
    list_of_assumptions = re.findall(number_pattern, assumptions)

    # print("List of assumptions: ", list_of_assumptions)
    problem_with_numerical_order_assumption = "\n".join(["Assumption " + str(i+1) + ": " + list_of_assumptions[i] for i in range(len(list_of_assumptions))])
    problem_with_numerical_order_assumption = problem_with_numerical_order_assumption
    return problem_with_numerical_order_assumption

def problem_with_numerical_order_assumption(row, input_original_problem_col: str = 'Original_Problem', output_numerical_order_of_assumption_col: str = 'Numerical_order_of_Assumption_Problem'):
    """
    Detect the numerical order of assumption from the original problem.
    input_original_problem_col: str = 'Original_Problem'
    output_numerical_order_of_assumption_col: str = 'Numerical_order_of_Assumption_Problem'
    """
    assumption = "Assumption:\n" + numerical_order_of_assumption(row, input_original_problem_col = 'Original_Problem')
    problem = row[input_original_problem_col][row[input_original_problem_col].find("Problem:") if row[input_original_problem_col].find("Problem:") != -1 else row[input_original_problem_col].find("problem:") :].strip()
    problem = problem if problem[0] == "P" else "P" + problem[1:]
    row[output_numerical_order_of_assumption_col] = assumption + "\n" + problem
    return row

    

def main():
    # Now we have an excel file we want to save samples as json file, one file json is corresponding to one row in the excel file
    parser = argparse.ArgumentParser()
    parser.add_argument('--filechuakiemtra_originalproblem', type=str, default='')
    parser.add_argument('--folder_to_save_json', type=str, default='./data/json')
    parser.add_argument('--file_json_without_extension', type=str, default='')
    args = parser.parse_args()

    
    data = pd.read_excel(args.filechuakiemtra_originalproblem)
    data = data.apply(lambda row: problem_with_numerical_order_assumption(row, input_original_problem_col="Original_Problem", output_numerical_order_of_assumption_col="Original_Problem_with_numerical_assumption"), axis=1)
    

    if os.path.lexists(Path(args.folder_to_save_json)):
        pass
    else:
        os.mkdir(Path(args.folder_to_save_json))

    for index, row in data.iterrows():
        json_data = {
            "Link_API": row["Link_API"],
            "Title": row["Title"],
            "Score": row["Score"],
            "Category": row["Category"],
            "Tags": row["Tags"],
            "Link": row["Link"],
            "Content": row["Content"],
            "AcceptedAnswer": row["AcceptedAnswer"],
            "llm_answer_create_structured_problem": row["llm_answer_create_structured_problem"],
            "reasoning_create_structured_problem": row["reasoning_create_structured_problem"],
            "Original_Problem": row["Original_Problem"],
            "Original_Problem_with_numerical_assumption": row["Original_Problem_with_numerical_assumption"],
            "Proof_problem": row["Proof_problem"],
            "Redundant_assumption": row["Redundant_assumption"],
            "Problem_with_redundant_assumption": row["Problem_with_redundant_assumption"]
        }

        if index == 5:
            print("-------------------------------- Original Problem --------------------------------")
            print(json_data["Original_Problem"])
            print("-------------------------------- Original Problem with numerical assumption --------------------------------")
            print(json_data["Original_Problem_with_numerical_assumption"])
            print("-------------------------------- End of Original Problem with numerical assumption --------------------------------")

        with open(Path(args.folder_to_save_json + f"/{args.file_json_without_extension}_{(4-len(str(index)))*"0" + str(index)}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()