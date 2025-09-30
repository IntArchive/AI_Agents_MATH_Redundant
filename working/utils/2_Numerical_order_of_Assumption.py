
import pandas as pd
import argparse
import re

#####
# Principle to set file name
# 1. filechuakiemtra_originalproblem: filename should be <filename_to_check>_originalproblem.xlsx
# 2. filedatthem_gtdu: filename should be <filename_to_check>_them_gtdu.xlsx
# command to run this file: 
COMMAND = """
python 2_Numerical_order_of_Assumption.py \
--filechuakiemtra_originalproblem "/Users/macos/Documents/Research/NCKH_2025/Redundant_Assumption_Project/Task/2_Numerically_define_assumption/math_stack_exchange_11k_filter_noContent_noAcceptedAnswer_0to8332_them_gtdu_remain3439samples_error_free_originalproblem.xlsx" \
--filedatthem_gtdu_without_extension "./math_stack_exchange_11k_filter_noContent_noAcceptedAnswer_0to8332_them_gtdu_remain3439samples_error_free"
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
def detect_problem(row, input_llm_answers_col: str = 'llm_answers', output_original_problem_col: str = 'Original_Problem'):
    """
    Detect the original problem from the llm_answers. original problem is the text between ###BEGIN_OF_FORMAT### and ###
    input_llm_answers_col: str = 'llm_answers'
    output_original_problem_col: str = 'Original_Problem'
    """

    prog = re.compile(PATTERN_DETECT_ORIGINAL_PROBLEM)
    string = row[input_llm_answers_col]
    
    result = prog.search(string)
    if result:
        row[output_original_problem_col] = result.group(1)
    else:
        row[output_original_problem_col] = "Error"

    return row

def numerical_order_of_assumption(row, input_original_problem_col: str = 'Original_Problem', input_gt_du_col: str = 'Redundant_assumption'):
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
    problem_with_numerical_order_assumption = problem_with_numerical_order_assumption + "\nAssumption " + str(len(list_of_assumptions)+1) + ": " + row[input_gt_du_col]
    return problem_with_numerical_order_assumption

def problem_with_numerical_order_assumption(row, input_original_problem_col: str = 'Original_Problem', input_gt_du_col: str = 'Redundant_assumption', output_numerical_order_of_assumption_col: str = 'Numerical_order_of_assumption_Problem'):
    """
    Detect the numerical order of assumption from the original problem.
    input_original_problem_col: str = 'Original_Problem'
    output_numerical_order_of_assumption_col: str = 'Numerical_order_of_assumption_Problem'
    """
    assumption = "Assumption:\n" + numerical_order_of_assumption(row, input_original_problem_col = 'Original_Problem', input_gt_du_col = 'Redundant_assumption')
    problem = row[input_original_problem_col][row[input_original_problem_col].find("Problem:") if row[input_original_problem_col].find("Problem:") != -1 else row[input_original_problem_col].find("problem:") :].strip()
    problem = problem if problem[0] == "P" else "P" + problem[1:]
    row[output_numerical_order_of_assumption_col] = assumption + "\n" + problem
    return row

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filechuakiemtra_originalproblem', type=str, default='')
    parser.add_argument('--filedatthem_gtdu_without_extension', type=str, default='')
    args = parser.parse_args()

    
    data = pd.read_excel(args.filechuakiemtra_originalproblem)
    
    if "Original_Problem_with_numerical_order_assumption" not in data.columns:
        data["Original_Problem_with_numerical_order_assumption"]=""
    
    data = data.apply(lambda row: problem_with_numerical_order_assumption(row, 
                                                 input_original_problem_col="Original_Problem",
                                                 input_gt_du_col="Redundant_assumption",
                                                 output_numerical_order_of_assumption_col="Original_Problem_with_numerical_order_assumption"), 
                                                 axis = 1)

    data.to_excel(args.filedatthem_gtdu_without_extension + f"danhsogiathietplusredundant_assumption_with_numerical_order.xlsx",index=False)

if __name__ == "__main__":
    main()