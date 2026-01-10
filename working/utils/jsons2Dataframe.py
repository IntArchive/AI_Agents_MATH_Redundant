import os
import json
import pandas as pd
from typing import List, Dict, Any

COMMAND_TO_RUN = \
"""
python ./utils/jsons2Dataframe.py \
--data_dir "/home/haidm/Documents/My_git/AI_Agents_MATH_Redundant/working/data/Prob_WITHOUT_RA" \
--save_file_path "/home/haidm/Documents/My_git/AI_Agents_MATH_Redundant/working/data/test_result/dataWITHOUT_RA.xlsx"
"""

def jsons2dataframe(*,com_name:str, data_dir: str, save_file_path: str) -> pd.DataFrame:
    """
    Load all JSON files with name pattern ``{com_name}_*.json`` in ``data_dir``
    into a pandas DataFrame and save it to ``save_file_path``.

    Parameters
    ----------
    data_dir : str
        Directory containing the JSON files.
    save_file_path : str
        Path to the Excel file to save the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame constructed from all loaded JSON objects.
    """
    json_files: List[str] = sorted(
        f
        for f in os.listdir(data_dir)
        if f.startswith(com_name) and f.endswith(".json")
    )

    data_list: List[Dict[str, Any]] = []
    for fname in json_files:
        path = os.path.join(data_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                data_list.append(obj)
        except Exception as e:
            print(f"Could not load {fname}: {e}")

    df = pd.DataFrame(data_list)
    df.to_excel(save_file_path, index=False)
    print(f"Loaded {len(df)} records from {data_dir} and saved to {save_file_path}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JSON {com_name}_*.json files to a single Excel file (DataFrame).")
    parser.add_argument("--com_name", type=str, required=True, help="Common name of the JSON files")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing {com_name}_*.json files")
    parser.add_argument("--save_file_path", type=str, required=True, help="Path to save the resulting Excel file")
    args = parser.parse_args()

    jsons2dataframe(com_name=args.com_name, data_dir=args.data_dir, save_file_path=args.save_file_path)