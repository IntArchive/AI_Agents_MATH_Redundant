import pandas as pd

def load_problem_column(file_path, target_problem_col):
    """
    Load data from the given file and return the specified column.

    Args:
        file_path (str): Path to the data file (CSV, Excel, etc.)
        target_problem_col (str): Name of the column to load.

    Returns:
        pandas.Series: The specified column.
    """
    # Try to infer file type by extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")

    if target_problem_col not in df.columns:
        raise KeyError(f"Column '{target_problem_col}' not found in the file.")

    return df[target_problem_col]
