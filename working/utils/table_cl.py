from utils.read_json_in_text import _strip_leading_json_label, _balanced_brace_slices, extract_json_blocks
from utils.read_json_in_text import FENCE_RE
from pandas import DataFrame

class Table(DataFrame):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     pd.set_option('display.max_columns', None)  # Show all columns when printing
    #     pd.set_option('display.max_colwidth', None)  # Show full content of each column
    @property
    def _constructor(self):
        return Table
    def extract_field_from_embedded_json(self, text: str, field: str) -> str | None:
        """
        Extract the value of a given field from the first JSON object embedded in text.
        Uses extract_json_blocks to parse the JSON.

        Parameters:
            text: str, the input text containing embedded JSON
            field: str, the field name to extract

        Returns:
            str or None: the value of the field if found, else None
        """
        try:
            json_obj = extract_json_blocks(text)
            if isinstance(json_obj, dict):
                return json_obj[field]
        except Exception:
            pass
        return None


    def apply_functions_to_columns(self, df, col_input, col_output, *function_respective_col_output):
        """
        Apply each function in function_respective_col_output to the corresponding column in col_input,
        and store the result in the corresponding column in col_output.

        Parameters:
            df: pandas.DataFrame
            *function_respective_col_output: functions to apply (same length as col_input/col_output)
            col_input: list of str, names of input columns
            col_output: list of str, names of output columns

        Returns:
            df: pandas.DataFrame with new columns added
        """
        if not (len(function_respective_col_output) == len(col_input) == len(col_output)):
            raise ValueError("Lengths of function_respective_col_output, col_input, and col_output must be equal.")

        for func, col_in, col_out in zip(function_respective_col_output, col_input, col_output):
            # Pass col_out as the 'field' argument to func if it accepts it
            df[col_out] = df[col_in].apply(lambda x: func(x, field=col_out))
        return df
