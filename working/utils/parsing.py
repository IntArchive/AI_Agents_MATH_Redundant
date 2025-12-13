from pydantic import BaseModel
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
import re
import json

# 1. Define the schema
class PlannerOutput(BaseModel):
    answer_to_Q1: str
    new_problem: Optional[str] = None
    assumptions: Optional[List[str]] = None
    redundant_assumption: Optional[str] = None
    proof_sketch: Optional[str] = None


def parse_json_from_text(text):
    """
    Extract and parse JSON object from text that contains JSON in ```json ... ``` format.
    
    Args:
        text (str): Text containing JSON in markdown code block format
        
    Returns:
        dict: Parsed JSON object, or None if no JSON block found or parsing fails
        
    Example:
        text = '''
        This text need to be parsed into json format.
        ```json
        {
          "key1": "string corresponding to the key 1",
          "key2": "string corresponding to the key 2"
        }
        ```
        '''
        result = parse_json_from_text(text)
        # Returns: {"key1": "string corresponding to the key 1", "key2": "string corresponding to the key 2"}
    """
    # Pattern to match ```json ... ``` blocks
    # The pattern uses non-greedy matching (.*?) to capture the JSON content
    pattern = r'```json\s*(.*?)\s*```'
    
    # Find all matches (in case there are multiple JSON blocks)
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return None
    
    # Get the first match (or you could return all matches if needed)
    json_str = matches[0].strip()
    
    try:
        # Parse the JSON string into a Python dictionary
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None