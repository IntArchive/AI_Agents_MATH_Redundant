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


def parse_json_from_text(text, mode:str = "proof_sketch"):
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
    pattern1 = r'(```json\s*(.*?)\s*```)'
    pattern2 = r'(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*\n###END_OF_FORMAT###)'
    # Find all matches (in case there are multiple JSON blocks)
    matches = re.search(pattern1, text, re.DOTALL)
    if not matches:
        matches = re.search(pattern2, text, re.DOTALL)
    
    if not matches:
        return None
    
    # Get the first match (or you could return all matches if needed)
    # Check which alternative matched: group(2) for ```json``` or group(4) for ###BEGIN_OF_FORMAT###
    json_str = matches.group(2) if matches.group(2) is not None else matches.group(4)
    if json_str is None:
        return None
    json_str = json_str.strip()
    
    try:
        # Parse the JSON string into a Python dictionary
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def parse_from_text(text, mode:str = "proof_sketch"):
    """
    Extract and parse JSON object from text that contains JSON in ###BEGIN_OF_FORMAT### ... ###END_OF_FORMAT### format.
    
    Args:
        text (str): Text containing JSON in ###BEGIN_OF_FORMAT### ... ###END_OF_FORMAT### format
        mode (str): Mode to parse the text. Default is "proof_sketch".
    Returns:
        dict: Parsed JSON object, or None if no JSON block found or parsing fails
        
    Example:
        text = '''
        This text need to be parsed into json format.
        ###BEGIN_OF_FORMAT###
        Proof sketch: <Write the proof sketch here>
        ###END_OF_FORMAT###
        '''
        result = parse_json_from_text(text, mode="proof_sketch")
        # Returns: {"key1": "string corresponding to the key 1", "key2": "string corresponding to the key 2"}
    """
    # Pattern to match ###BEGIN_OF_FORMAT### ... ###END_OF_FORMAT### blocks
    # The pattern uses non-greedy matching (.*?) to capture the JSON content

    if mode == "new_problem" or mode == "proof_sketch" or mode == "detailed_proof":
        pattern1 = r'(```json\s*([\s\S]*?)\s*```)'

    if mode == "new_problem":
        pattern2 = r'(###BEGIN_OF_FORMAT###\s*New problem:\s*([\s\S]*?)\s*###)|(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*\n###END_OF_FORMAT###)'
    elif mode == "proof_sketch":
        pattern2 = r'(###BEGIN_OF_FORMAT###\s*Proof sketch:\s*([\s\S]*?)\s*###)|(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*\n###END_OF_FORMAT###)'
    elif mode == "detailed_proof":
        pattern2 = r'(###BEGIN_OF_FORMAT###\s*Detailed proof:\s*([\s\S]*?)\s*###)|(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*\n###END_OF_FORMAT###)'
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

    # Find all matches (in case there are multiple JSON blocks)
    matches = re.search(pattern1, text, re.DOTALL)

    if not matches:
        matches = re.search(pattern2, text, re.DOTALL)
        content = matches.group(2) if matches.group(2) is not None else matches.group(4)
        if content is None:
            return None
        content = content.strip()
        return content
    else:
        content = matches.group(2).strip()
        content = json.loads(content)
        if mode == "new_problem":
            return content.get("new_problem")
        elif mode == "proof_sketch":
            return content.get("proof_sketch")
        elif mode == "detailed_proof":
            return content.get("detailed_proof")
        elif mode == "finished":
            return content.get("finished")
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    # Get the first match (or you could return all matches if needed)
    # Check which alternative matched: group(2) for first pattern or group(4) for second pattern
    
    
    
    
