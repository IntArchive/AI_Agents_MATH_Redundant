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
    pattern1 = r'(```json\s*([\s\S]*?)\s*```)'
    pattern2 = r'(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*###END_OF_FORMAT###)'
    # Find all matches (in case there are multiple JSON blocks)
    matches = re.search(pattern1, text, re.DOTALL)
    if not matches:
        matches = re.search(pattern2, text, re.DOTALL)
    
    if not matches:
        return None
    
    # Get the first match (or you could return all matches if needed)
    # Check which alternative matched: group(2) for ```json``` or group(4) for ###BEGIN_OF_FORMAT###
    print("matches :", matches)
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
        pattern2 = r'(###BEGIN_OF_FORMAT###\s*New problem:\s*([\s\S]*?)\s*###)|(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*###END_OF_FORMAT###)'
    elif mode == "proof_sketch":
        pattern2 = r'(###BEGIN_OF_FORMAT###\s*Proof sketch:\s*([\s\S]*?)\s*###)|(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*###END_OF_FORMAT###)'
    elif mode == "detailed_proof":
        pattern2 = r'(###BEGIN_OF_FORMAT###\s*Detailed proof:\s*([\s\S]*?)\s*###)|(###BEGIN_OF_FORMAT###\s*([\s\S]*?)\s*###END_OF_FORMAT###)'
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
        print("content = matches.group(2).strip(): ", content)
        content = json.dumps(content)
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
    
def repair_json_backslashes(json_text: str) -> str:
    r"""
    Sửa JSON có backslash bậy trong string (LaTeX/regex/path):
    - Trong string JSON: chỉ giữ các escape an toàn: \", \\, \/, \uXXXX
    - Còn lại (vd \lambda, \det, \begin, \d, \s, ...) -> biến thành literal bằng cách thêm 1 backslash.
    - Đồng thời escape các control char thật (newline/tab/CR) nếu lỡ nằm trong string.
    """
    out = []
    i = 0
    in_str = False

    while i < len(json_text):
        ch = json_text[i]

        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue

        # inside a JSON string
        if ch == '"':
            out.append(ch)
            in_str = False
            i += 1
            continue

        if ch == '\\':
            if i + 1 >= len(json_text):
                out.append('\\\\')
                i += 1
                continue

            nxt = json_text[i + 1]

            # safe escapes
            if nxt in ['"', '\\', '/']:
                out.append('\\' + nxt)
                i += 2
                continue

            if nxt == 'u':
                hexpart = json_text[i + 2:i + 6]
                if len(hexpart) == 4 and all(c in string.hexdigits for c in hexpart):
                    out.append('\\u' + hexpart)
                    i += 6
                    continue

            # otherwise: make the backslash literal
            out.append('\\\\')
            i += 1
            continue

        # escape literal control chars inside string
        if ch == '\n':
            out.append('\\n'); i += 1; continue
        if ch == '\r':
            out.append('\\r'); i += 1; continue
        if ch == '\t':
            out.append('\\t'); i += 1; continue

        out.append(ch)
        i += 1

    return ''.join(out)

    

def repair_json(text: str, key: list[str]) -> str:
    d = dict()
    for k in key:
        pos = text.find(k)
        end = text.find('\n', pos)
        d[k] = text[pos + len(k) + 3: end]
    return d