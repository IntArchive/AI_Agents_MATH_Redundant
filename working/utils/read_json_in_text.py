import json, re

FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

def _strip_leading_json_label(s: str) -> str:
    # If the snippet begins with a line 'json' then a brace, drop that label
    # Handles: "json\n{ ... }" and "json\r\n{ ... }"
    return re.sub(r'^\s*json\s*\r?\n\s*(?=\{)', '', s, count=1, flags=re.IGNORECASE)

def _balanced_brace_slices(text: str):
    """Yield (start, end_inclusive) indices of balanced JSON-like blocks.
    Tracks strings/escapes so braces inside strings are ignored.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != '{':
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        yield (i, j)  # inclusive
                        break
            j += 1
        i = j + 1

def extract_json_blocks(text: str, first_only: bool = True):
    """Return one dict (default) or a list of dicts extracted from text."""
    # 1) fenced blocks
    for m in FENCE_RE.finditer(text):
        raw = _strip_leading_json_label(m.group(1)).strip()
        try:
            return json.loads(raw) if first_only else [json.loads(raw)]
        except Exception:
            pass  # fall through

    # 2) balanced brace scanning (collect candidates, try parse)
    results = []
    for a, b in _balanced_brace_slices(text):
        raw = _strip_leading_json_label(text[a:b+1])
        try:
            obj = json.loads(raw)
            if first_only:
                return obj
            results.append(obj)
        except Exception:
            continue
    if not results:
        raise ValueError("No valid JSON block found.")
    return results

if __name__ == "__main__":

    # Example
    sample = """The result of judge is
```json
{
"answer_to_Q1": "yes",
"assumptions": [
"$k$ is a positive integer."
],
"redundant_assumption": "$$\\\\frac{1}{\\\\sin^2 x} + \\\\frac{1}{\\\\sin^2 (\\\\frac{\\\\pi}{2}-x)} = \\\\frac{\\\\cos^2 x + \\\\sin^2 x}{\\\\cos^2 x \\\\cdot \\\\sin^2 x} = \\\\frac{4}{\\\\sin^2 2x}.$$"
}```
    """
    data = extract_json_blocks(sample)
    # data is now a proper Python dict
