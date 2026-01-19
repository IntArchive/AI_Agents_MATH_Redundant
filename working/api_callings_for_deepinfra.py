import os
from typing import List
import json
import time
import argparse
import re
from pathlib import Path
from openai import OpenAI
import pandas as pd
from utils import setup
from utils.dataloader import load_data


if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("GOOGLE_API_KEY"):
    try:
        setup.setup()
    except Exception:
        # If setup fails, let the underlying client raise a clearer error later.
        pass
# API Keys Configuration
API_KEYS = {
    "qwen_api": os.environ.get("QWEN_API_KEY"),
    "deepseek_api": os.environ.get("DEEPSEEK_API_KEY"),
    "openai_api": os.environ.get("OPENAI_API_KEY"),
}

# Model Configurations
QWEN_MODELS = {
    "qwen-3-32b": {
        "api_key": API_KEYS["qwen_api"],
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3-32B",
        "name": "qwen_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    },
    "Qwen3-Next-80B-A3B-Instruct":{
        "api_key": API_KEYS["qwen_api"],
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "name": "qwen_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    },
    "Qwen3-72B-Instruct":{
        "api_key": API_KEYS["qwen_api"],
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3-72B-Instruct",
        "name": "qwen_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    },
    "Qwen3-14B-Instruct":{
        "api_key": API_KEYS["qwen_api"],
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3-14B-Instruct",
        "name": "qwen_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    },
    "Qwen3-7B-Instruct":{
        "api_key": API_KEYS["qwen_api"],
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3-7B-Instruct",
        "name": "qwen_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    }
}

DEEPSEEK_MODELS = {
    "deepseek-chat": {
        "api_key": API_KEYS["deepseek_api"],
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "name": "deepseek_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    },
    "deepseek-reasoner": {
        "api_key": API_KEYS["deepseek_api"],
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner",
        "name": "deepseek_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    }
}

GEMINI_MODELS = {
    "gemini-2.5-flash": {
        "api_key": API_KEYS.get("gemini_api", "your-gemini-key-here"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "name": "gemini_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    }
}

OPENAI_MODELS = {
    "openai": {
        "api_key": API_KEYS["openai_api"],
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.2-chat-latest",
        "name": "openai_api",
        "retryable_errors": ['rate limit', 'quota', '429', '500', '503', '504', 'service unavailable', 'timeout', 'connection']
    }
}

# Combine all models
ALL_MODELS = {**QWEN_MODELS, **DEEPSEEK_MODELS, **GEMINI_MODELS}

# System message
SYSTEM_MESSAGE = """
Act as a direct, efficient, knowledgeable, and English mathematics 
professor. Your task is to read the structured mathematics problem and 
answer the following question.

Question: Does this problem have redundant assumption?

To answer the question, you must follow the format below:
###BEGIN_OF_FORMAT###
Answer: <yes/no> 

Ordinal number of redundant assumption: 
<If the problem has a redundant assumption, write the ordinal number of redundant assumption, which is the same as the ordinal number of assumption in the problem. If the problem does not have a redundant assumption, write -1> 

Redundant assumption: 
<If the problem has a redundant assumption, write the redundant assumption here, write the same as assumption the problem
has. If the problem does not have a redundant assumption, write "no"> 

Your explanation: 
<if you suppose the problem has at least one redundant assumption, your explanation must include a solution (start the proof with **PROOF**) for the problem which don't use the redundant assumption. If the problem does not have a redundant assumption, your explanation must be "The problem does not have a redundant assumption.">
###END_OF_FORMAT###

"""


class LLMClient:
    def __init__(self, model_name):
        """Initialize the LLM client with model configuration."""
        if model_name not in ALL_MODELS:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        self.config = ALL_MODELS[model_name]
        self.model_name = model_name
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )
        self.model = self.config["model"]
        self.retryable_errors = self.config["retryable_errors"]
    
    def is_retryable_error(self, error_message):
        """Check if an error is retryable based on config."""
        error_lower = str(error_message).lower()
        return any(err in error_lower for err in self.retryable_errors)
    
    def chat_completion(self, messages, max_retries=3, retry_delay=2, **kwargs):
        """
        Send a chat completion request with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional parameters for the API call
        
        Returns:
            Response from the API
        """
        # Remove None values from kwargs
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        for attempt in range(max_retries):
            # If the model is openai, add reasoning to the kwargs
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                return response
            
            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                
                if attempt < max_retries - 1 and self.is_retryable_error(error_msg):
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise


def call_llm_with_thinking_mode(llm_name, prompt, system_message, **llm_kwargs):
    """
    Call LLM in thinking mode (for models that support reasoning/thinking).
    For DeepSeek Reasoner and similar models that return thinking process.
    """
    client = LLMClient(llm_name)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(
        messages=messages,
        reasoning=llm_kwargs.get("reasoning", False),
        max_retries=llm_kwargs.get("max_retries", 2),
        temperature=llm_kwargs.get("temperature", 0),
        max_tokens=llm_kwargs.get("max_tokens"),
        timeout=llm_kwargs.get("timeout")
    )
    
    # Extract reasoning and content for thinking models
    reasoning_content = ""
    assistant_content = ""
    
    for choice in response.choices:
        message = choice.message
        # Check if the model returned reasoning content
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning_content = message.reasoning_content
        if message.content:
            assistant_content = message.content
    
    # Combine reasoning and response
    if reasoning_content:
        return f"[REASONING]\n{reasoning_content}\n\n[RESPONSE]\n{assistant_content}"
    else:
        return assistant_content


def call_llm_with_non_thinking_mode(llm_name, prompt, system_message, **llm_kwargs):
    """
    Call LLM in non-thinking mode (standard completion).
    """
    client = LLMClient(llm_name)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(
        messages=messages,
        max_retries=llm_kwargs.get("max_retries", 2),
        temperature=llm_kwargs.get("temperature", 0),
        max_tokens=llm_kwargs.get("max_tokens"),
        timeout=llm_kwargs.get("timeout")
    )
    
    return response.choices[0].message.content

def preprocess_text(text: str) -> str:
    """
    Preprocess the text to remove the extra spaces and newlines.
    """
    text = text.strip()
    # text = text.replace("\\\\( ", "$")
    # text = text.replace("\\\\(", "$")
    # text = text.replace(" \\\\)", "$")
    # text = text.replace("\\\\)", "$")

    # text = text.replace("\\\\[ ", "$$\\n")
    # text = text.replace("\\\\[", "$$\\n")
    # text = text.replace("\\\\[\\n", "$$\\n")

    # text = text.replace(" \\\\]", "\\n$$")
    # text = text.replace("\\\\]", "\\n$$")
    # text = text.replace("\\n\\\\]", "\\n$$")

    text = text.replace("\\( ", "$")
    text = text.replace("\\(", "$")
    text = text.replace(" \\)", "$")
    text = text.replace("\\)", "$")

    text = text.replace("\\[ ", "$$\\n")
    text = text.replace("\\[", "$$\\n")
    text = text.replace("\\[\\n", "$$\\n")

    text = text.replace(" \\]", "\\n$$")
    text = text.replace("\\]", "\\n$$")
    text = text.replace("\\n\\]", "\\n$$")
    return text


def output_format_as_json_object(text: str, keys: List[str]) -> dict:
    """
    Output the text in the format of a JSON object.
    """
    dictionary = {
        "Answer": r"Answer:\s*([\s\S]*?)\s*(?=Ordinal number of redundant assumption:)",
        "Ordinal number of redundant assumption": r"Ordinal number of redundant assumption:\s*([\s\S]*?)\s*(?=Redundant assumption:)",
        "Redundant assumption": r"Redundant assumption:\s*([\s\S]*?)\s*(?=Your explanation:)",
        "Your explanation": r"Your explanation:\s*([\s\S]*?)(?=###END_OF_FORMAT###|$)"
    }
    
    answer = {}
    for key in keys:
        match = re.search(dictionary[key], text)
        if match:
            answer[key] = match.group(1).strip()
        else:
            print(f"Warning: Could not find pattern for key '{key}'")
            answer[key] = None
    
    return answer



def main():
    """Main function with argparse for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Call LLM APIs with thinking/non-thinking mode support"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file"
    )
    parser.add_argument(
        "--prompt_column1",
        type=str,
        required=True,
        help="Name of the column containing the prompt"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the results"
    )
    parser.add_argument(
        "--id_from",
        type=int,
        required=True,
        help="Index of the first task to process"
    )
    parser.add_argument(
        "--id_to",
        type=int,
        required=True,
        help="Index of the last task to process"
    )
    # LLM selection
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        choices=list(GEMINI_MODELS.keys()) + list(DEEPSEEK_MODELS.keys()) + list(QWEN_MODELS.keys()),
        help="LLM to use (e.g., gemini-2.5-flash, deepseek-chat, qwen-3-32b)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["thinking", "non-thinking"],
        help="Mode to use: 'thinking' or 'non-thinking'"
    )
    
    # Optional LLM parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for LLM (default: 0)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens for response (default: None)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (default: None)"
    )
    
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum retries (default: 2)"
    )
    
    args = parser.parse_args()
    save_path = args.save_path
    system_message = SYSTEM_MESSAGE
    data = load_data(args.data_path)
    



    llm_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    
    for index, row in data.iterrows():
        if index < args.id_from or index >= args.id_to:
            continue
        else:
            # Build prompt from columns
            prompt = row[args.prompt_column1]
            # if args.prompt_column2:
            #     prompt = f"{prompt}\n\n{row[args.prompt_column2]}"
            
            # Call appropriate function based on mode
            if args.mode == "thinking":
                response = call_llm_with_thinking_mode(
                    llm_name=args.llm,
                    prompt=prompt,
                    system_message=system_message,
                    **llm_kwargs
                )
            else:  # non-thinking
                response = call_llm_with_non_thinking_mode(
                    llm_name=args.llm,
                    prompt=prompt,
                    system_message=system_message,
                    **llm_kwargs
                )
            
            # Create save directory if needed
            if not os.path.exists(Path(save_path)):
                os.makedirs(Path(save_path))
            
            # Save response to file
            response = preprocess_text(response)
            answer = output_format_as_json_object(
                response, 
                ["Answer", "Ordinal number of redundant assumption", "Redundant assumption", "Your explanation"]
            )
            
            # Format index with leading zeros
            index_str = str(index).zfill(4)
            
            # Save result as JSON
            with open(f"{save_path}/{args.llm}_{args.mode}_result_task_{index_str}.json", "w", encoding="utf-8") as f:
                json.dump({
                    'Link_API': row['Link_API'], 
                    'Title': row['Title'], 
                    'Score': row['Score'], 
                    'Category': row['Category'], 
                    'Tags': row['Tags'], 
                    'Link': row['Link'], 
                    'Content': row['Content'],
                    'AcceptedAnswer': row['AcceptedAnswer'], 
                    'llm_answer_create_structured_problem': row['llm_answer_create_structured_problem'], 
                    'reasoning_create_structured_problem': row['reasoning_create_structured_problem'], 
                    'Proof_problem': row['Proof_problem'],
                    'Original_Problem': row['Original_Problem'], 
                    'Original_Problem_with_numerical_assumption': row['Original_Problem_with_numerical_assumption'], 
                    'Number_of_Assumption': row['Number_of_Assumption'], 
                    'Groundtruth_redundant_assumption': row['Groundtruth_redundant_assumption'], 
                    'Groundtruth_redundant_assumption_number': row['Groundtruth_redundant_assumption_number'], 
                    'Problem_with_redundant_assumption': row['Problem_with_redundant_assumption'],
                    "System_message and Prompt": system_message + "\n\n" + prompt,
                    'llm_answer_yesno_redundant_assumption': answer['Answer'],
                    'llm_ordinal_number_of_redundant_assumption': answer['Ordinal number of redundant assumption'],
                    'llm_redundant_assumption': answer['Redundant assumption'],
                    'llm_explanation': answer['Your explanation']
                }, f, ensure_ascii=False, indent=4)
            
            print(f"Processed task {index_str}")


if __name__ == "__main__":
    main()