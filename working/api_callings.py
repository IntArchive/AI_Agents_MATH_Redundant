"""
API calling module for LLMs with thinking/non-thinking mode support.
"""

import argparse
import os
from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils import setup
from utils.dataloader import load_data
import json
from pathlib import Path
import re


COMMAND_TO_RUN = """
python working/api_callings.py \
--llm deepseek-chat \
--mode non-thinking \
--data_path "./working/data/data200.xlsx" \
--prompt_column1 "Problem_with_redundant_assumption" \
--save_path "./working/data/Task1_PRWITHRA" \
--id_from 0 \
--id_to 50 \
--temperature 0 \
--max-retries 2

python working/api_callings.py \
--llm deepseek-chat \
--mode non-thinking \
--data_path "./working/data/data200.xlsx" \
--prompt_column1 "Original_Problem_with_numerical_assumption" \
--save_path "./working/data/Task1_PRWITHOUTRA" \
--id_from 21 \
--id_to 30 \
--temperature 0 \
--max-retries 2

python ./working/api_callings.py \
--llm gemini-2.5-pro \
--mode non-thinking \
--prompt "What is the capital of France?"

python ./working/api_callings.py --llm gemini-3.0-flash --mode non-thinking --prompt "What is the capital of France?"
python ./working/api_callings.py --llm gemini-3.0-pro --mode non-thinking --prompt "What is the capital of France?"
python ./working/api_callings.py --llm deepseek-chat --mode non-thinking --prompt "What is the capital of France?"

python ./working/api_callings.py --llm gemini-2.5-pro --mode thinking --prompt "What is the capital of France?"
python ./working/api_callings.py --llm gemini-3.0-pro --mode thinking --prompt "What is the capital of France?"
python ./working/api_callings.py --llm deepseek-reasoner --mode thinking --prompt "What is the capital of France?"
"""


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
# """
# Act as a direct, efficient, knowledgeable, and English mathematics professor. Your task is to read the definition to understand the concept of non-trivial redundant assumption and then read the structured mathematics problem and 
# answer the following question.

# # 1.Definition of non-trivial redundant assumption:
# A non-trivial redundant assumption (we will call TYPE_NON_TRIVIAL_REDUNDANT_ASSUMPTION) is an assumption that is not necessary to prove the theorem (we can indicate it as redundant assumption by create a proof for the problem without using that assumption) or unapparent result which can be proved by using the other assumptions.
# For example: Consider the following problem:
# Original problem is
# Assumption:
# Assumption 1: $a\in\mathbb{Z}$ 
# Assumption 2: $3\mid a$
# Assumption 3: $2\mid a$
# Problem: Prove that $6 \mid (a^2 + a)$.

# One student can solve the problem by using the following proof:
# Student's Proof:
# Since $2\mid a$ and $3\mid a$, we have $6\mid a$. Then, $6 \mid a^2$ and $6 \mid a$.
# Therefore, $6 \mid (a^2 + a)$.
# This student proof for this problem is correct but it uses the assumption $2\mid a$. The using the assumption $2\mid a$ is not necessary to prove the problem. Indeed the assumption $2\mid a$ is actually redundant because $a^2 + a = a(a+1)$ is the product of two consecutive numbers which is divisible by $2$. So we can create and prove a new problem which is equivalent to the original problem but without using the assumption $2\mid a$.
# New problem is
# Assumption:
# Assumption 1: $a\in\mathbb{Z}$ 
# Assumption 2: $3\mid a$
# Problem: Prove that $6 \mid (a^2 + a)$.

# New proof:
# Since $3\mid a$, we have $3\mid a^2 + a$. Besides, $a^2 + a = a(a+1)$ is the product of two consecutive numbers which is divisible by $2$. Therefore, $6 \mid (a^2 + a)$.
# This new proof for this new problem which is equivalent to the original problem but without the assumption $2\mid a$ is correct. So the assumption $2\mid a$ is actually redundant.


# # 2.Definition of non-redundant assumption:
# There are some assumptions that we usually consider them as trivial assumptions (We will call TYPE_TRIVIAL_ASSUMPTION). They usually follow signs:
# - They assign the mathematical objects of variables.

# # 3.Question:  Does this problem have non-trivial redundant assumption?

# # 4.To answer the question, you must follow the format below:
# ###BEGIN_OF_FORMAT###
# Answer: <yes/no> 

# Ordinal number of redundant assumption: 
# <If the problem has a non-trivial redundant assumption, write the ordinal number of redundant assumption, which is the same as the ordinal number of assumption in the problem. If the problem does not have a non-trivial redundant assumption, write -1> 

# Redundant assumption: 
# <If the problem has a non-trivial redundant assumption, write the non-trivial redundant assumption here, write the same as assumption the problem has. If the problem does not have a non-trivial redundant assumption, write "no"> 

# Your explanation: 
# <if you suppose the problem has at least one non-trivial redundant assumption, you should choose one of the following method to prove that chosen assumption are non-trivial redundant:
# 1. Logical analysis: Try to prove the theorem without using one of the premises. If the proof still works, that premise was non-trivial redundant.
# 2. Counterexample testing: If removing a premise would genuinely weaken the theorem, you should be able to find a counterexample where all remaining premises hold but the conclusion fails. If you can't find such a counterexample, the premise might be non-trivial redundant.
# 3. Minimal premise systems: mathematicians often seek minimal sets of non-trivial redundant premises—removing any non-trivial redundant premise that can be derived from others.
# >
# ###END_OF_FORMAT###

# """
# Prompt for create a redundant assumption
###########################################################################################################################
# You are competive professor in the field of mathematics. You are given a problem and a solution (the proof) for that problem. You need to figure out the important results to prove the proposition step by step in the solution (or the proof) and then take out randomly 1 result from the list of important results.
# You need to output the result in the following format:
# IMPORTANT_RESULT: ###START_OF_RESULT### <result> ###END_OF_RESULT###
# Model name mappings (API model IDs)
GEMINI_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    # "gemini-3.0-flash-preview": "gemini-3.0-flash-preview",
}

DEEPSEEK_MODELS = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}
OPENAI_MODELS = {
    "gpt-5.2": "gpt-5.2",
    "gpt-5-mini": "gpt-5-mini"
}

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


def get_model_name(llm_name: str, thinking_mode: bool) -> str:
    """
    Get the appropriate model name based on LLM selection and thinking mode.
    
    Args:
        llm_name: Name of the LLM (e.g., "gemini-2.5-flash", "deepseek-chat")
        thinking_mode: Whether to use thinking mode
        
    Returns:
        The model name string to use for API calls
    """
    if llm_name in GEMINI_MODELS:
        # Gemini thinking is controlled via parameters (thinking_budget / thinking_level),
        # not by changing the model name.
        return GEMINI_MODELS[llm_name]
    elif llm_name in DEEPSEEK_MODELS:
        # For DeepSeek, we map mode to the appropriate model:
        # - non-thinking  -> deepseek-chat
        # - thinking      -> deepseek-reasoner
        if thinking_mode:
            return DEEPSEEK_MODELS["deepseek-reasoner"]
        else:
            return DEEPSEEK_MODELS["deepseek-chat"]
    elif llm_name in OPENAI_MODELS:
        # OpenAI models currently do not expose a native "thinking mode";
        # we route both modes to the same chat model.
        return OPENAI_MODELS[llm_name]
    else:
        raise ValueError(f"Unknown LLM: {llm_name}")


def initialize_llm(llm_name: str, thinking_mode: bool, **kwargs) -> Any:
    """
    Initialize the appropriate LLM instance.
    
    Args:
        llm_name: Name of the LLM
        thinking_mode: Whether to use thinking mode
        **kwargs: Additional parameters for LLM initialization
        
    Returns:
        Initialized LLM instance
    """
    # Ensure API keys are loaded from config.yml via setup.py.
    # This populates DEEPSEEK_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, etc. from working/config.yml.
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
        try:
            setup.setup()
        except Exception:
            # If setup fails, let the underlying client raise a clearer error later.
            pass

    model_name = get_model_name(llm_name, thinking_mode)
    
    # Default parameters
    default_params = {
        "temperature": kwargs.get("temperature", 0),
        "max_tokens": kwargs.get("max_tokens", None),
        "timeout": kwargs.get("timeout", None),
        "max_retries": kwargs.get("max_retries", 2),
    }

    # Configure Gemini thinking behaviour correctly:
    # - Gemini 2.x (Pro only): thinking_budget
    # - Gemini 3.x (Pro only): thinking_level
    # - Flash models: no thinking mode support
    if llm_name in GEMINI_MODELS:
        is_flash = "flash" in llm_name
        is_pro = "pro" in llm_name

        if is_flash and thinking_mode:
            raise ValueError(
                f"{llm_name} does not support thinking mode. "
                "Use non-thinking mode or switch to a *-pro model."
            )

        # Only pro variants get explicit thinking controls
        if is_pro:
            if llm_name.startswith("gemini-2.5"):
                # Gemini 2.5 uses thinking_budget
                # - thinking_mode=True  -> dynamic / higher budget (-1)
                # - thinking_mode=False -> small but non-zero budget (cannot fully disable)
                default_params["thinking_budget"] = -1 if thinking_mode else 256
            elif llm_name.startswith("gemini-3"):
                # Gemini 3.0 uses thinking_level
                if thinking_mode: 
                    default_params["thinking_level"] = "high"

        return ChatGoogleGenerativeAI(model=model_name, **default_params)
    elif llm_name in DEEPSEEK_MODELS:
        return ChatDeepSeek(model=model_name, **default_params)
    elif llm_name in OPENAI_MODELS:
        default_params["reasoning"] = {"effort": "medium"} 
        return ChatOpenAI(model=model_name, **default_params)
    else:
        raise ValueError(f"Unknown LLM: {llm_name}")


def call_llm_with_thinking_mode(
    llm_name: str,
    prompt: str,
    system_message: Optional[str] = None,
    **kwargs
) -> str:
    """
    Call LLM API with thinking mode enabled.
    
    Args:
        llm_name: Name of the LLM to use
        prompt: User prompt/message
        system_message: Optional system message
        **kwargs: Additional parameters for LLM initialization
        
    Returns:
        Response text from the LLM
    """
    llm = initialize_llm(llm_name, thinking_mode=True, **kwargs)
    
    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(HumanMessage(content=prompt))
    
    response = llm.invoke(messages)
    return response.content


def call_llm_with_non_thinking_mode(
    llm_name: str,
    prompt: str,
    system_message: Optional[str] = None,
    **kwargs
) -> str:
    """
    Call LLM API with non-thinking mode (standard mode).
    
    Args:
        llm_name: Name of the LLM to use
        prompt: User prompt/message
        system_message: Optional system message
        **kwargs: Additional parameters for LLM initialization
        
    Returns:
        Response text from the LLM
    """
    llm = initialize_llm(llm_name, thinking_mode=False, **kwargs)
    
    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(HumanMessage(content=prompt))
    
    response = llm.invoke(messages)
    return response.content


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
        "--prompt_column2",
        type=str,
        default="",
        required=False,
        help="Name of the column containing the prompt"
    )
    parser.add_argument(
        "--response_column",
        type=str,
        required=False,
        help="Name of the column containing the response"
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
        choices=list(GEMINI_MODELS.keys()) + list(DEEPSEEK_MODELS.keys()) + list(OPENAI_MODELS.keys()),
        help="LLM to use (e.g., gemini-2.5-flash, deepseek-chat, gpt-4.1)"
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
        "--max-tokens",
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
            # Note that you should modify prompt upto the task
            prompt = row[args.prompt_column1]
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
            
            if os.path.lexists(Path(save_path)):
                pass
            else:
                os.mkdir(Path(save_path))
            print(response)
            if args.llm in OPENAI_MODELS:
                response = response[1]['text']
            elif args.llm in GEMINI_MODELS and args.mode == "thinking":
                response = response[0]["text"]
            else:
                response = response
            # Save respond to the result_task_{index}.json

            response = preprocess_text(response)
            answer = output_format_as_json_object(response, ["Answer", "Ordinal number of redundant assumption", "Redundant assumption", "Your explanation"])
            with open(f"{save_path}/{args.llm}_{args.mode}_result_task_{(4 - len(str(index))) * '0' + str(index)}.json", "w", encoding="utf-8") as f:
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
    
    
    
    



if __name__ == "__main__":
    main()

