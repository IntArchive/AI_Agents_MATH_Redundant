"""
API calling module for LLMs with thinking/non-thinking mode support.
"""

import argparse
import os
from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from utils import setup
from utils.dataloader import load_data
import json
from pathlib import Path


COMMAND_TO_RUN = """
python ./working/api_callings.py \
--llm deepseek-chat \
--mode non-thinking \
--data_path "./working/data/Second_data.ods" \
--prompt_column1 "Original_Problem" \
--prompt_column2 "AcceptedAnswer" \
--response_column "Important_Result" \
--save_path "./working/data/take_redundant_assumption" \
--id_from 0 \
--id_to 10 \
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

# Model name mappings (API model IDs)
GEMINI_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3.0-flash": "gemini-3.0-flash",
    "gemini-3.0-pro": "gemini-3.0-pro",
}

DEEPSEEK_MODELS = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}


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
    # This populates DEEPSEEK_API_KEY and GOOGLE_API_KEY from working/config.yml.
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("GOOGLE_API_KEY"):
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
            elif llm_name.startswith("gemini-3.0"):
                # Gemini 3.0 uses thinking_level
                default_params["thinking_level"] = "high" if thinking_mode else "low"

        return ChatGoogleGenerativeAI(model=model_name, **default_params)
    elif llm_name in DEEPSEEK_MODELS:
        return ChatDeepSeek(model=model_name, **default_params)
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
        required=True,
        help="Name of the column containing the prompt"
    )
    parser.add_argument(
        "--response_column",
        type=str,
        required=True,
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
        choices=list(GEMINI_MODELS.keys()) + list(DEEPSEEK_MODELS.keys()),
        help="LLM to use (e.g., gemini-2.5-flash, deepseek-chat)"
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
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries (default: 2)"
    )
    
    args = parser.parse_args()
    save_path = args.save_path
    system_message = """
    You are competive professor in the field of mathematics. You are given a problem and a solution (the proof) for that problem. You need to figure out the important results to prove the proposition step by step in the solution (or the proof) and then take out randomly 1 result from the list of important results.
    You need to output the result in the following format:
    IMPORTANT_RESULT: ###START_OF_RESULT### <result> ###END_OF_RESULT###
    """
    data = load_data(args.data_path)  
    llm_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    for index, row in data.iterrows():
        if index < args.id_from or index > args.id_to:
            continue
        else:
            prompt = row[args.prompt_column1] + "\n" + row[args.prompt_column2]
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
            # Save respond to the result_task_{index}.json
            with open(f"{save_path}/result_task_{(4 - len(str(index))) * '0' + str(index)}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "prompt": prompt,
                    "system_message": system_message,
                    "Important_Result": response,
                }, f, ensure_ascii=False, indent=4)
    
    
    
    



if __name__ == "__main__":
    main()

