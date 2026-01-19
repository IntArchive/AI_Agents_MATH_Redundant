# multi_agents.py
# Run: python multi_agents.py --file_path data.csv --save_path results --target_problem_col problem --id_from 0 --id_to 10

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import os
from dataclasses import dataclass
from openai import OpenAI
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import setup
from utils.struc_output import struct_out  
import getpass
import os
from utils.dataloader import load_problem_column, load_data
from utils import parsing
from utils.parsing import parse_json_from_text, parse_from_text, repair_json_backslashes, repair_json

from pydantic import BaseModel
from langchain_classic.output_parsers import PydanticOutputParser
import re
import json
import argparse


def preprocess_text_after_regex(text: str) -> str:
    text = text.strip()
    return text

def output_format_as_json_object(text: str, keys: List[str]) -> dict:
    """
    Output the text in the format of a JSON object.
    """
    dictionary = {
        "Answer": r"Answer:\s*([\s\S]*?)\s*(?=Ordinal number of redundant assumption:)",
        "Ordinal number of redundant assumption": r"Ordinal number of redundant assumption:\s*([\s\S]*?)\s*(?=###END_OF_FORMAT###)",
        "Redundant assumption": r"Redundant assumption:\s*([\s\S]*?)\s*(?=Your explanation:)",
        "Your explanation": r"Your explanation:\s*([\s\S]*?)(?=###END_OF_FORMAT###|$)",
        "Answer to Q1": r"Answer to Q1:\s*([\s\S]*?)\s*(?=Redundant assumption:)",
        "Redundant assumption": r"Redundant assumption:\s*([\s\S]*?)\s*(?=Assumptions:)",
        "Assumptions": r"Assumptions:\s*([\s\S]*?)\s*(?=Ordinal number of redundant assumption:)",
        "proof sketch": r"proof sketch:\s*([\s\S]*?)\s*(?=end_of_proof_sketch)",
        "detailed proof": r"detailed proof:\s*([\s\S]*?)\s*(?=end_of_detailed_proof)",
        "proof review": r"proof review:\s*([\s\S]*?)\s*(?=finished:)",
        "finished": r"finished:\s*([\s\S]*?)\s*(?=clear answer:)",
        "clear answer": r"clear answer:\s*([\s\S]*?)\s*(?=end_of_proof_review)",
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

# ---------- tools ----------
_shared_notes: list[str] = []
_problem_statement: str = ""

@tool
def save_note(note: str) -> str:
    """append a short note to a shared note list accessible by all agents."""
    _shared_notes.append(note)
    return f"saved. notes now have {len(_shared_notes)} entries."

@tool
def read_notes(_: str = "") -> str:
    """read all shared notes, concatenated."""
    return "\n".join(f"- {n}" for n in _shared_notes) or "(no notes yet)"

def find_text_in_rddassumption(text: str) -> str:
    """
    The text in redundant assumption have the structure Assumption <number>: <content>.
    I want this function return the content of the assumption.
    """
    prefix = "Assumption "
    if text.startswith(prefix):
        parts = text.split(":", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return text


def extract_json_obj(text: str) -> dict:
    """
    Extract the first JSON object found in `text` and return it as a Python dict.
    Assumes there are no extra braces `{` or `}` inside string values.
    """
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the input text.")
    
    json_str = match.group(0)
    return json.loads(json_str)


# ---------- agent factory ----------
def build_agent(
    llm: Union[ChatDeepSeek, ChatOpenAI],
    name: str,
    goal: str,
    guidelines: str,
    tools: list,
) -> tuple[AgentExecutor, str]:
    """create a tool-calling agent with a role-specific system prompt."""
    system = f"""you are {name}.
goal: {goal}
guidelines: {guidelines}

general rules:
- think step-by-step, cite what you did.
- be concise and specific.
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True), system
  
# ---------- orchestrator ----------
@dataclass
class role:
    name: str
    executor: AgentExecutor
    system_prompt: str

class MultiAgentSystem:
    """
    very small round-robin controller:
    - feeds the latest transcript to each agent in turn.
    - stops when any agent outputs a line starting with 'final:'.
    """
    def __init__(self, roles: list[role], max_rounds: int = 1):
        self.roles = roles
        self.max_rounds = max_rounds
        self.transcript: list[dict[str, any]] = []

    def run(self, user_task: str) -> str:
        self.transcript = [{"speaker": "user", "text": user_task}]
        print("==============******")
        print("user: \n", user_task)
        print("=====================******")
        running_input = user_task
        process = {}
        finished: str = "no"
        clear_answer: str = "yes"
        running_input_log: list[dict[str, Any]] = []
        new_problem: str = ""
        proof_sketch: str = ""
        detailed_proof: str = ""
        ordinal_number_of_redundant_assumption: Union[str, int] = "10000"
        proof_review: str = ""
        
        for round_idx in range(1, self.max_rounds + 1):
            for role in self.roles:
                context = running_input
                print("THIS IS THE TESTING OUTPUT")
                print("role.name is ", role.name)
                print("context is ", context)
                print("="*100)
                print("="*100)
                result = role.executor.invoke({"input": context})
                process[role.name] = result["output"]
                output = result["output"]

                if role.name == "judge":
                    parser = output_format_as_json_object(output, ["Answer to Q1", "Redundant assumption", "Assumptions", "Ordinal number of redundant assumption"])
                    answer_to_Q1 = parser.get("Answer to Q1")
                    redundant_assumption = parser.get("Redundant assumption")
                    assumptions = parser.get("Assumptions")
                    ordinal_number_of_redundant_assumption = parser.get("Ordinal number of redundant assumption")

                    if answer_to_Q1 is None or answer_to_Q1.strip() == "":
                        raise ValueError("Answer to Q1 is None or empty")
                    elif redundant_assumption is None or redundant_assumption.strip() == "":
                        raise ValueError("Redundant assumption is None or empty")
                    elif assumptions is None or assumptions.strip() == "":
                        raise ValueError("Assumptions is None or empty")
                    elif ordinal_number_of_redundant_assumption is None or ordinal_number_of_redundant_assumption.strip() == "":
                        raise ValueError("Ordinal number of redundant assumption is None or empty")

                    if "yes" in answer_to_Q1.strip().lower():
                        new_problem = assumptions + "\nProblem:\n" + "Prove that " + redundant_assumption
                        print("new_problem: ", new_problem)
                    else:
                        new_problem = "The problem does not have a redundant assumption. You must write down 'no'"
                        print("new_problem: ", new_problem)
                            
                elif role.name == "proof strategy planner":
                    parser = output_format_as_json_object(output, ["proof sketch"])
                    proof_sketch = parser.get("proof sketch")

                    if proof_sketch is None or proof_sketch.strip() == "":
                        raise ValueError("Proof sketch is None or empty")

                    print("proof_sketch: ", proof_sketch)
                    
                    parser["running_input"] = running_input
                    parser["output"] = output
                    parser["role"] = role.name
                    parser["round"] = round_idx

                elif role.name == "mathematician and proof writer":
                    parser = output_format_as_json_object(output, ["detailed proof"])
                    detailed_proof = parser.get("detailed proof")
                    if detailed_proof is None or detailed_proof.strip() == "":
                        raise ValueError("Detailed proof is None or empty")

                    print("detailed_proof: ", detailed_proof)
                        
                elif role.name == "final reviewer":
                    parser = output_format_as_json_object(output, ["proof review", "finished", "clear answer"])
                    proof_review = parser.get("proof review")
                    finished = parser.get("finished")
                    clear_answer = parser.get("clear answer")
                    if proof_review is None or proof_review.strip() == "":
                        raise ValueError("Proof review is None or empty")
                    if finished is None or finished.strip() == "":
                        raise ValueError("Finished is None or empty")
                    if clear_answer is None or clear_answer.strip() == "":
                        raise ValueError("Clear answer is None or empty")

                    print("proof_review: ", proof_review)
                    print("finished: ", finished)
                    print("clear_answer: ", clear_answer)


                # Save system_prompt + "\n" + running_input
                full_context = role.system_prompt + "\n" + running_input

                if role.name == "judge":
                    running_input = "The new problem is \n" + new_problem
                elif role.name == "proof strategy planner":
                    running_input = "The new problem is \n" + new_problem + "\n" "Proof sketch is \n" + proof_sketch
                elif role.name == "mathematician and proof writer":
                    running_input = "The new problem is \n" + new_problem + "\n" "Detailed proof is \n" + detailed_proof
                elif role.name == "final reviewer":
                    # Use reasoning_content from deepseek-reasoner as feedback if available, otherwise use proof_review
                    running_input = user_task + "\n" + "Proof review is \n" + output

                self.transcript.append({"speaker": role.name, "text": output})

                
                
                running_input_log.append(
                    {
                        "round": round_idx,
                        "role": role.name,
                        "output": output,
                        "running_input": running_input,
                        "system_prompt_" + role.name: full_context,
                        "llm_answer_yesno_redundant_assumption": answer_to_Q1,
                        "llm_answer_ordinal_number_of_redundant_assumption": ordinal_number_of_redundant_assumption,  
                        "llm_answer_predicted_redundant_assumption": redundant_assumption,
                        "llm_answer_proof_review": proof_review,
                        "llm_answer_clear_answer": clear_answer,

                    }
                )

                for line in output.splitlines():
                    if finished.strip().lower() == "yes" and clear_answer.strip().lower() == "yes":
                        running_input_log.insert(0, {"user": user_task})

                        process["__transcript__"] = self.transcript
                        process["__running_log__"] = running_input_log
                        return process 
                    elif finished.strip().lower() == "no" and clear_answer.strip().lower() == "yes" or (finished.strip().lower() == "no" and clear_answer.strip().lower() == "no"):
                        continue
                    elif (finished.strip().lower() == "yes" and clear_answer.strip().lower() == "no"):
                        return {
                            "error": "The proof is not clear.",
                            "__transcript__": self.transcript,
                            "__running_log__": running_input_log,
                        }
                    
        return {
            "error": "no agent produced a final answer within the round limit.",
            "__transcript__": self.transcript,
            "__running_log__": running_input_log,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-agent system for mathematical proof verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # File paths
    parser.add_argument(
        '--file_path',
        type=str,
        required=True,
        help='Path to the input data file (CSV or other supported format)'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Directory path where results will be saved'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        default="PRWITHRA",
        help='Task to solve'
    )
    parser.add_argument(

        '--target_problem_col',
        type=str,
        required=True,
        help='Name of the column containing the problems to solve'
    )
    
    # Processing range
    parser.add_argument(
        '--id_from',
        type=int,
        default=0,
        help='Starting index for processing problems (inclusive)'
    )
    parser.add_argument(
        '--id_to',
        type=int,
        default=1,
        help='Ending index for processing problems (exclusive)'
    )
    
    # Model configuration
    parser.add_argument(
        '--deepinfra_api_key',
        type=str,
        default=None,
        help='DeepInfra API key (if not set, will use DEEPINFRA_API_KEY env variable)'
    )
    parser.add_argument(
        '--max_rounds',
        type=int,
        default=2,
        help='Maximum number of rounds for the multi-agent system'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for LLM sampling'
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    # Load data
    data = load_data(args.file_path)
    data["judge"] = ""
    data["proof strategy planner"] = ""
    data["mathematician and proof writer"] = ""
    data["final reviewer"] = ""
    data["predicted_redundant_assumption"] = ""
    data["redundant_assumption_number"] = "10000"
    problem_column = load_problem_column(args.file_path, args.target_problem_col)
    
    # Initialize LLMs using DeepInfra
    # DeepSeek Chat model for general reasoning
    llm_deepseek_chat = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_base="https://api.deepinfra.com/v1/openai",
        openai_api_key=os.environ.get("DEEPINFRA_API_KEY"),
        temperature=args.temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # DeepSeek Reasoner model for complex reasoning tasks
    llm_deepseek_reasoner = ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=args.temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


    # Build agents with system prompts
    judge_executor, judge_system = build_agent(
        llm=llm_deepseek_chat,
        name="judge",
        goal="""
    Read a structured mathematics problem and write answers carefully and concisely.
    Answer the question Q1: 'Does the problem have a redundant assumption?'
    If the problem has a redundant assumption.
    You need to write down the answer follow the below format.
    ####BEGIN_OF_FORMAT###
    Answer to Q1: <yes/no; If the problem has a redundant assumption you must write down yes, if the problem does not have a redundant assumption you must write down no>

    Redundant assumption: <if the problem has a redundant assumption you must write down the redundant assumption, if the problem does not have a redundant assumption you must write down "The problem does not have a redundant assumption.">

    Assumptions: <if the problem has a redundant assumption you must exclude the redundant assumption and write down the other assumptions as I give you, else write "no">
    
    Ordinal number of redundant assumption: <if the problem has a redundant assumption you must write down the ordinal number of the redundant assumption follow the structure, if the problem does not have a redundant assumption you must write down "-1">
    ###END_OF_FORMAT###
    """,
        guidelines="Guideline_1: ",
        tools=[],
    )

    planner_executor, planner_system = build_agent(
        llm=llm_deepseek_chat,
        name="proof strategy planner",
        goal="""
    Read a structured mathematics problem and write answers carefully and concisely follow the JSON structure or JSON object as mentioned in guidelines.
    Now break this mathematic problem into clear, minimal steps and note them. 
    Use save_note to let mathematician and proof writer read your proof sketch.
    Your answer should be in the following format:
    ###BEGIN_OF_FORMAT###
    proof sketch: 
    <Write the proof sketch here if you were given a new problem, if you were not given a new problem you must write down "We don't have a new problem so we don't have a proof sketch">
    end_of_proof_sketch
    ###END_OF_FORMAT###
    """,
        guidelines=(
            "Guideline_1: "
            "Guideline_2: Use the following format for your proof sketch: Step 1) ... \nStep 2) ... \nStep <number_of_steps>) ..."
        ),
        tools=[],
    )

    mathematician_executor, mathematician_system = build_agent(
        llm=llm_deepseek_chat,
        name="mathematician and proof writer",
        goal="""Read the new problem and the proof sketch and write a detailed proof for those subgoals in proof sketch.
        Your answer should be in the following format:
        ###BEGIN_OF_FORMAT###
        detailed proof: 
        <Write the detailed proof here if you were given a new problem and proof sketch, if you were not given a proof sketch you must write down "We don't have a proof sketch so we don't have a detailed proof">
        end_of_detailed_proof
        ###END_OF_FORMAT###
    """,
        guidelines="Guideline_1: ",
        tools=[],
    )

    reviewer_executor, reviewer_system = build_agent(

        llm=llm_deepseek_reasoner,
        name="final reviewer",
        goal="""
        Check correctness of the proof; you should output the answer in the following format:
        ###BEGIN_OF_FORMAT###
        answer_proof_review:
        proof review: <True/False; If the proof is correct, fill True, if the proof is incorrect, you should write down False>
        finished: <yes/no; If there isn't detailed proof, you should return yes. If there is detailed proof, you should check the correctness of the proof and return the answer. If the proof is correct, you should write down yes, if the proof is incorrect, you should write down no. >
        clear answer: <yes/no; If the proof is correct, you should write down yes, if the proof is incorrect, you should write down no>
        end_of_proof_review
        ###END_OF_FORMAT###
        
        """,
        guidelines="Guideline_1: ",
        tools=[],
    )

    # Create multi-agent system
    system = MultiAgentSystem(
        roles=[
            role("judge", judge_executor, judge_system),
            role("proof strategy planner", planner_executor, planner_system),
            role("mathematician and proof writer", mathematician_executor, mathematician_system),
            role("final reviewer", reviewer_executor, reviewer_system),
        ],
        max_rounds=args.max_rounds,
    )

    # Process problems in the specified range
    for i in range(args.id_from, args.id_to):
        if i >= len(problem_column):
            print(f"Skipping index {i}: out of range (max index: {len(problem_column)-1})")
            continue
        
        if args.task == "PRWITHRA":
            if args.target_problem_col == "Problem_with_redundant_assumption":
                task = args.task
                problem = problem_column.iloc[i]
            else:
                print(f"Invalid target problem column: {args.target_problem_col}")
                continue
        elif args.task == "PRWITHOUTRA":
            if args.target_problem_col == "Original_Problem_with_numerical_assumption":
                task = args.task
                problem = problem_column.iloc[i]
            else:
                print(f"Invalid target problem column: {args.target_problem_col}")
                continue

        
        print(f"\n\n=========================== TASK {i} ===================================\n")
        final_answer = system.run(problem)
        
        # Extract conversation transcript and running log
        conversation = final_answer.get("__transcript__", [])
        running_log = final_answer.get("__running_log__", [])

        # Build per-role context
        role_names = [
            "judge",
            "proof strategy planner",
            "mathematician and proof writer",
            "final reviewer",
        ]
        role_contexts = {
            role_name: "\n\n".join(
                entry.get("running_input", "")
                for entry in running_log
                if entry.get("role") == role_name
            )
            for role_name in role_names
        }

        # Save context into DataFrame
        data.at[i, "judge"] = role_contexts.get("judge", "")
        data.at[i, "proof strategy planner"] = role_contexts.get("proof strategy planner", "")
        data.at[i, "mathematician and proof writer"] = role_contexts.get("mathematician and proof writer", "")
        data.at[i, "final reviewer"] = role_contexts.get("final reviewer", "")

        data.at[i, "system_prompt_judge"] = running_log[-4].get("system_prompt_judge", "") if running_log else ""
        data.at[i, "system_prompt_proof strategy planner"] = running_log[-3].get("system_prompt_proof strategy planner", "") if running_log else ""
        data.at[i, "system_prompt_mathematician and proof writer"] = running_log[-2].get("system_prompt_mathematician and proof writer", "") if running_log else ""
        data.at[i, "system_prompt_final reviewer"] = running_log[-1].get("system_prompt_final reviewer", "") if running_log else ""
        data.at[i, "llm_answer_yesno_redundant_assumption"] = running_log[-1].get("llm_answer_yesno_redundant_assumption", "") if running_log else ""
        data.at[i, "llm_answer_predicted_redundant_assumption"] = running_log[-1].get("llm_answer_predicted_redundant_assumption", "") if running_log else ""
        data.at[i, "llm_answer_ordinal_number_of_redundant_assumption"] = running_log[-1].get("llm_answer_ordinal_number_of_redundant_assumption", "10000") if running_log else "10000"
        data.at[i, "llm_answer_proof_review"] = running_log[-1].get("llm_answer_proof_review", "") if running_log else ""
        data.at[i, "llm_answer_clear_answer"] = running_log[-1].get("llm_answer_clear_answer", "") if running_log else ""

        
        # Save results
        row_json = data.iloc[i].to_json(force_ascii=False, indent=4)

        # Create save directory if it doesn't exist
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save per-task result
        task_id_str = f"{(4 - len(str(i))) * '0' + str(i)}"
        with open(save_dir / f"result_task_{task_id_str}.json", "w", encoding="utf-8") as f_json:
            f_json.write(row_json)

        # Save per-task conversation log
        with open(save_dir / f"conversation_task_{task_id_str}.json", "w", encoding="utf-8") as f_conv:
            json.dump(
                {
                    "task_index": i,
                    "task": task,
                    "transcript": conversation,
                    "running_log": running_log,
                    "role_contexts": role_contexts,
                },
                f_conv,
                ensure_ascii=False,
                indent=4,
            )
    
    print(f"\n\nProcessing complete! Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()