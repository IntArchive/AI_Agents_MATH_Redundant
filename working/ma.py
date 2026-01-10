# multi_agents.py
# Run: python multi_agents.py --id_from 0 --id_to 10

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import os
import sys
from dataclasses import dataclass
import asyncio
import pandas as pd
import re
import json
import argparse

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

# Import your utility modules
from utils import setup
from utils.struc_output import struct_out  
from utils.dataloader import load_problem_column
from utils import parsing
from utils.parsing import (
    parse_json_from_text, 
    parse_from_text, 
    repair_json_backslashes, 
    repair_json
)

# ========== Pydantic Models ==========

class JudgeOutput(BaseModel):
    answer_to_Q1: str = Field(default="", description="Answer to whether there is a redundant assumption")
    assumptions: Optional[List[str]] = Field(default=None, description="List of assumptions without redundant one")
    redundant_assumption: Optional[str] = Field(default=None, description="The redundant assumption text")
    redundant_assumption_number: Optional[int] = Field(default=None, description="Index of redundant assumption")
    new_problem: Optional[str] = Field(default=None, description="Reformulated problem")
    solution_for_new_problem: Optional[str] = Field(default=None, description="Solution sketch")

class PlannerOutput(BaseModel):
    new_problem: Optional[str] = Field(default=None, description="The problem to prove")
    proof_sketch: Optional[str] = Field(default=None, description="Step-by-step proof strategy")

class MathematicianOutput(BaseModel):
    new_problem: Optional[str] = Field(default=None, description="The problem statement")
    detailed_proof: Optional[str] = Field(default=None, description="Complete detailed proof")

class FinalReviewerOutput(BaseModel):
    proof_review: Optional[bool] = Field(default=None, description="Whether proof is correct")
    clear_answer: Optional[str] = Field(default="yes", description="Whether answer is clear")
    finished: Optional[str] = Field(default="no", description="Whether process is complete")

# ========== Shared State and Tools ==========

_shared_notes: List[str] = []
_problem_statement: str = ""

@tool
def save_note(note: str) -> str:
    """Append a short note to a shared note list accessible by all agents."""
    _shared_notes.append(note)
    return f"Saved. Notes now have {len(_shared_notes)} entries."

@tool
def read_notes(placeholder: str = "") -> str:
    """Read all shared notes, concatenated. Pass empty string as argument."""
    return "\n".join(f"- {n}" for n in _shared_notes) or "(no notes yet)"

# ========== Helper Functions ==========

def find_text_in_rddassumption(text: str) -> str:
    """Extract content from 'Assumption <number>: <content>' format."""
    prefix = "Assumption "
    if text.startswith(prefix):
        parts = text.split(":", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return text

def extract_json_obj(text: str) -> dict:
    """Extract the first JSON object found in text."""
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the input text.")
    
    json_str = match.group(0)
    return json.loads(json_str)

# ========== Agent Factory ==========

def build_agent(
    llm: Union[ChatDeepSeek, ChatOpenAI, ChatGoogleGenerativeAI],
    name: str,
    goal: str,
    guidelines: str,
    tools: List,
) -> AgentExecutor:
    """Create a tool-calling agent with role-specific system prompt."""
    system = f"""You are {name}.

Goal: {goal}

Guidelines: {guidelines}

General rules:
- Think step-by-step and cite what you did.
- Use 'save_note' to persist/share notes.
- Use 'read_notes' to see shared context.
- Be concise and specific.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        handle_parsing_errors=True,
        max_iterations=15
    )

# ========== Multi-Agent Orchestrator ==========

@dataclass
class Role:
    name: str
    executor: AgentExecutor

class MultiAgentSystem:
    """Round-robin controller that feeds transcript to each agent in turn."""
    
    def __init__(self, roles: List[Role], max_rounds: int = 6):
        self.roles = roles
        self.max_rounds = max_rounds
        self.transcript: List[Dict[str, Any]] = []

    def run(self, user_task: str) -> Dict[str, Any]:
        """Execute the multi-agent workflow."""
        self.transcript = [{"speaker": "user", "text": user_task}]
        print("=" * 80)
        print("USER TASK:\n", user_task)
        print("=" * 80)
        
        running_input = user_task
        process = {}
        running_input_log: List[Dict[str, Any]] = []
        
        # State tracking
        new_problem: str = ""
        proof_sketch: str = ""
        detailed_proof: str = ""
        finished: str = "no"
        clear_answer: str = "yes"
        rda: str = ""
        redundant_assumption_number: str = "10000"
        proof_review: str = ""
        
        for round_idx in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_idx} ---")
            
            for role in self.roles:
                print(f"\nExecuting: {role.name}")
                
                # Invoke agent
                result = role.executor.invoke({"input": running_input})
                output = result["output"]
                process[role.name] = output
                
                # Parse output based on role
                if role.name == "judge":
                    parser = parse_json_from_text(output, mode="new_problem")
                    try: 
                        rda = parser.get("redundant_assumption")
                        redundant_assumption_number = parser.get("redundant_assumption_number")
                    except Exception as e:
                        parser = repair_json(output, ["assumptions", "redundant_assumption"])
                        rda = parser.get("redundant_assumption")
                        redundant_assumption_number = parser.get("redundant_assumption_number")
                    
                    if rda is not None:
                        problem = "Prove that " + (rda if "Assumption" not in rda else rda[13:].strip())
                        new_problem = "Assumption:\n" + "\n".join(
                            [f"Assumption {i+1}: {assumption}" 
                             for i, assumption in enumerate(parser.get("assumptions", []))]
                        ) + "\nProblem:\n" + problem
                    else:
                        new_problem = "The problem doesn't have redundant assumptions."
                    
                    running_input = new_problem
                    
                elif role.name == "proof strategy planner":
                    parser = parse_json_from_text(output, mode="proof_sketch")
                    if parser is None:
                        try:
                            proof_sketch = parse_from_text(output, mode="proof_sketch")
                        except Exception as e:
                            print(f"Error parsing proof_sketch: {e}")
                    else:
                        proof_sketch = parser.get("proof_sketch", "")
                    
                    running_input = new_problem + "\n" + proof_sketch
                    
                elif role.name == "mathematician and proof writer":
                    parser = parse_json_from_text(output, mode="detailed_proof")
                    if parser is None:
                        try:
                            detailed_proof = parse_from_text(output, mode="detailed_proof")
                        except Exception as e:
                            print(f"Error parsing detailed_proof: {e}")
                    else:
                        detailed_proof = parser.get("detailed_proof", "")
                    
                    running_input = new_problem + "\n" + detailed_proof
                    
                elif role.name == "final reviewer":
                    parser = parse_json_from_text(output, mode="finished")
                    proof_review = parser.get("proof_review", "")
                    finished = parser.get("finished", "no")
                    clear_answer = parser.get("clear_answer", "yes")
                    
                    running_input = output
                
                # Record in transcript
                self.transcript.append({"speaker": role.name, "text": output})
                
                # Log running state
                running_input_log.append({
                    "round": round_idx,
                    "role": role.name,
                    "output": output,
                    "running_input": running_input,
                    "redundant_assumption_number": redundant_assumption_number,
                    "predicted_redundant_assumption": rda,
                    "proof_review": proof_review,
                    "clear_answer": clear_answer,
                })
                
                # Check stop condition
                if finished.strip().lower() == "yes" and clear_answer.strip().lower() == "yes":
                    running_input_log.insert(0, {"user": user_task})
                    process["__transcript__"] = self.transcript
                    process["__running_log__"] = running_input_log
                    return process
                elif finished.strip().lower() == "yes" and clear_answer.strip().lower() == "no":
                    return {
                        "error": "The proof is not clear.",
                        "__transcript__": self.transcript,
                        "__running_log__": running_input_log,
                    }
        
        # Max rounds reached
        return {
            "error": "No agent produced a final answer within the round limit.",
            "__transcript__": self.transcript,
            "__running_log__": running_input_log,
        }

# ========== Main Function ==========

def main():
    # Setup configuration
    config = setup.setup()
    save_path = config.save_path
    data = pd.read_excel(config.file_path)
    
    # Add output columns
    data["judge"] = ""
    data["proof strategy planner"] = ""
    data["mathematician and proof writer"] = ""
    data["final reviewer"] = ""
    data["predicted_redundant_assumption"] = ""
    data["redundant_assumption_number"] = "10000"
    data["proof_review"] = ""
    data["clear_answer"] = ""
    
    problem_column = load_problem_column(config.file_path, config.target_problem_col)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_from', type=int, default=0)
    parser.add_argument('--id_to', type=int, default=1)
    args = parser.parse_args()
    
    id_from = args.id_from
    id_to = args.id_to
    
    # Initialize LLMs
    llm_deepseek = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    llm_gemini_pro = ChatGoogleGenerativeAI( 
        model="gemini-2.0-flash-thinking-exp-01-21", 
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Initialize parsers
    parser1 = PydanticOutputParser(pydantic_object=JudgeOutput)    
    parser2 = PydanticOutputParser(pydantic_object=PlannerOutput)
    parser3 = PydanticOutputParser(pydantic_object=MathematicianOutput)
    parser4 = PydanticOutputParser(pydantic_object=FinalReviewerOutput)
    
    # Build agents
    judge = build_agent(
        llm=llm_deepseek,
        name="judge",
        goal="""Read a structured mathematics problem and determine if it has a redundant assumption.
If it does, create a new problem to deduce the redundant assumption from other assumptions.""",
        guidelines=(
            "Output as JSON with keys: 'answer_to_Q1', 'assumptions', 'redundant_assumption', "
            "'redundant_assumption_number', 'new_problem', 'solution_for_new_problem'. "
            "Store plan via save_note, then hand off succinctly.\n"
            + parser1.get_format_instructions().replace("{", "{{").replace("}", "}}")
        ),
        tools=[save_note, read_notes],
    )

    planner = build_agent(
        llm=llm_deepseek,
        name="proof strategy planner",
        goal="Break the mathematics problem into clear, minimal proof steps.",
        guidelines=(
            "Output as JSON with keys: 'new_problem', 'proof_sketch'. "
            "Use format: Step 1) ... \\nStep 2) ... \\nStep n) ...\n"
            + parser2.get_format_instructions().replace("{", "{{").replace("}", "}}")
        ),
        tools=[save_note, read_notes],
    )

    mathematician = build_agent(
        llm=llm_deepseek,
        name="mathematician and proof writer",
        goal="Read the problem and proof sketch, then write a complete detailed proof.",
        guidelines=(
            "Output as JSON with keys: 'new_problem', 'detailed_proof'.\n"
            + parser3.get_format_instructions().replace("{", "{{").replace("}", "}}")
        ),
        tools=[save_note, read_notes],
    )

    reviewer = build_agent(
        llm=llm_gemini_pro,
        name="final reviewer",
        goal="Check proof correctness and clarity, then present the final result.",
        guidelines=(
            "Output as JSON with keys: 'proof_review' (True/False), 'finished' (yes/no), "
            "'clear_answer' (yes/no).\n"
            + parser4.get_format_instructions().replace("{", "{{").replace("}", "}}")
        ),
        tools=[read_notes, save_note],
    )

    # Create multi-agent system
    system = MultiAgentSystem(
        roles=[
            Role("judge", judge),
            Role("proof strategy planner", planner),
            Role("mathematician and proof writer", mathematician),
            Role("final reviewer", reviewer),
        ],
        max_rounds=6,
    )

    # Process tasks
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(id_from, id_to):
        task = problem_column.iloc[i]
        print(f"\n\n{'=' * 80}\nTASK {i}\n{'=' * 80}\n")
        
        final_answer = system.run(task)
        
        # Extract logs
        conversation = final_answer.get("__transcript__", [])
        running_log = final_answer.get("__running_log__", [])
        
        # Build role contexts
        role_names = ["judge", "proof strategy planner", 
                      "mathematician and proof writer", "final reviewer"]
        role_contexts = {
            role_name: "\n\n".join(
                entry.get("running_input", "")
                for entry in running_log
                if entry.get("role") == role_name
            )
            for role_name in role_names
        }
        
        # Save to dataframe
        for role_name in role_names:
            data.at[i, role_name] = role_contexts.get(role_name, "")
        
        if running_log:
            last_log = running_log[-1]
            data.at[i, "predicted_redundant_assumption"] = last_log.get("predicted_redundant_assumption", "")
            data.at[i, "redundant_assumption_number"] = last_log.get("redundant_assumption_number", "10000")
            data.at[i, "proof_review"] = last_log.get("proof_review", "")
            data.at[i, "clear_answer"] = last_log.get("clear_answer", "")
        
        # Save results
        row_json = data.iloc[i].to_json(force_ascii=False, indent=4)
        
        with open(Path(f"{save_path}/result_task_{i:04d}.json"), "w", encoding="utf-8") as f:
            f.write(row_json)
        
        with open(Path(f"{save_path}/conversation_task_{i:04d}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "task_index": i,
                "problem": task,
                "transcript": conversation,
                "running_log": running_log,
                "role_contexts": role_contexts,
            }, f, ensure_ascii=False, indent=4)
    
    # Save final dataframe
    data.to_excel(Path(f"{save_path}/results_final.xlsx"), index=False)
    print(f"\n✅ Processing complete. Results saved to {save_path}")

if __name__ == "__main__":
    main()