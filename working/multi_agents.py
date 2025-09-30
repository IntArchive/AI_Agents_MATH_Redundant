# multi_agents.py
# Run: python multi_agents.py "Build a quick Fibonacci function and verify f(10)=55"

from typing import List, Dict, Any, Union, Optional
import os
import sys
from dataclasses import dataclass
from openai import OpenAI
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import setup
from utils.struc_output import struct_out  
import getpass
import os
from utils.dataloader import load_problem_column
from utils import parsing
from utils.parsing import PlannerOutput

from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
import json
# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\haidm18\Documents\My_git\ma_IMO\working\asset\client_secret_351517674646-cq0k028ebrpdfgio4avoeacbeiuvdpd8.apps.googleusercontent.com.json"

# 1. Define the schema
class JudgeOutput(BaseModel):
    answer_to_Q1: str
    assumptions: Optional[List[str]] = None
    redundant_assumption: Optional[str] = None

class PlannerOutput(BaseModel):
    proof_sketch: Optional[str] = None

# ---------- tools ----------
# (you can add more tools later. keep them simple to start.)

# a tiny scratchpad tool (shared note) so agents can exchange short text.
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

# add a safe python tool so the coder can run tiny snippets.
python_repl = PythonREPLTool()


# ---------- agent factory ----------
def build_agent(
    llm: Union[ChatDeepSeek, ChatOpenAI],
    name: str,
    goal: str,
    guidelines: str,
    tools: list,
) -> AgentExecutor:
    """create a tool-calling agent with a role-specific system prompt."""
    system = f"""you are {name}.
goal: {goal}
guidelines: {guidelines}

general rules:
- think step-by-step, cite what you did.
- if you need to persist/share notes, use the 'save_note' tool.
- if you need to see shared context, use 'read_notes'.
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
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


# ---------- orchestrator ----------
@dataclass
class role:
    name: str
    executor: AgentExecutor

class MultiAgentSystem:
    """
    very small round-robin controller:
    - feeds the latest transcript to each agent in turn.
    - stops when any agent outputs a line starting with 'final:'.
    """

    def __init__(self, roles: list[role], max_rounds: int = 6):
        self.roles = roles
        self.max_rounds = max_rounds
        self.transcript: list[dict[str, any]] = []

    def run(self, user_task: str) -> str:
        self.transcript = [{"speaker": "user", "text": user_task}]
        running_input = user_task
        process = {}

        for round_idx in range(1, self.max_rounds + 1):
            for role in self.roles:
                # assemble a short context window from the transcript
                if role.name != "proof strategy planner":
                    context = "\n".join([f"{t['speaker']}: {t['text']}" for t in self.transcript[-1:]])
                else:
                    context = running_input
                
                # each agent receives the last few turns as input
                result = role.executor.invoke({"input": context})
                if role.name == "judge":
                    with open("data.json", "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=4)

                process[role.name] = result["output"]
                output = result["output"]


                if role.name == "judge":
                    print("Is it here?*********************************")
                    try:
                        parser = PydanticOutputParser(pydantic_object=JudgeOutput)
                        parsed = parser.parse(output)
                        output = f"Answer to Q1: {parsed.answer_to_Q1}\n"
                        print("co redundant assumption:", parsed.redundant_assumption)
                        if parsed.redundant_assumption:
                            l = [i for i in parsed.assumptions if i != parsed.redundant_assumption] 
                            l = ["Assumption " + str(i+1) + ": " + a for i, a in enumerate(l)] if l else []
                            output += "Assumptions:\n" + "\n".join(l) + "\n"
                            output += f"Problem: \n{find_text_in_rddassumption(parsed.redundant_assumption)}\n"
                        else:
                            output += "Redundant Assumption: no\n. End final:"
                    except Exception as e:
                        print("Parsing error:", e)
                        output += "\n(Note: There was an error parsing the structured output.)\n"
                    running_input = "New_problem:\n" + output  # Update running_input to the processed output

                self.transcript.append({"speaker": role.name, "text": running_input})

                # stop condition
                for line in output.splitlines():
                    if line.strip().startswith("final:"):
                        return process 

                # keep passing along the latest output
                running_input = output
        # if nobody concluded with final:
        return "no agent produced a final answer within the round limit."

def main():    
    config = setup.setup()
    data = pd.read_excel(config.file_path)
    data["judge"] = ""
    data["proof strategy planner"] = ""
    data["mathematician and proof writer"] = ""
    data["final reviewer"] = ""
    problem_column = load_problem_column(config.file_path, config.target_problem_col)
    num_tasks = 20
    

    llm_deepseek = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
        )
    
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )


    parser1 = PydanticOutputParser(pydantic_object=JudgeOutput )    
    parser2 = PydanticOutputParser(pydantic_object=PlannerOutput)
    # define three agents with different responsibilities
    # 3. Add parser instructions to your guidelines or prompt
    judge = build_agent(
        llm=llm_gemini,
        name="judge",
        goal="""
    Read a structured mathematics problem. 
    Answer the question Q1: 'Does it problem have a redundant assumption?'
    If it has, create a new problem that we can deduce the redundant assumption from the other assumptions.""",
        guidelines=(
            "Guideline_1: Answer the question Q1 using the format 'answer_to_Q1'. "
            "Guideline_2: If there is a redundant assumption, output your answer as a JSON object with keys: 'answer_to_Q1', 'assumptions' and 'redundant_assumption'. "
            "Guideline_3: If there is not a redundant assumption, output JSON with 'answer_to_Q1' and 'redundant_assumption: no'. "
            "Guideline_4: Store the plan via save_note, then hand off succinctly. "
            + parser1.get_format_instructions().replace("{", "{{").replace("}", "}}")  # <-- This tells the LLM how to format its output
        ),
        tools=[save_note, read_notes],
    )

    planner = build_agent(
        llm=llm_deepseek,
        name="proof strategy planner",
        goal="""
    Read a structured mathematics problem. 
    Now break this mathematic problem into clear, minimal steps and note them. 
    Use save_note to let mathematician and proof writer read your proof sketch.""",
        guidelines=(
            "Guideline_1: Output your answer as a JSON object with keys:'proof_sketch'. "
            "Guideline_3: Store the plan via save_note, then hand off succinctly. "
            "Guideline_4: Use the following format for your proof sketch: Step 1) ... \nStep 2) ... \nStep <number_of_steps>) ..."
            + parser2.get_format_instructions().replace("{", "{{").replace("}", "}}")  # <-- This tells the LLM how to format its output
        ),
        tools=[save_note, read_notes],
    )

    mathematician = build_agent(
        llm=llm_deepseek,
        name="mathematician and proof writer",
        goal="Read the proof sketch of proof strategy planner and write a detailed proof for that subgoals. Write a complete proof for the problem instead.",
        guidelines=(
            "follow the plan from shared notes, write a complete proof. "
        ),
        tools=[python_repl, save_note, read_notes],
    )

    reviewer = build_agent(
        llm=llm_gemini,
        name="final reviewer",
        goal="check correctness of the proof; then present the clean, final result.",
        guidelines=(
            "Guideline_1: read shared notes, verify the proof from mathematician and proof writer , and only conclude when satisfied (follow the format 'Proof: <True/False>). "
            "Guideline_2: the final line must start with 'proof:' followed by the final answer."
            "Guideline_3: if you believe the task is finished, clearly signal with a single line that starts with 'final:' followed by the result (no extra commentary after that line)."
            "Guideline_4: After reviewing the proof, if it is correct, output the final answer and the original problem without redundant assumption. Then output 'final:'"
        ),
        tools=[read_notes, save_note],
    )

    
    system = MultiAgentSystem(
        roles=[
            role("judge", judge),
            role("proof strategy planner", planner),
            role("mathematician and proof writer", mathematician),
            role("final reviewer", reviewer),
        ],
        max_rounds=6,
    )

    for i in range(0, 60, 1):
        task = problem_column.iloc[i]
        print(f"\n\n=========================== TASK {i} ===================================\n" + task)
        final_answer = system.run(task)
        data.at[i, "judge"] = final_answer.get("judge", "")
        data.at[i, "proof strategy planner"] = final_answer.get("proof strategy planner", "")
        data.at[i, "mathematician and proof writer"] = final_answer.get("mathematician and proof writer", "")
        data.at[i, "final reviewer"] = final_answer.get("final reviewer", "")
        if "Redundant Assumption:" in final_answer:
            print("Here -------------")
            redundant_assumption = final_answer.split("Redundant Assumption:")[-1].strip()
            data.at[i, "Redundant_assumption"] = redundant_assumption
        # Save the current row as JSON for inspection
        row_json = data.iloc[i].to_json(force_ascii=False, indent=4)
        with open(f"./Problem_WOUT_RA/result_task_{i}.json", "w", encoding="utf-8") as f_json:
            f_json.write(row_json)
        # data.to_excel("result.xlsx", index=False)
        # print("\n=== final answer ===\n" + final_answer)


if __name__ == "__main__":
    main()


