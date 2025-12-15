# multi_agents.py
# Run: python multi_agents.py "Build a quick Fibonacci function and verify f(10)=55"

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import os
import sys
from dataclasses import dataclass
import asyncio
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
from utils.parsing import parse_json_from_text, parse_from_text, repair_json_backslashes, repair_json

from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
import re
import json
# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\haidm18\Documents\My_git\ma_IMO\working\asset\client_secret_351517674646-cq0k028ebrpdfgio4avoeacbeiuvdpd8.apps.googleusercontent.com.json"

# 1. Define the schema
class JudgeOutput(BaseModel):
    answer_to_Q1: str = ""
    assumptions: Optional[List[str]] = None
    redundant_assumption: Optional[str] = None
    new_problem: Optional[str] = None
    solution_for_new_problem: Optional[str] = None

class PlannerOutput(BaseModel):
    new_problem: Optional[str] = None
    proof_sketch: Optional[str] = None

class MathematicianOutput(BaseModel):
    new_problem: Optional[str] = None
    detailed_proof: Optional[str] = None

class FinalReviewerOutput(BaseModel):
    proof_review: Optional[bool] = None
    clear_answer: Optional[str] = "yes"
    finished: Optional[str] = "no"
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


def extract_json_obj(text: str) -> dict:
    """
    Extract the first JSON object found in `text` and return it as a Python dict.
    Assumes there are no extra braces `{` or `}` inside string values.
    """
    # Find text between the first '{' and the last '}' (including newlines)
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the input text.")
    
    json_str = match.group(0)
    return json.loads(json_str)





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
        print("==============******")
        print("user: \n", user_task)
        print("=====================******")
        running_input = user_task
        process = {}
        running_input_log: list[dict[str, Any]] = []
        new_problem: str = ""
        proof_sketch: str = ""
        detailed_proof: str = ""
        finished: str = "no"
        clear_answer: str = "yes"
        rda: str = ""
        for round_idx in range(1, self.max_rounds + 1):
            for role in self.roles:
                # assemble a short context window from the transcript
                
                context = running_input
                # print("role: ", role.name)
                # print("context: ", context)
                # print("="*100)
                # print("="*100)
                # each agent receives the last few turns as input
                result = role.executor.invoke({"input": context})
                process[role.name] = result["output"]
                output = result["output"]

                # parse LLM output; prefer async parser when available
                def _parse_with_async_support(parser, text):
                    async def _aparse():
                        if hasattr(parser, "aparse"):
                            return await parser.aparse(text)
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(None, parser.parse, text)

                    try:
                        return asyncio.run(_aparse())
                    except RuntimeError:
                        # If already inside a running loop (e.g., notebook), fall back to sync parse.
                        return parser.parse(text)
                

                with open("running_input_test.json", "a", encoding="utf-8") as f:
                    json.dump({"role": role.name, "running_input": running_input, "output": output}, f, ensure_ascii=False, indent=4)
                    f.write("\n")
                if role.name == "judge":
                    parser = parse_json_from_text(output, mode="new_problem")
                    try: 
                        rda = parser.get("redundant_assumption")
                    except Exception as e:
                        parser = repair_json(output, ["assumptions", "redundant_assumption"])
                        rda = parser.get("redundant_assumption")
                    problem = "Prove that " + rda if "Assumption" not in rda else "Prove that " + rda[13:].strip()
                    new_problem = "Assumption:\n" + "\n".join([f"Assumption {i+1}: {assumption}" for i, assumption in enumerate(parser.get("assumptions"))]) + "\nProblem:\n" + problem
                    print("new_problem: ", new_problem)
                    if new_problem is None:
                        try:
                            new_problem = parse_from_text(output, mode="new_problem")
                            print("new_problem: ", new_problem)
                        except Exception as e:
                            print("Error parsing new_problem: ", e)
                elif role.name == "proof strategy planner":
                    parser = parse_json_from_text(output, mode="proof_sketch")
                    if parser is None:
                        try:
                            proof_sketch = parse_from_text(output, mode="proof_sketch")
                            print("proof_sketch: ", proof_sketch)
                        except Exception as e:
                            print("Error parsing proof_sketch: ", e)
                    else:
                        proof_sketch = parser.get("proof_sketch")
                        print("proof_sketch: ", proof_sketch)
                    
                        
                elif role.name == "mathematician and proof writer":
                    parser = parse_json_from_text(output, mode="detailed_proof")
                    if parser is None:
                        try:
                            detailed_proof = parse_from_text(output, mode="detailed_proof")
                            print("detailed_proof: ", detailed_proof)
                        except Exception as e:
                            print("Error parsing detailed_proof: ", e)
                    else:
                        detailed_proof = parser.get("detailed_proof")
                        print("detailed_proof: ", detailed_proof)
                        
                elif role.name == "final reviewer":
                    parser = parse_json_from_text(output, mode="finished")
                    finished = parser.get("finished")
                    clear_answer = parser.get("clear_answer")
                    print("finished: ", finished)
                    if finished is None:
                        finished = output
                        print("finished: ", finished)
                        clear_answer = "yes"
                        print("clear_answer: ", clear_answer)
                    

                    
                


                if role.name == "judge":
                    # # print("Is it here?*********************************")
                    # try:
                    #     parser = PydanticOutputParser(pydantic_object=JudgeOutput)
                    #     parsed = _parse_with_async_support(parser, output)
                    #     # Build a clear handoff message for the reviewer including the new problem and its solution
                    #     handoff_lines = []
                    #     handoff_lines.append(f"Answer to Q1: {parsed.answer_to_Q1}")
                    #     # print("co redundant assumption? Answer: ", parsed.redundant_assumption)
                    #     if parsed.redundant_assumption:
                    #         filtered_assumptions = [a for a in (parsed.assumptions or []) if a in parsed.redundant_assumption]
                    #         filtered_assumptions = [
                    #             "Assumption " + str(i + 1) + ": " + a for i, a in enumerate(filtered_assumptions)
                    #         ] if filtered_assumptions else []
                    #         if filtered_assumptions:
                    #             handoff_lines.append("Assumptions (without redundant one):")
                    #             handoff_lines.extend(filtered_assumptions)
                    #         # Prefer explicit new_problem from the judge output, otherwise fall back
                    #         new_problem_text = parsed.new_problem or find_text_in_rddassumption(parsed.redundant_assumption)
                    #         handoff_lines.append("New_problem:")
                    #         handoff_lines.append(new_problem_text)
                    #         # Include the judge's solution for reviewer to verify
                    #         if parsed.solution_for_new_problem:
                    #             handoff_lines.append("Solution_for_new_problem:")
                    #             handoff_lines.append(parsed.solution_for_new_problem)
                    #     else:
                    #         handoff_lines.append("Redundant Assumption: no")
                    #         # No new problem to solve; still pass concise status forward for review
                    #     output = "\n".join(handoff_lines) + "\n"
                    #     # Ensure next role receives the enriched message
                    #     running_input = output
                    # except Exception as e:
                    #     # print("Parsing error:", e)
                    #     output += "\n(Note: There was an error parsing the structured output.)\n"
                    # # If parsing failed, still pass along the best-effort output
                    # running_input = output
                    # # print("This is judge and the running output is: ")
                    # # print("*()++++++++")
                    # # print(output)
                    # # print("*()++++++++")
                    running_input = new_problem
                elif role.name == "proof strategy planner":
                    running_input = new_problem + "\n" + proof_sketch
                elif role.name == "mathematician and proof writer":
                    running_input = new_problem + "\n" + detailed_proof
                elif role.name == "final reviewer":
                    running_input = output

                # Record the actual output from this role in the transcript
                self.transcript.append({"speaker": role.name, "text": output})

                running_input_log.append(
                    {
                        "round": round_idx,
                        "role": role.name,
                        "output": output,
                        "running_input": running_input,
                    }
                )

                # print("="*100)
                # print("running_input:  ", running_input)
                # print("="*100)

                # stop condition
                for line in output.splitlines():
                    if finished.strip().lower() == "yes" and clear_answer.strip().lower() == "yes":
                        running_input_log.insert(0, {"user": user_task})
                        with open("running_input_Prob_WITHOUT_RA.json", "a", encoding="utf-8") as f:
                            json.dump(running_input_log, f, ensure_ascii=False, indent=4)
                        # attach conversation logs so caller can persist them
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
                    
        # if nobody concluded with final:
        return {
            "error": "no agent produced a final answer within the round limit.",
            "__transcript__": self.transcript,
            "__running_log__": running_input_log,
        }

def main():    
    config = setup.setup_WITHOUT_RA()
    save_path = config.save_path
    data = pd.read_excel(config.file_path)
    data["judge"] = ""
    data["proof strategy planner"] = ""
    data["mathematician and proof writer"] = ""
    data["final reviewer"] = ""
    problem_column = load_problem_column(config.file_path, config.target_problem_col)
    # using argparse to load the id_from and id_to

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_from', type=int, default=0)
    parser.add_argument('--id_to', type=int, default=0)
    args = parser.parse_args()
    id_from = args.id_from
    id_to = args.id_to
   
    

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

    llm_gemini_2 = ChatGoogleGenerativeAI( 
        model="gemini-2.5-pro", 
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    parser1 = PydanticOutputParser(pydantic_object=JudgeOutput )    
    parser2 = PydanticOutputParser(pydantic_object=PlannerOutput)
    parser3 = PydanticOutputParser(pydantic_object=MathematicianOutput)
    parser4 = PydanticOutputParser(pydantic_object=FinalReviewerOutput)
    
    # define three agents with different responsibilities
    # 3. Add parser instructions to your guidelines or prompt
    judge = build_agent(
        llm=llm_deepseek,
        name="judge",
        goal="""
    Read a structured mathematics problem and write answers carefully and concisely follow the JSON structure or JSON object as mentioned in guidelines.
    Answer the question Q1: 'Does it problem have a redundant assumption?'
    If it has, create a new problem that we can deduce the redundant assumption from the other assumptions.

    Now we have the new problem. Your task is to prove the new problem.
    You need to write down the new problem follow the structure and then add the new problem to JSON object as mentioned in Guideline_2.
    ####BEGIN_OF_FORMAT###
    New problem:
    Assumptions:
    Assumption <number>: <Write the first assumption here>
    Assumption <number>: ...
    ...
    Problem:
    <Write the redundant assumption here>
    ###END_OF_FORMAT###
    """,
        guidelines=(
            "Guideline_1: Answer the question Q1 using the format 'answer_to_Q1'. "
            "Guideline_2: If there is a redundant assumption, output your answer as a JSON object with keys: 'answer_to_Q1', 'assumptions' (without redundant assumption), 'redundant_assumption', 'new_problem', 'solution_for_new_problem'. "
            "Guideline_3: If there is not a redundant assumption, output JSON with 'answer_to_Q1', 'assumptions', 'redundant_assumption: no', 'new_problem: no' and 'solution_for_new_problem: no'. "
            "Guideline_4: Store the plan via save_note, then hand off succinctly. "
            + parser1.get_format_instructions().replace("{", "{{").replace("}", "}}")  # <-- This tells the LLM how to format its output
        ),
        tools=[save_note, read_notes],
    )

    planner = build_agent(
        llm=llm_deepseek,
        name="proof strategy planner",
        goal="""
    Read a structured mathematics problem and write answers carefully and concisely follow the JSON structure or JSON object as mentioned in guidelines.
    Now break this mathematic problem into clear, minimal steps and note them. 
    Use save_note to let mathematician and proof writer read your proof sketch.
    Your answer should be in the following format:
    ###BEGIN_OF_FORMAT###
    Proof sketch: <Write the proof sketch here>
    ###END_OF_FORMAT###
    """,
        guidelines=(
            "Guideline_1: Output your answer as a JSON object with keys:'new_problem', 'proof_sketch'. "
            "Guideline_3: Store the plan via save_note, then hand off succinctly. "
            "Guideline_4: Use the following format for your proof sketch: Step 1) ... \nStep 2) ... \nStep <number_of_steps>) ..."
            + parser2.get_format_instructions().replace("{", "{{").replace("}", "}}")  # <-- This tells the LLM how to format its output
        ),
        tools=[save_note, read_notes],
    )

    mathematician = build_agent(
        llm=llm_deepseek,
        name="mathematician and proof writer",
        goal="""Read the new problem and the proof sketch and write a detailed proof for those subgoals in proof sketch, note that write new_problem as what you were given and write complete detailed_proof carefully follow the JSON structure or JSON object as mentioned in guidelines.
        Your answer should be in the following format:
        ###BEGIN_OF_FORMAT###
        Detailed proof: <Write the detailed proof here>
        ###END_OF_FORMAT###
    """,
        guidelines=(
            "Guideline_1: Output your answer as a JSON object with keys:'new_problem', 'detailed_proof'. "
            + parser3.get_format_instructions().replace("{", "{{").replace("}", "}}")  # <-- This tells the LLM how to format its output
            # "Guideline_2: follow the plan from shared notes, write a complete proof. "
        ),
        tools=[save_note, read_notes],
    )

    reviewer = build_agent(
        llm=llm_gemini_2,
        name="final reviewer",
        goal="If the proof are clear. Check correctness of the proof; write carefully down answers follow the JSON structure or JSON object as mentioned in guidelines then present the clean, final result. Otherwise, output clear_answer: 'no' and finished: 'yes'.",
        guidelines=(
            "Guideline_1: Output your answer as a JSON object with keys:'proof_review' (<True> or <False>), 'finished' (<yes> or <no>)', 'clear_answer' (<yes> or <no>). "
            # "Guideline_2: read shared notes, verify the proof from mathematician and proof writer , and only conclude when satisfied (follow the format 'Proof: <True/False>). "
            # "Guideline_3: the final line must start with 'proof:' followed by the final answer."
            # "Guideline_5: After reviewing the proof, if it is correct, output the final answer and the original problem without redundant assumption. Then output 'final:'"
            + parser4.get_format_instructions().replace("{", "{{").replace("}", "}}")
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

    for i in range(id_from, id_to, 1):
        task = problem_column.iloc[i]
        print(f"\n\n=========================== TASK {i} ===================================\n")
        final_answer = system.run(task) 
        # print("="*100)
        # print("final_answer['judge']: ", final_answer["judge"])
        # print("final_answer['proof strategy planner']: ", final_answer["proof strategy planner"])
        # print("final_answer['mathematician and proof writer']: ", final_answer["mathematician and proof writer"])
        # print("final_answer['final reviewer']: ", final_answer["final reviewer"])
        # print("="*100)
        # Extract conversation transcript for logging
        conversation = final_answer.get("__transcript__", [])
        running_log = final_answer.get("__running_log__", [])

        # Build per-role *context seen by the agent* using running_log (what each role received as input)
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

        # Save that context into the DataFrame columns
        data.at[i, "judge"] = role_contexts.get("judge", "")
        data.at[i, "proof strategy planner"] = role_contexts.get("proof strategy planner", "")
        data.at[i, "mathematician and proof writer"] = role_contexts.get("mathematician and proof writer", "")
        data.at[i, "final reviewer"] = role_contexts.get("final reviewer", "")
        if "Redundant Assumption:" in str(final_answer):
            # print("Here -------------")
            redundant_assumption = str(final_answer).split("Redundant Assumption:")[-1].strip()
            data.at[i, "Redundant_assumption"] = redundant_assumption
        # Save the current row as JSON for inspection
        row_json = data.iloc[i].to_json(force_ascii=False, indent=4)

        if os.path.lexists(Path(save_path)):
            pass
        else:
            os.mkdir(Path(save_path))

        # Save per-task result
        with open(Path(f"{save_path}/result_task_{(4 - len(str(i))) * '0' + str(i)}.json"), "w", encoding="utf-8") as f_json:
            f_json.write(row_json)

        # Save per-task conversation log (user, judge, planner, mathematician, reviewer)
        with open(Path(f"{save_path}/conversation_task_{(4 - len(str(i))) * '0' + str(i)}.json"), "w", encoding="utf-8") as f_conv:
            json.dump(
                {
                    "task_index": i,
                    "problem": task,
                    "transcript": conversation,
                    "running_log": running_log,
                    "role_contexts": role_contexts,
                },
                f_conv,
                ensure_ascii=False,
                indent=4,
            )



if __name__ == "__main__":
    main()


