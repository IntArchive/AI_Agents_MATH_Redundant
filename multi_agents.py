# multi_agents.py
# Run: python multi_agents.py "Build a quick Fibonacci function and verify f(10)=55"

from typing import List, Dict, Any, Union
import os
import sys
from dataclasses import dataclass
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils import setup
import getpass
import os

# ---------- Tools ----------
# (You can add more tools later. Keep them simple to start.)

# A tiny scratchpad tool (shared note) so agents can exchange short text.
_SHARED_NOTES: List[str] = []

@tool
def save_note(note: str) -> str:
    """Append a short note to a shared note list accessible by all agents."""
    _SHARED_NOTES.append(note)
    return f"Saved. Notes now have {len(_SHARED_NOTES)} entries."

@tool
def read_notes(_: str = "") -> str:
    """Read all shared notes, concatenated."""
    return "\n".join(f"- {n}" for n in _SHARED_NOTES) or "(no notes yet)"


# Add a safe Python tool so the Coder can run tiny snippets.
python_repl = PythonREPLTool()


# ---------- Agent factory ----------
def build_agent(
    llm: Union[ChatDeepSeek, ChatOpenAI],
    name: str,
    goal: str,
    guidelines: str,
    tools: list,
) -> AgentExecutor:
    """Create a tool-calling agent with a role-specific system prompt."""
    system = f"""You are {name}.
Goal: {goal}
Guidelines: {guidelines}

General rules:
- Think step-by-step, cite what you did.
- If you need to persist/share notes, use the 'save_note' tool.
- If you need to see shared context, use 'read_notes'.
- Be concise and specific.
- If you believe the task is finished, clearly signal with a single line that starts with 'FINAL:' followed by the result (no extra commentary after that line).
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


# ---------- Orchestrator ----------
@dataclass
class Role:
    name: str
    executor: AgentExecutor

class MultiAgentSystem:
    """
    Very small round-robin controller:
    - Feeds the latest transcript to each agent in turn.
    - Stops when any agent outputs a line starting with 'FINAL:'.
    """

    def __init__(self, roles: List[Role], max_rounds: int = 6):
        self.roles = roles
        self.max_rounds = max_rounds
        self.transcript: List[Dict[str, Any]] = []

    def run(self, user_task: str) -> str:
        self.transcript = [{"speaker": "USER", "text": user_task}]
        running_input = user_task

        for round_idx in range(1, self.max_rounds + 1):
            for role in self.roles:
                # Assemble a short context window from the transcript
                context = "\n".join([f"{t['speaker']}: {t['text']}" for t in self.transcript[-6:]])

                # Each agent receives the last few turns as input
                result = role.executor.invoke({"input": context})

                output = result["output"]
                with open("debug.log", "a", encoding="utf-8") as f:
                    f.write(f"\n--- Round {round_idx} | {role.name} ---\n{output}\n")
                    f.write(f"Tools used: {result.get('intermediate_steps', [])}\n")
                self.transcript.append({"speaker": role.name, "text": output})

                # Stop condition
                for line in output.splitlines():
                    if line.strip().startswith("FINAL:"):
                        return line.strip()[len("FINAL:"):].strip()

                # Keep passing along the latest output
                running_input = output

        # If nobody concluded with FINAL:
        return "No agent produced a FINAL answer within the round limit."

def main():    
    setup.setup()
    
    if len(sys.argv) < 2:
        print("Usage: python multi_agents.py \"<task>\"")
        sys.exit(1)

    task = sys.argv[1]

    # LLM (you can adjust model & temperature)
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
        )

    # Define three agents with different responsibilities
    planner = build_agent(
        llm=llm,
        name="Planner",
        goal="Break down the user's task into clear, minimal steps and note them.",
        guidelines=(
            "Produce a 2–5 step plan. "
            "Store the plan via save_note, then hand off succinctly."
        ),
        tools=[save_note, read_notes],
    )

    coder = build_agent(
        llm=llm,
        name="Coder",
        goal="Write and execute small Python snippets to implement or verify parts of the plan.",
        guidelines=(
            "Prefer concrete code. Use the Python REPL tool to test or compute. "
            "If you generate code, run a minimal check."
        ),
        tools=[python_repl, save_note, read_notes],
    )

    reviewer = build_agent(
        llm=llm,
        name="Reviewer",
        goal="Check correctness, minimality, and clarity; then present the clean, final result.",
        guidelines=(
            "Read shared notes, verify results, and only conclude when satisfied. "
            "The final line must start with 'FINAL:' followed by the final answer."
        ),
        tools=[read_notes, save_note],
    )

    system = MultiAgentSystem(
        roles=[
            Role("Planner", planner),
            Role("Coder", coder),
            Role("Reviewer", reviewer),
        ],
        max_rounds=6,
    )

    final_answer = system.run(task)
    print("\n=== FINAL ANSWER ===\n" + final_answer)


if __name__ == "__main__":
    main()
