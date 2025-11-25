from __future__ import annotations

import argparse
import uuid

from src.agents.final_reviewer import FinalReviewerAgent
from src.agents.judge import JudgeAgent
from src.schemas.messages import Task
from src.telemetry.logging import setup_logging
from src.tools.proof_checker import ProofCheckerTool
from src.utils.settings import load_config
from src.workflows.orchestrator import Orchestrator, Role


def build_agents(config):
    judge_prompt = config.agents["judge"].prompt_path
    reviewer_prompt = config.agents["final_reviewer"].prompt_path
    judge = JudgeAgent(prompt_path=judge_prompt)
    reviewer = FinalReviewerAgent(
        prompt_path=reviewer_prompt,
        proof_checker=ProofCheckerTool(),
    )
    return [
        Role(name="judge", agent=judge),
        Role(name="final reviewer", agent=reviewer),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the math proof multi-agent workflow.")
    parser.add_argument("task", help="Problem statement for the agents.")
    parser.add_argument("--env", default="base", help="Config environment (base, dev, prod, ...).")
    args = parser.parse_args()

    config = load_config(args.env)
    setup_logging(config.logging.level)
    orchestrator = Orchestrator(roles=build_agents(config), max_rounds=config.workflow.max_rounds)
    task = Task(id=str(uuid.uuid4()), description=args.task)
    outputs = orchestrator.run(task)
    for role, content in outputs.items():
        print(f"\n[{role}]\n{content}")


if __name__ == "__main__":
    main()

