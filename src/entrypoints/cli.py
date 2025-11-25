from __future__ import annotations

import argparse
import uuid

from src.schemas.messages import Task
from src.utils.settings import load_config
from src.workflows.orchestrator import Orchestrator, Role


def build_agents(config):
    # Placeholder: hook up actual agent implementations later.
    from src.agents.base import Agent

    class EchoAgent(Agent):
        def step(self, state):
            return state.memory[-1]

    return [
        Role(name="judge", agent=EchoAgent("judge")),
        Role(name="final reviewer", agent=EchoAgent("final reviewer")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the math proof multi-agent workflow.")
    parser.add_argument("task", help="Problem statement for the agents.")
    parser.add_argument("--env", default="base", help="Config environment (base, dev, prod, ...).")
    args = parser.parse_args()

    config = load_config(args.env)
    orchestrator = Orchestrator(roles=build_agents(config), max_rounds=config.workflow.max_rounds)
    task = Task(id=str(uuid.uuid4()), description=args.task)
    outputs = orchestrator.run(task)
    for role, content in outputs.items():
        print(f"\n[{role}]\n{content}")


if __name__ == "__main__":
    main()

