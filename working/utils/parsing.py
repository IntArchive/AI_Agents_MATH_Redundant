from pydantic import BaseModel
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser

# 1. Define the schema
class PlannerOutput(BaseModel):
    answer_to_Q1: str
    new_problem: Optional[str] = None
    assumptions: Optional[List[str]] = None
    redundant_assumption: Optional[str] = None
    proof_sketch: Optional[str] = None