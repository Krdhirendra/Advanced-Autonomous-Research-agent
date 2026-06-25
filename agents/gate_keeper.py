# --------------IMPORTING LIBRARIES--------------- #

import time
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts import *


# --------------DEFINE SCHEMA--------------- #

class QueryClassification(BaseModel):
    reasoning: str = Field(
        description="One brief sentence explaining why the query belongs in the selected category."
    )
    classification: Literal["deep_research", "general_knowledge", "chit_chat", "out_of_scope"] = Field(
        description="Categorize the query into one of the four strict buckets."
    )
    direct_response: str = Field(
        default="",
        description="Provide the final answer for general_knowledge, a polite greeting for chit_chat, or a firm refusal for out_of_scope. Leave empty ONLY if deep_research."
    )


# --------------GATEKEEPER AGENT--------------- #

def gatekeeper_agent(state, llm):

    strt_time = time.perf_counter()
    print("\n=== GATEKEEPER AGENT: Executing Intent Analysis ===")
    original_query = state['original_query']

    messages = [
        SystemMessage(content=GATEKEEPER_PROMPT),
        HumanMessage(content=f"User Query: {original_query}")
    ]
    
    structured_llm = llm.with_structured_output(QueryClassification)
    response = structured_llm.invoke(messages)

    classification = response.classification
    print(f"> Gatekeeper Reasoning: {response.reasoning}")

    end_time = time.perf_counter()
    if classification in ["general_knowledge", "chit_chat", "out_of_scope"]:
        print(f"> Gatekeeper Action: Classified as '{classification}'. Terminating graph and returning instant response.\nTime taken by gatekeeper: {end_time-strt_time}")
        return {
            "query_type": classification,
            "draft_report": response.direct_response,
            "is_complete": True
        }
    else:
        print("> Gatekeeper Action: Valid complex research topic detected. Passing to Supervisor.")
        return {"query_type": "deep_research"}