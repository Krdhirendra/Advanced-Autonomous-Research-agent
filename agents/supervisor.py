# --------------IMPORTING LIBRARIES--------------- #

import time
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts import *


# --------------DEFINE SCHEMA--------------- #

# Define the precise JSON blueprint using Pydantic
class SingleSearchStrategy(BaseModel):
    query: str = Field(description="The optimized, multi-faceted search query string")
    target_domains: List[str] = Field(default=[], description="Authoritative niche domains to search within")
    exclude_domains: List[str] = Field(default=[], description="Low quality domains to filter out")
    preferred_tool: str = Field(default="web_search", description="Preferred tool: 'web_search' or 'local_retrieve'")

class SupervisorSearchSchema(BaseModel):
    searches: List[SingleSearchStrategy] = Field(description="List of 3-5 distinct search strategies")


# --------------SUPERVISOR AGENT--------------- #

def supervisor_agent(state: dict, llm):
    strt_time = time.perf_counter()
    print("=== Supervisor Agent working ===")

    original_query = state['original_query']
    missing = state.get("missing_information", [])
    failed_links = state.get("failed_urls", [])
    iteration_count = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 3)

    if iteration_count > max_iterations:
        print(f"> Supervisor: Reached max iterations ({max_iterations}). Ending loop.")
        return {
            "search_queries": {"searches": []},
            "iteration_count": iteration_count,
            "is_complete": True,
        }

    if missing:
        system_message = GENERATE_MISSING_INFO_QUERY_PROMPT
        human_message = f"""
        Original Topic: {original_query}\nMissing Information: {missing}
        
        CRITICAL: The following URLs previously failed to load or blocked our scraper. You MUST use the 'exclude_domains' parameter to avoid these sites: {failed_links}"""
        print(f"> Supervisor: Executing Correction Search for: {missing}")

    else:
        system_message = GENERATE_SEARCH_QUERY_PROMPT
        human_message = f"Topic: {original_query}"
        print("> Supervisor: Executing Initial Broad Search Strategy")

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    # Bind the schema directly to Gemini LLM instance
    structured_llm = llm.with_structured_output(SupervisorSearchSchema)
    structured_response = structured_llm.invoke(messages)

    # Convert Pydantic object directly to the dictionary that LangGraph expects
    search_queries = structured_response.model_dump()
    end_time = time.perf_counter()
    print(f"> supervisor action completed in {end_time - strt_time}")
    return {"search_queries": search_queries, "iteration_count": iteration_count}