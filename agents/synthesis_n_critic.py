# --------------IMPORTING LIBRARIES--------------- #

import json
import time
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts import *
from retriever import Retriever


# --------------DEFINE SCHEMA--------------- #
class IncompleteInfoStrategy(BaseModel):
        status: Literal["complete", "incomplete"] = Field(default="complete")
        missing_information: List[str] = Field(default_factory=list)


# --------------SYNTHESIS & CRITIQUE AGENT--------------- #
def synthesis_and_critique_agent(state: dict, llm, retriever:Retriever):
    """Genrates the Report based on Retrieved info and Reviews it"""
    print("\n=== SYNTHESIS AGENT: Drafting and Critiquing ===")

    # fetching state
    original_query = state["original_query"]
    ingested_docs = state.get("gathered_chunks", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # PHASE 1: RETRIEVAL & CONTEXT FORMATTING
    retrival_strt = time.perf_counter()
    print("> Pulling best context from AWS RDS...")

    if retriever is None:
        raise RuntimeError("Retriever not initialized. Ensure orchestrator creates and initializes Retriever before running synthesis agent.")
    else:
        retriever.update_documents(ingested_docs)

    best_chunks = retriever.advanced_hybrid_retrieval(original_query)
    
    formatted_references = []
    for doc in best_chunks:
        formatted_references.append({
            "content": doc.page_content,
            "url": doc.metadata.get("url", "Unknown Source")
        })
        
    context_string = json.dumps(formatted_references, indent=2)

    retrival_time = time.perf_counter()-retrival_strt
    print(f"> context string retrieved in {retrival_time} seconds")


    # PHASE 2: DRAFTING THE REPORT
    print("> Drafting report with citations...")
    
    draft_response = llm.invoke([
        SystemMessage(content=GENERATE_FINAL_REPORT_PROMPT),
        HumanMessage(content=f"CONTEXT:{context_string}, human_query:{original_query}")
    ])
    draft = draft_response.content
    

    # PHASE 3: CRITIQUING THE DRAFT
    print("> Executing self-critique loop...")

    messages = [
        SystemMessage(content=GENERATE_CRITIQUE_PROMPT),
        HumanMessage(content=f"ORIGINAL QUERY: {original_query}\n\nDRAFT REPORT:\n{draft}")
    ]
    critique_structured_llm = llm.with_structured_output(IncompleteInfoStrategy)
    critique_response = critique_structured_llm.invoke(messages)
    

    # PHASE 4: STATE UPDATING & ROUTING
    critique_status = str(getattr(critique_response, "status", "complete")).lower()
    critique_missing = list(getattr(critique_response, "missing_information", []) or [])

    if critique_status == "incomplete" and critique_missing:
        if iteration_count >= max_iterations:
            print(
                f"> Critique still incomplete, but max iterations ({max_iterations}) were reached. Ending with current draft."
            )
            return {
                "draft_report": draft,
                "missing_information": critique_missing,
                "is_complete": True,
                "formatted_references": formatted_references,
                "iteration_count": iteration_count,
            }

        print(f"> Critique Failed. Missing Info: {critique_missing}")
        return {
            "draft_report": draft,
            "missing_information": critique_missing,
            "is_complete": False,
            "iteration_count": iteration_count,
        }
    else:
        print("> Critique Passed. Research complete.")
        return {
            "draft_report": draft,
            "missing_information": [],
            "is_complete": True,
            "formatted_references": formatted_references,
            "iteration_count": iteration_count,
        }