import json
from typing import List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from A_ARA.agents.prompts import *
from retriever import advanced_hybrid_retrieval


def synthesis_and_critique_agent(state: dict, embed_model, llm):
    print("\n=== SYNTHESIS AGENT: Drafting and Critiquing ===")
    
    original_query = state["original_query"]
    ingested_docs = state.get("gathered_chunks", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # ---------------------------------------------------------
    # PHASE 1: RETRIEVAL & CONTEXT FORMATTING
    # ---------------------------------------------------------
    print("-> Pulling best context from AWS RDS...")
    best_chunks = advanced_hybrid_retrieval(original_query, ingested_docs,embedding_manager=embed_model)
    
    formatted_references = []
    for doc in best_chunks:
        formatted_references.append({
            "content": doc.page_content,
            "url": doc.metadata.get("url", "Unknown Source")
        })
        
    context_string = json.dumps(formatted_references, indent=2)
    
    # ---------------------------------------------------------
    # PHASE 2: DRAFTING THE REPORT
    # ---------------------------------------------------------
    print("-> Drafting report with citations...")
    # draft_system_prompt = (
    #     "You are an expert technical researcher. Write a comprehensive report answering the user's query.\n"
    #     "You MUST cite your sources using the URLs provided in the JSON context.\n\n"
    #     f""
    # )
    
    draft_response = llm.invoke([
        SystemMessage(content=GENERATE_FINAL_REPORT_PROMPT),
        HumanMessage(content=f"CONTEXT:{context_string}, human_query:{original_query}")
    ])
    draft = draft_response.content
    
    # ---------------------------------------------------------
    # PHASE 3: CRITIQUING THE DRAFT
    # ---------------------------------------------------------
    class IncompleteInfoStrategy(BaseModel):
        status: Literal["complete", "incomplete"] = Field(default="complete")
        missing_information: List[str] = Field(default_factory=list)

    print("-> Executing self-critique loop...")

    messages = [
        SystemMessage(content=GENERATE_CRITIQUE_PROMPT),
        HumanMessage(content=f"ORIGINAL QUERY: {original_query}\n\nDRAFT REPORT:\n{draft}")
    ]
    critique_structured_llm = llm.with_structured_output(IncompleteInfoStrategy)
    critique_response = critique_structured_llm.invoke(messages)
    
    # ---------------------------------------------------------
    # PHASE 4: STATE UPDATING & ROUTING
    # ---------------------------------------------------------
    critique_status = str(getattr(critique_response, "status", "complete")).lower()
    critique_missing = list(getattr(critique_response, "missing_information", []) or [])

    if critique_status == "incomplete" and critique_missing:
        if iteration_count >= max_iterations:
            print(
                f"-> Critique still incomplete, but max iterations ({max_iterations}) were reached. Ending with current draft."
            )
            return {
                "draft_report": draft,
                "missing_information": critique_missing,
                "is_complete": True,
                "formatted_references": formatted_references,
                "iteration_count": iteration_count,
            }

        print(f"-> Critique Failed. Missing Info: {critique_missing}")
        return {
            "draft_report": draft,
            "missing_information": critique_missing,
            "is_complete": False,
            "iteration_count": iteration_count,
        }
    else:
        print("-> Critique Passed. Research complete.")
        return {
            "draft_report": draft,
            "missing_information": [],
            "is_complete": True,
            "formatted_references": formatted_references,
            "iteration_count": iteration_count,
        }