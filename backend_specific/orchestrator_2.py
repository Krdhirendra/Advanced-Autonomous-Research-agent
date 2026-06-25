# -------------- IMPORTING LIBRARIES --------------- #

import operator
from functools import partial
from typing import List, Dict, Annotated, TypedDict, Any, Literal

from langgraph.graph import StateGraph, END

from agents.gate_keeper import gatekeeper_agent
from agents.supervisor import supervisor_agent
from agents.web_ingestion import web_and_ingestion_agent
from agents.synthesis_n_critic import synthesis_and_critique_agent


# -------------- GLOBAL STATE --------------- #

class GlobalState(TypedDict):
    original_query: str
    query_type: str
    search_queries: dict
    gathered_chunks: List[Any]
    failed_urls: Annotated[List[str], operator.add]
    missing_information: List[str]
    draft_report: str
    is_complete: bool
    formatted_references: List[dict]
    iteration_count: int
    max_iterations: int


# -------------- ROUTING LOGICS --------------- #

def routing_logic(state: GlobalState) -> str:
    """Decides if the graph should loop back or finish."""
    if state.get("is_complete", False):
        print("-> Routing: Data is complete. Ending graph.")
        return "end"
    else:
        print(f"-> Routing: Missing {state['missing_information']}. Looping back to Supervisor.")
        return "continue"


def gatekeeper_router(state: GlobalState) -> str:
    """Evaluates the gatekeeper's state classification."""
    if state.get("query_type") in ["general_knowledge", "chit_chat", "out_of_scope"]:
        return "end"
    return "deep_research"


# -------------- ORCHESTRATOR FACTORY --------------- #

def create_orchestrator(embed_model, retriever, rds_wrapper, llm):
    """
    Factory function to create a compiled LangGraph orchestrator.
    
    Args:
        embed_model: Pre-initialized embedding model
        retriever: Pre-initialized retriever
        rds_wrapper: Pre-initialized RDS vector store wrapper
        llm: Pre-initialized language model
    
    Returns:
        Compiled LangGraph app ready for invocation
    """

    # Bind the extra arguments to the imported functions
    bound_gatekeeper = partial(gatekeeper_agent, llm=llm)
    bound_supervisor = partial(supervisor_agent, llm=llm)
    bound_ingestion = partial(web_and_ingestion_agent, embed_model=embed_model, retriever=retriever, rds_wrapper=rds_wrapper)
    bound_synthesis = partial(synthesis_and_critique_agent, llm=llm, retriever=retriever)

    # Create the workflow graph
    workflow = StateGraph(GlobalState)
    
    # Add nodes
    workflow.add_node("gatekeeper", bound_gatekeeper)
    workflow.add_node("supervisor", bound_supervisor)
    workflow.add_node("ingestion", bound_ingestion)
    workflow.add_node("synthesis", bound_synthesis)

    # Set entry point
    workflow.set_entry_point("gatekeeper")

    # Add conditional edges
    workflow.add_conditional_edges(
        "gatekeeper",
        gatekeeper_router,
        {
            "deep_research": "supervisor",
            "end": END
        }
    )

    # Add regular edges
    workflow.add_edge('supervisor', "ingestion")
    workflow.add_edge('ingestion', "synthesis")

    # Add second conditional edge
    workflow.add_conditional_edges(
        "synthesis",
        routing_logic,
        {
            "continue": "supervisor",
            "end": END
        }
    )

    # Compile and return
    return workflow.compile()
