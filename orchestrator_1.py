#import the dependencies
import os
import json
import operator
import importlib
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import List, Dict, Annotated, TypedDict, Any, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document as LangChainDoc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from markdown_pdf import MarkdownPdf, Section
from pathlib import Path
from functools import partial

# from prompts import *
from tools import tavily_search, extract_text
from RAG import chunks
from vector_store import RDSVectorStore
from retriever import advanced_hybrid_retrieval # Import your hybrid search
from agents.gate_keeper import gatekeeper_agent
from agents.supervisor import supervisor_agent
from agents.web_ingestion import web_and_ingestion_agent
from agents.synthesis_n_critic import synthesis_and_critique_agent

load_dotenv(".env")



# Our Embedding Model
try:
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
except Exception as e:
    print(f"Error Loading the Embedding model\nError: {e}")


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0
)


#Global state
class GlobalState(TypedDict):
    original_query: str
    query_type: str
    search_queries: dict
    gathered_chunks: List[Any]
    failed_urls: Annotated[List[str],operator.add]
    missing_information: List[str]
    draft_report: str
    is_complete: bool
    formatted_references: List[dict]
    iteration_count: int
    max_iterations: int

research_query = input("-> Please Enter your Research topic.\n\tTo get an better result please specify the topic.\n\t=>~")

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

workflow = StateGraph(GlobalState)

# 1. Bind the extra arguments (llm, embed_model) to the imported functions
bound_gatekeeper = partial(gatekeeper_agent, llm=llm)
bound_supervisor = partial(supervisor_agent, llm=llm)
bound_ingestion = partial(web_and_ingestion_agent, embed_model=embed_model)
bound_synthesis = partial(synthesis_and_critique_agent, embed_model=embed_model, llm=llm)

# 2. Add nodes using the bound functions (NO parentheses!)
workflow.add_node("gatekeeper", bound_gatekeeper)
workflow.add_node("supervisor", bound_supervisor)
workflow.add_node("ingestion", bound_ingestion)
workflow.add_node("synthesis", bound_synthesis)

# 3. Add Entry point
workflow.set_entry_point("gatekeeper")

# 4. Add conditional edges (FIXED: "deep_research" string match)
workflow.add_conditional_edges(
    "gatekeeper",
    gatekeeper_router,
    {
        "deep_research": "supervisor", # Matches the router's exact return string
        "end": END                     # Skip everything
    }
)

workflow.add_edge('supervisor', "ingestion")
workflow.add_edge('ingestion', "synthesis")

workflow.add_conditional_edges(
    "synthesis",
    routing_logic, 
    {
        "continue": "supervisor", 
        "end": END 
    }
)

app = workflow.compile()


# Explicitly pass the correct structural string



initial_state = {
    "original_query": research_query,
    "search_queries": {},
    "query_type": "",
    "gathered_chunks": [],
    "failed_urls": [],
    "missing_information": [],
    "draft_report": "",
    "is_complete": False,
    "formatted_references": [],
    "iteration_count": 0,
    "max_iterations": 3,
}

final_state = app.invoke(initial_state)
print("\n=== FINAL REPORT ===")
print(final_state["draft_report"])


final_report = final_state["draft_report"]
if isinstance(final_report, list) and final_report and isinstance(final_report[0], dict) and "text" in final_report[0]:
    final_report = final_report[0]["text"]


pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(final_report))



base_dir = Path.cwd().parent
report_folder = base_dir / "reports"
report_folder.mkdir(parents=True, exist_ok=True)
filename = "my_report4.pdf"
file_path = report_folder / filename
pdf.save(str(file_path))
print(f"Report saved successfully at: {file_path}")