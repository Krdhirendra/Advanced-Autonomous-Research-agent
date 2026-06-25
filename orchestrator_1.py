# --------------IMPORTING LIBRARIES--------------- #

import operator
from pathlib import Path
from time import perf_counter
from functools import partial

from typing import List, Dict, Annotated, TypedDict, Any, Literal

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langgraph.graph import StateGraph, END

from markdown_pdf import MarkdownPdf, Section

from tools import load_env_variables
from vector_store import RDSVectorStore
from retriever import Retriever

from agents.gate_keeper import gatekeeper_agent
from agents.supervisor import supervisor_agent
from agents.web_ingestion import web_and_ingestion_agent
from agents.synthesis_n_critic import synthesis_and_critique_agent


# --------------INITIALIZING VARIOUS COMPONENTS--------------- #

load_env_variables(".env")

init_time = perf_counter()
# Loading Embedding Model
try:
    print(">Initializing the Embedding model (HuggingFace Inference API)...")
    embed_model = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
except Exception as e:
    print(f"Error Loading the Embedding model\nError: {e}")


# Initializing Vector store
try:
    print(">Initializing the vector_store")
    rds_wrapper = RDSVectorStore(collection_name='research_paper')
    rds_wrapper.initialize_store(embedding_manager=embed_model)
except Exception as e:
    print(f"Warning: vector_store initialization failed: {e}")
    rds_wrapper = None


# Initializing Retriever
try:
    print(">Initializing the retriever...")
    retriever = Retriever(rds_wrapper=rds_wrapper, all_documents=[], k=5, top_n=5, collection_name='research_paper')
    retriever.initialize_retriever()
except Exception as e:
    print(f"Warning: retriever initialization failed: {e}")
    retriever = None



# Initializing llm
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0
)

# --------------INITIALIZING GLOBAL STATE--------------- #

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


# --------------ROUTING LOGICS--------------- #

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



# Bind the extra arguments (llm, embed_model) to the imported functions
bound_gatekeeper = partial(gatekeeper_agent, llm=llm)
bound_supervisor = partial(supervisor_agent, llm=llm)
bound_ingestion = partial(web_and_ingestion_agent, embed_model=embed_model, retriever=retriever, rds_wrapper=rds_wrapper)
bound_synthesis = partial(synthesis_and_critique_agent, llm=llm, retriever=retriever)


# --------------CREATING GRAPH--------------- #

workflow = StateGraph(GlobalState)
# Add Nodes
workflow.add_node("gatekeeper", bound_gatekeeper)
workflow.add_node("supervisor", bound_supervisor)
workflow.add_node("ingestion", bound_ingestion)
workflow.add_node("synthesis", bound_synthesis)

# Add Entry point
workflow.set_entry_point("gatekeeper")

# Add conditional edge 1
workflow.add_conditional_edges(
    "gatekeeper",
    gatekeeper_router,
    {
        "deep_research": "supervisor", # Matches the router's exact return string
        "end": END                     # Skip everything
    }
)

# Add Edges
workflow.add_edge('supervisor', "ingestion")
workflow.add_edge('ingestion', "synthesis")

# Add conditional edge 2
workflow.add_conditional_edges(
    "synthesis",
    routing_logic, 
    {
        "continue": "supervisor", 
        "end": END 
    }
)

#compile the workflow
app = workflow.compile()


# --------------START--------------- #

# write an detailed report on Embeddings, what is embeddings? How is embedding model trained? How does embeddings work? how does embedding model works? 
# write an detailed report on how to make the retrieval from aws rds vector store faster, and how to make the local HuggingFaceEmbeddings faster


# Input Research Query
research_query = input("-> Please Enter your Research topic.\nTo get an better result please specify the topic.\n=>~") 

# Initialise state
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

# --------------RUNNING THE GRAPH--------------- #
final_state = app.invoke(initial_state)

print("\n=== FINAL REPORT ===")
print(final_state["draft_report"])

# Does Something
final_report = final_state["draft_report"]
if isinstance(final_report, list) and final_report and isinstance(final_report[0], dict) and "text" in final_report[0]:
    final_report = final_report[0]["text"]

# Writing down PDf in proper Markdown formatting
pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(final_report))

# --------------SAVE THE PDF REPORT--------------- #

base_dir = Path.cwd().parent
report_folder = base_dir / "reports"
report_folder.mkdir(parents=True, exist_ok=True)
filename = "my_report5.pdf"
file_path = report_folder / filename
pdf.save(str(file_path))
print(f"Report saved successfully at: {file_path}")

total_time_taken = perf_counter() - init_time
print(f"\nAction completed in {total_time_taken}")