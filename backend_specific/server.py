import uuid
import json
import time
from time import perf_counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from markdown_pdf import MarkdownPdf, Section

import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from tools import load_env_variables
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import RDSVectorStore
from retriever import Retriever
from orchestrator_2 import create_orchestrator

# Load environment variables at module load time
load_env_variables(".env")

# Global variables to hold initialized components
embed_model = None
retriever = None
rds_wrapper = None
llm = None
langgraph_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize components once at server startup
    and clean up on shutdown.
    """
    global embed_model, retriever, rds_wrapper, llm, langgraph_app
    
    # --------- STARTUP ---------
    print("\n========== SERVER STARTUP ==========")
    
    # Initialize Embedding Model
    try:
        print("> Initializing the Embedding model (HuggingFace Inference API)...")
        embed_model = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-small-en-v1.5",
            huggingfacehub_api_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        print("✓ Embedding model initialized")
    except Exception as e:
        print(f"✗ Error Loading the Embedding model: {e}")
        raise

    # Initialize Vector Store
    try:
        print(">Initializing the vector_store...")
        rds_wrapper = RDSVectorStore(collection_name='research_paper')
        rds_wrapper.initialize_store(embedding_manager=embed_model)
        print("✓ Vector store initialized")
    except Exception as e:
        print(f"✗ Vector store initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Initialize Retriever
    try:
        print(">Initializing the retriever...")
        retriever = Retriever(
            rds_wrapper=rds_wrapper,
            all_documents=[],
            k=5,
            top_n=5,
            collection_name='research_paper'
        )
        retriever.initialize_retriever()
        print("✓ Retriever initialized")
    except Exception as e:
        print(f"✗ Retriever initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Initialize LLM
    try:
        print(">Initializing the LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0
        )
        print("✓ LLM initialized")
    except Exception as e:
        print(f"✗ LLM initialization failed: {e}")
        raise

    # Create the orchestrator with pre-initialized components
    try:
        print(">Creating orchestrator...")
        langgraph_app = create_orchestrator(embed_model, retriever, rds_wrapper, llm)
        print("✓ Orchestrator created")
    except Exception as e:
        print(f"✗ Orchestrator creation failed: {e}")
        raise

    print("========== SERVER READY ==========\n")
    
    yield  # Server runs here
    
    # --------- SHUTDOWN ---------
    print("\n========== SERVER SHUTDOWN ==========")
    # Add cleanup logic here if needed
    print("========== GOODBYE ==========\n")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a dedicated folder for reports
base_dir = Path.cwd()
report_folder = base_dir / "reports"
report_folder.mkdir(parents=True, exist_ok=True)

# Mount the folder so the frontend can access the PDFs via URL
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


# -------------- REQUEST MODELS --------------- #

class ResearchRequest(BaseModel):
    query: str


# -------------- API ENDPOINTS --------------- #

@app.post("/api/research")
async def generate_research(request: ResearchRequest):
    init_time = perf_counter()
    print(f"\n{'='*60}")
    print(f"Received query from frontend: {request.query}")
    print(f"{'='*60}\n")
    
    async def event_stream():
        try:
            start_time = time.time()
            initial_state = {
                "original_query": request.query,
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
            
            # 1. STREAM REAL-TIME LOGS
            final_state = initial_state
            for event in langgraph_app.stream(initial_state):
                for node_name, state_update in event.items():
                    # Broadcast the real log to the frontend instantly
                    log_message = f"[{node_name.upper()}] Execution completed."
                    yield json.dumps({"type": "log", "message": log_message}) + "\n"
                    
                    # Keep tracking the state so we have the final output at the end
                    final_state.update(state_update)
            
            end_time = time.time()
            execution_time = round(end_time - start_time, 2)
            
            # 2. FINALIZE THE REPORT
            final_report = final_state.get("draft_report", "")
            query_type = final_state.get("query_type", "deep_research")

            # Handle different report formats
            if isinstance(final_report, list):
                # Handle LangChain dictionary list format: [{'text': '...'}]
                if len(final_report) > 0 and isinstance(final_report[0], dict) and "text" in final_report[0]:
                    final_report = final_report[0]["text"]
                else:
                    # Fallback: join the list items into a single string
                    final_report = "\n".join(str(item) for item in final_report)
            elif not isinstance(final_report, str):
                # Catch-all for any other weird data types
                final_report = str(final_report)

            # 3. ROUTE RESPONSE BY QUERY TYPE
            if query_type in ["chit_chat", "general_knowledge", "out_of_scope"]:
                yield json.dumps({
                    "type": "result", 
                    "response_type": "text", 
                    "content": final_report,
                    "time": execution_time
                }) + "\n"
            else:
                filename = f"AARA_Report_{uuid.uuid4().hex[:6]}.pdf"
                file_path = report_folder / filename

                pdf = MarkdownPdf(toc_level=2)
                pdf.add_section(Section(final_report))
                pdf.save(str(file_path))
                
                yield json.dumps({
                    "type": "result", 
                    "response_type": "pdf", 
                    "pdf_url": f"/reports/{filename}",
                    "time": execution_time
                }) + "\n"
        
        except Exception as e:
            print(f"Error in research pipeline: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"

    # Return the stream with the specific NDJSON media type
    total_time_taken = perf_counter() - init_time
    print(f"\nAction completed in {total_time_taken}")
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)