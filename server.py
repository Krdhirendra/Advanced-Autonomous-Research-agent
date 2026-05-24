import uuid
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from markdown_pdf import MarkdownPdf, Section
import json
from fastapi.responses import StreamingResponse
# # Import your compiled graph
from orchestrator_1 import app as langgraph_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Create a dedicated folder for reports
base_dir = Path.cwd()
report_folder = base_dir / "reports"
report_folder.mkdir(parents=True, exist_ok=True)

# 2. Mount the folder so the frontend can access the PDFs via URL
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

class ResearchRequest(BaseModel):
    query: str



@app.post("/api/research")
async def generate_research(request: ResearchRequest):
    print(f"Received query from frontend: {request.query}")
    
    async def event_stream():
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
        # Instead of invoke(), we stream(). This yields an event every time a node finishes.
        final_state = initial_state
        for event in langgraph_app.stream(initial_state):
            for node_name, state_update in event.items():
                # Broadcast the real log to the frontend instantly
                log_message = f"[{node_name.upper()}] Execution completed."
                yield json.dumps({"type": "log", "message": log_message}) + "\n"
                
                # Keep tracking the state so we have the final output at the end
                final_state.update(state_update)

        # 2. FINALIZE THE REPORT
        final_report = final_state.get("draft_report", "")
        query_type = final_state.get("query_type", "deep_research")

        if query_type in ["chit_chat", "general_knowledge", "out_of_scope"]:
            yield json.dumps({
                "type": "result", 
                "response_type": "text", 
                "content": final_report
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
                "pdf_url": f"/reports/{filename}"
            }) + "\n"

    # Return the stream with the specific NDJSON media type
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)