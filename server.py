from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your compiled graph from your orchestrator file
from orchestrator_1 import app as langgraph_app

# Initialize the API
app = FastAPI()

# CRITICAL: Configure CORS so your frontend is allowed to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your frontend domain later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define what the frontend will send us
class ResearchRequest(BaseModel):
    query: str

# Create the endpoint
@app.post("/api/research")
async def generate_research(request: ResearchRequest):
    print(f"Received query from frontend: {request.query}")
    
    # Pass the frontend's query into your existing state
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
    
    # Run the graph!
    final_state = langgraph_app.invoke(initial_state)
    
    # Send the final drafted report back to the frontend
    return {
        "status": "success",
        "report": final_state["draft_report"],
        "references": final_state["formatted_references"]
    }

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)