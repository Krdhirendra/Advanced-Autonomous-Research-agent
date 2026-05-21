#import the dependencies
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# V2: PRODUCTION GATEKEEPER AGENT
# ---------------------------------------------------------
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

def gatekeeper_agent(state, llm):
    print("\n=== GATEKEEPER AGENT: Executing Intent Analysis ===")
    original_query = state['original_query']

    system_prompt = """You are the elite semantic routing firewall for an autonomous academic and technical research agent. 
    Analyze the user's prompt and strictly classify it into one of these four categories:

    1. 'deep_research': Complex technical, architectural, analytical, or comparative queries that require pulling fresh data from the web and processing it through a vector database. (e.g., "Analyze thermal mass properties of CSEB vs fired clay", "Explain LangGraph multi-agent architectures").
    2. 'general_knowledge': Standard facts, definitions, or historical data that you already know with 100% confidence. No web search needed. (e.g., "What is the capital of France?", "Define photosynthesis").
    3. 'chit_chat': Standard conversational filler, greetings, or pleasantries. (e.g., "Hello", "Who are you?", "Thanks").
    4. 'out_of_scope': Requests that violate the purpose of a technical research agent, such as creative writing, roleplay, coding whole applications, or giving medical/legal advice.

    If the query is NOT 'deep_research', you MUST provide a complete, polished response in the 'direct_response' field."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {original_query}")
    ]
    
    # Use your fastest/cheapest model configuration here (e.g., gemini-1.5-flash)
    structured_llm = llm.with_structured_output(QueryClassification)
    response = structured_llm.invoke(messages)

    classification = response.classification
    print(f"-> Gatekeeper Reasoning: {response.reasoning}")

    if classification in ["general_knowledge", "chit_chat", "out_of_scope"]:
        print(f"-> Gatekeeper Action: Classified as '{classification}'. Terminating graph and returning instant response.")
        return {
            "query_type": classification,
            "draft_report": response.direct_response,
            "is_complete": True
        }
    else:
        print("-> Gatekeeper Action: Valid complex research topic detected. Passing to Supervisor.")
        return {"query_type": "deep_research"}

# Updated Router Logic
