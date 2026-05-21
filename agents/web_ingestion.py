from langchain_core.documents import Document as LangChainDoc

from A_ARA.agents.prompts import *
from tools import tavily_search, extract_text
from RAG import chunks
from vector_store import RDSVectorStore

def web_and_ingestion_agent(state: dict, embed_model):
    """Executes searches, scrapes, chunks (LlamaIndex), and stores in Vector DB."""
    print("=== INGESTION AGENT: Searching and RAG processing...")

    queries = state['search_queries']

    links, response = tavily_search(queries['searches'], max_results=3)

    valid_docs = []
    failed_urls = []

    for link in links:
        extracted_data = extract_text(link)
        if extracted_data and extracted_data.get('page_content'):
            doc = LangChainDoc(
                page_content=extracted_data['page_content'],
                metadata=extracted_data['metadata']
            )
            valid_docs.append(doc)
        else:
            failed_urls.append(link)

    if not valid_docs:
        print("-> No valid documents extracted. Routing back.")
        return {
            "gathered_chunks": [], 
            "failed_urls": failed_urls,
            "missing_information": []
        }
    
    rds_wrapper = RDSVectorStore(collection_name='research_paper')
    rds_wrapper.initialize_store(embedding_manager=embed_model)

    chunker = chunks(vector_store=rds_wrapper.vectorstore)
    retriever = chunker.split_texts()

    print(f"Pushing {len(valid_docs)} source documents through the chunking pipeline...")
    retriever.add_documents(valid_docs)
    print("documents chunked successfully...")

    return {
        "gathered_chunks": valid_docs, 
        "failed_urls": failed_urls,
        "missing_information": []
    }