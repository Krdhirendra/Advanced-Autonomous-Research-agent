# --------------IMPORTING LIBRARIES--------------- #

from time import perf_counter
from langchain_core.documents import Document as LangChainDoc
import concurrent.futures

from agents.prompts import *
from RAG import chunks
from tools import tavily_search, extract_text




# --------------WEB INGESTION AGENT--------------- #

def web_and_ingestion_agent(state: dict, embed_model, rds_wrapper, retriever=None):
    """Executes searches, scrapes, chunks (LlamaIndex), and stores in Vector DB.

    This version respects the `preferred_tool` field produced by the Supervisor.
    - `web_search` uses Tavily (web)
    - `local_retrieve` uses the pre-initialized `retriever`
    """
    print("=== INGESTION AGENT: Searching and RAG processing ===")


    # fetching state
    queries = state.get('search_queries', {}).get('searches', [])


    # Split strategies by preferred tool
    web_strategies = [s for s in queries if s.get('preferred_tool') == 'web_search']
    local_strategies = [s for s in queries if s.get('preferred_tool') == 'local_retrieve']

    valid_docs = []
    failed_urls = []


    # 1) Handle local retrieval strategies (fast, uses pre-built retriever)
    if local_strategies:
        if retriever is None:
            print("> Warning: local_retrieve requested but no retriever available. Skipping local retrieval.")
        else:
            for strat in local_strategies:
                q = strat.get('query')
                try:
                    print(f"-> Local retrieval for: {q}")
                    docs = retriever.advanced_hybrid_retrieval(q)
                    for d in docs:
                        # Expect LangChain Document-like objects
                        if hasattr(d, 'page_content'):
                            valid_docs.append(d)
                except Exception as e:
                    print(f"> Local retrieval failed for {q}: {e}")


    # 2) Handle web search strategies
    if web_strategies:
        links, response = tavily_search(web_strategies, max_results=3)

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
        print("> No valid documents extracted. Routing back.")
        return {
            "gathered_chunks": [],
            "failed_urls": failed_urls,
            "missing_information": []
        }

    # Chunk the documents
    chnk_strt = perf_counter()
    chunker = chunks(vector_store=rds_wrapper.vectorstore)
    print(f"> Pushing {len(valid_docs)} source documents through the chunking pipeline...")
    chunkss = chunker.split_texts().child_splitter.split_documents(valid_docs)
    chnk_end = perf_counter()
    print(f"> Chunking Completed in {chnk_end - chnk_strt}sec")

    
    # Embed the chunks
    embd_strt = perf_counter()

    print("> Starting hosted HuggingFace API vector embedding...")
    texts_to_embed = [doc.page_content for doc in chunkss]
    embeddings = embed_model.embed_documents(texts_to_embed)
    print(f"Successfully generated {len(embeddings)} vectors!")
    
    embd_end = perf_counter()
    print(f"Embedding Completed in {embd_end - embd_strt}")

    # Upload the Embeddings
    print("> Starting AWS RDS parallel upload...")

    # Bundle our data for clean batching
    all_texts = texts_to_embed
    all_embeddings = embeddings
    all_metadatas = [doc.metadata for doc in chunkss]

    # We will upload 200 chunks per network request
    batch_size = 200
    batches = []

    for i in range(0, len(all_texts), batch_size):
        batches.append({
            "texts": all_texts[i: i + batch_size],
            "embeddings": all_embeddings[i: i + batch_size],
            "metadatas": all_metadatas[i: i + batch_size]
        })


    # Define the worker function that actually talks to AWS
    def upload_batch_to_aws(batch_data):
        rds_wrapper.vectorstore.add_embeddings(
            texts=batch_data["texts"],
            embeddings=batch_data["embeddings"],
            metadatas=batch_data["metadatas"]
        )

    print(f"> Firing {len(batches)} simultaneous upload requests to AWS...")
    # Open 10 concurrent network pipelines to AWS and blast the data across all of them at once
    upld_strt = perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(upload_batch_to_aws, batches)
    upld_end = perf_counter()
    print(f"> Uploading completed in {upld_end-upld_strt}")

    return {
        "gathered_chunks": valid_docs,
        "failed_urls": failed_urls,
        "missing_information": []
    }