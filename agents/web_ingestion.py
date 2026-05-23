from langchain_core.documents import Document as LangChainDoc
from time import perf_counter
import concurrent.futures
from agents.prompts import *
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
    print(f"Pushing {len(valid_docs)} source documents through the chunking pipeline...")
    chunkss = chunker.split_texts().child_splitter.split_documents(valid_docs)
    print(f"generated {len(chunkss)} chunks for {len(valid_docs)} documents /")

    
    # 2. TIME THE EMBEDDING
    print("-> Starting local vector embedding...")
    start_embed = perf_counter()

    texts_to_embed = [doc.page_content for doc in chunkss]
    embeddings = embed_model.embed_documents(texts_to_embed)
    embed_time = perf_counter() - start_embed
    print(f"-> Embedding complete in {embed_time:.2f} seconds.")



    

    # 3. TIME THE DATABASE UPLOAD (PARALLELIZED)
    print("-> Starting AWS RDS parallel upload...")
    start_db = perf_counter()
    
    # Bundle our data for clean batching
    all_texts = texts_to_embed
    all_embeddings = embeddings
    all_metadatas = [doc.metadata for doc in chunkss]
    
    # We will upload 200 chunks per network request
    batch_size = 200
    batches = []
    
    for i in range(0, len(all_texts), batch_size):
        batches.append({
            "texts": all_texts[i : i + batch_size],
            "embeddings": all_embeddings[i : i + batch_size],
            "metadatas": all_metadatas[i : i + batch_size]
        })

    print(f"-> Firing {len(batches)} simultaneous upload requests to AWS...")

    # Define the worker function that actually talks to AWS
    def upload_batch_to_aws(batch_data):
        rds_wrapper.vectorstore.add_embeddings(
            texts=batch_data["texts"],
            embeddings=batch_data["embeddings"],
            metadatas=batch_data["metadatas"]
        )

    # Open 10 concurrent network pipelines to AWS and blast the data across all of them at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(upload_batch_to_aws, batches)
        
    db_time = perf_counter() - start_db
    print(f"-> AWS Database sync complete in {db_time:.2f} seconds.")
    

    return {
        "gathered_chunks": valid_docs, 
        "failed_urls": failed_urls,
        "missing_information": []
    }