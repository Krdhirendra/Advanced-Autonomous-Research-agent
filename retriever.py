from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from vector_store import RDSVectorStore
# from RAG import cohere_embed # Assuming your embedding model is imported

def advanced_hybrid_retrieval(query: str, all_documents: list, embedding_manager):
    """
    Executes Hybrid Search (Vector + BM25) and reranks with a Cross-Encoder.
    
    Note on BM25: LangChain's BM25 is in-memory. In a massive production system, 
    you would use Postgres Full Text Search, but for this architecture, 
    we build the BM25 index on the fly from the fetched/available documents.
    """
    print(f"=== EXECUTING HYBRID RETRIEVAL & RERANKING FOR: '{query}' ===")

    # 1. Initialize AWS Dense Vector Retriever
    rds_wrapper = RDSVectorStore()
    rds_wrapper.initialize_store(embedding_manager=embedding_manager)
    
    # Set to fetch a wider net of results (e.g., top 10) because the reranker will filter them down
    vector_retriever = rds_wrapper.vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Initialize BM25 Keyword Retriever
    # It requires the actual text to build its keyword frequency map
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 10 

    # 3. Combine them using Reciprocal Rank Fusion (RRF)
    # weights=[0.5, 0.5] means we care equally about exact keywords and general semantics
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )

    # 4. Initialize the Cross-Encoder (The Judge)
    # BAAI/bge-reranker-base is currently one of the best open-source, lightweight rerankers
    print("-> Initializing Cross-Encoder...")
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    # Tell the reranker to only pass the absolute best 5 documents to the Synthesis Agent
    reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=5)

    # 5. Wrap the Hybrid Retriever in the Compression (Reranking) pipeline
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=hybrid_retriever
    )

    # 6. Execute the final, hyper-accurate search
    print("-> Fetching, Fusing, and Reranking documents...")
    final_docs = compression_retriever.invoke(query)
    
    return final_docs