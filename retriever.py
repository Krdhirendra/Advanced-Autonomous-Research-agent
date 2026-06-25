# -------------- IMPORTING LIBRARIES --------------- #

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from vector_store import RDSVectorStore


# -------------- RETRIEVER CLASS --------------- #

class Retriever:
    def __init__(self, rds_wrapper, all_documents, k:int, top_n:int, collection_name: str = "research_paper"):
        self.all_documents = all_documents
        self.k = k
        self.top_n = top_n
        self.collection_name = collection_name
        self.rds_wrapper = rds_wrapper
        
        # Initialize placeholders
        self.vector_retriever = None
        self.bm25_retriever = None
        self.reranker = None
        self.hybrid_retriever = None
        self.compression_retriever = None


    def _build_retrieval_stack(self):
        """Build the Retriever"""
        if self.bm25_retriever is not None:
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[0.5, 0.5],
            )
        else:
            self.hybrid_retriever = self.vector_retriever

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.hybrid_retriever,
        )


    def update_documents(self, documents):
        """Refresh the BM25 side once chunk documents exist"""

        self.all_documents = documents or []

        if not self.all_documents:
            self.bm25_retriever = None
            self._build_retrieval_stack()
            return

        self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
        self.bm25_retriever.k = self.k
        self._build_retrieval_stack()
        

    def initialize_retriever(self):
        """Initialize all retrieval components once at startup."""

        self.vector_retriever = self.rds_wrapper.vectorstore.as_retriever(search_kwargs={"k": self.k})

        # Initialize BM25 only when documents already exist.
        if self.all_documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
            self.bm25_retriever.k = self.k
        else:
            self.bm25_retriever = None

        # Initialize the Cross-Encoder (The Judge)
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=self.top_n)

        self._build_retrieval_stack()


    def advanced_hybrid_retrieval(self, query: str):
        """
        Executes Hybrid Search (Vector + BM25) and reranks with a Cross-Encoder.
        Uses pre-initialized retrievers for minimal latency.
        """
        
        if self.compression_retriever is None:
            raise RuntimeError("Retriever not initialized. Call initialize_retriever() first.")
        
        print(f"=== EXECUTING HYBRID RETRIEVAL & RERANKING FOR: '{query}' ===")

        # Execute the final, hyper-accurate search using pre-built retrievers
        print("-> Fetching, Fusing, and Reranking documents...")
        final_docs = self.compression_retriever.invoke(query)
        
        return final_docs