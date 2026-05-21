from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding

class chunks:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        # 1. Load weights locally to completely bypass network latency
        # print("-> Loading embedding model into local RAM...")
        # local_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        # self.embed_engine = LangchainEmbedding(local_model)

    def split_texts(self):
        
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

        # 3. Build the retriever
        retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=InMemoryStore(),
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )
        return retriever