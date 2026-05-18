from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
import os


class RDSVectorStore:
    def __init__(self, collection_name: str = "research_papers"):
        self.collection_name = collection_name
        
        # AWS RDS Connection String
        self.connection_string = os.environ.get("AWS_RDS_URL") 
        self.vectorstore = None

    def initialize_store(self, embedding_manager):
        """Connects to AWS RDS and initializes the pgvector table"""
        print("Connecting to AWS RDS pgvector...")
        
        self.vectorstore = PGVector(
            connection_string=self.connection_string,
            embedding_function=embedding_manager, 
            collection_name=self.collection_name,
            use_jsonb=True # Stores metadata efficiently
        )

    def add_documents(self, documents: list[Document]):
        """Adds LangChain documents directly to the AWS database"""
        if not self.vectorstore:
            raise ValueError("Store not initialized. Call initialize_store first.")
            
        print(f"Pushing {len(documents)} chunks to AWS RDS...")
        self.vectorstore.add_documents(documents)