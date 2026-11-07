import os
import chromadb
from typing import Dict, List, Any



class chromavectorstore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection_name = None
        self._initialise_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection_name = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "document embedding for RAG"}
            )

        except Exception as e:
            print(f"Error Initializing the vector store: {e}")
            
        
    def add_documents(self,
                      documents: List[Any],
                      embeddings: np.array):
        
        if len

