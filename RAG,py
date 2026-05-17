import os
from dotenv import load_dotenv
from typing import List
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document as LangChainDoc
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from llama_index.core import Document as LlamaIndexDoc
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter


load_dotenv('.env')
COHERE_API_KEY = os.environ['COHERE_API_KEY']

"""Our Embedding Model"""
try:
    cohere_embed = CohereEmbeddings(cohere_api_key=COHERE_API_KEY,
                                    model='model="embed-english-v3.0"')
except Exception as e:
    print(f"Error Loading the Embedding model\nError: {e}")


# Wrapper class
"""since the ParentDocumentRetriever expects a child splitter that implements a .split_documents() method. LlamaIndex uses a .get_nodes_from_documents() method. SO a AttributeError will arise if SemanticSplitterNodeParser used directly into ParentDocumentRetriever"""

__all__ = ['chunks', 'EmbeddingManager']

class Llama_SemanticSplitterWrapper(RecursiveCharacterTextSplitter):
    
    def __init__(self, llama_splitter: SemanticSplitterNodeParser):
        super().__init__()
        self.llama_splitter = llama_splitter

    def split_text(self, text:str)->List[str]:
        llama_doc = LlamaIndexDoc(text=text)
        nodes = self.llama_splitter.get_nodes_from_documents([llama_doc])

        return [node.get_content() for node in nodes]
    
    def split_documents(self, documents:List[LangChainDoc])->List[LangChainDoc]:
        final_docs = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for chunk in chunks:
                final_docs.append(LangChainDoc(page_content=chunk,                                               metadata=doc.metadata.copy()))
        return final_docs
    


class chunks:

    def __init__(self, embed_model:CohereEmbeddings=cohere_embed):
        self.embed_model = embed_model

    def split_texts(self, texts:str, buffer_size:int=1, breakpoint_percentile_threshold=80):

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)

        llama_semantic_parser = Llama_SemanticSplitterWrapper(
            embed_model= self.embed_model,
            buffer_size = buffer_size,
            breakpoint_percentile_threshold = breakpoint_percentile_threshold)

        child_splitter = Llama_SemanticSplitterWrapper(llama_splitter=llama_semantic_parser)

        vector_store = None #incoming
        doc_store = InMemoryDocstore()

        retriver = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=doc_store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )

        return retriver
    


class EmbeddingManager:
    """Initialise the embedding manager"""
    def __init__(self, model=cohere_embed):
        self.model = model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text"""
        if not self.model:
            raise ValueError("Embedding model not loaded")
        
        print(f"Generating Embedding for {len(texts)} texts....")
        if isinstance(texts[0], str):
            texts = texts
        else:
            texts = [doc for doc in texts]

        embeddings = self.model.encode(texts)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
        

        