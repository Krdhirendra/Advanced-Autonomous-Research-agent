import os
import numpy as np
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document as LangChainDoc
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from llama_index.core import Document as LlamaIndexDoc
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_cohere import CohereEmbeddings




# Wrapper class
"""since the ParentDocumentRetriever expects a child splitter that implements a .split_documents() method. LlamaIndex uses a .get_nodes_from_documents() method. SO a AttributeError will arise if SemanticSplitterNodeParser used directly into ParentDocumentRetriever"""

__all__ = ['chunks']

class Llama_SemanticSplitterWrapper(RecursiveCharacterTextSplitter):
    """Wraps LlamaIndex's semantic splitter to work with LangChain's retriever"""
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
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                final_docs.append(LangChainDoc(page_content=chunk,
                                               metadata=doc.metadata.copy()))
        return final_docs
    


class chunks:

    def __init__(self, vector_store, embed_model:CohereEmbeddings):
        self.embed_model = embed_model
        self.vector_store = vector_store

    def split_texts(self, documents:List[LangChainDoc], buffer_size:int=1, breakpoint_percentile_threshold=80):
        print(f'=> chunks split_texts in action....\n')

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)

        llama_semantic_parser = SemanticSplitterNodeParser(
            embed_model= self.embed_model,
            buffer_size = buffer_size,
            breakpoint_percentile_threshold = breakpoint_percentile_threshold)

        child_splitter = Llama_SemanticSplitterWrapper(llama_splitter=llama_semantic_parser)

        vector_store = self.vector_store
        doc_store = InMemoryDocstore()

        retriver = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=doc_store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )

        return retriver
    