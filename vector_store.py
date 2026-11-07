# -------------- IMPORTING LIBRARIES --------------- #

import os
import uuid
import threading
from tools import load_env_variables
from langchain_core.documents import Document
from langchain_chroma import Chroma


# -------------- VECTOR STORE CLASS --------------- #

load_env_variables()


class ChromaVectorStore:
	"""Persistent local ChromaDB vector store (replaces the AWS RDS pgvector store).

	Keeps the same surface the rest of the pipeline relies on:
	`.vectorstore` (a LangChain vector store usable with `.as_retriever()`),
	`initialize_store(embedding_manager)` and `add_documents(...)`.
	"""

	def __init__(self, collection_name: str = "research_papers", persist_directory: str | None = None):
		self.collection_name = collection_name
		self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
		self.vectorstore = None
		self._lock = threading.Lock()

	def initialize_store(self, embedding_manager):
		"""Opens (or creates) the persistent Chroma collection on disk."""
		os.makedirs(self.persist_directory, exist_ok=True)
		print(f"Connecting to ChromaDB at '{self.persist_directory}'...")
		self.vectorstore = Chroma(
			collection_name=self.collection_name,
			embedding_function=embedding_manager,
			persist_directory=self.persist_directory,
		)

	def add_documents(self, documents: list[Document]):
		"""Adds LangChain documents (embedded with the store's embedding function)."""
		if not self.vectorstore:
			raise ValueError("Store not initialized. Call initialize_store first.")

		print(f"Pushing {len(documents)} chunks to ChromaDB...")
		with self._lock:
			self.vectorstore.add_documents(documents)

	def add_embeddings(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict] | None = None):
		"""Adds pre-computed embeddings (used by the ingestion agent to avoid re-embedding)."""
		if not self.vectorstore:
			raise ValueError("Store not initialized. Call initialize_store first.")

		# Chroma metadata values must be scalars and non-empty dicts are required
		clean_metas = [
			{k: v for k, v in (m or {}).items() if isinstance(v, (str, int, float, bool))} or {"source": "unknown"}
			for m in (metadatas or [{}] * len(texts))
		]
		ids = [uuid.uuid4().hex for _ in texts]
		with self._lock:
			self.vectorstore._collection.upsert(
				ids=ids, documents=texts, embeddings=embeddings, metadatas=clean_metas
			)


# Backwards-compatible alias so existing imports keep working
RDSVectorStore = ChromaVectorStore
