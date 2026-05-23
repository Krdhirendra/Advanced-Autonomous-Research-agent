from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()




class RDSVectorStore:
	def __init__(self, collection_name: str = "research_papers"):
		self.collection_name = collection_name
		# Read and normalize the connection string from environment
		self.connection_string = os.getenv("AWS_RDS_URI")
		self.vectorstore = None

	def initialize_store(self, embedding_manager):
		"""Connects to AWS RDS pgvector via SQLAlchemy URL.

		Raises a helpful error if the environment variable is missing.
		"""
		if not self.connection_string:
			raise ValueError(
				"AWS_RDS_URI is not set. Add a valid SQLAlchemy URL to your .env, e.g.\n"
				"AWS_RDS_URI=postgresql+psycopg2://user:pass@host:5432/database"
			)


		print("Connecting to AWS RDS pgvector...")
		self.vectorstore = PGVector(
			connection_string=self.connection_string,
			embedding_function=embedding_manager,
			collection_name=self.collection_name,
			use_jsonb=True,
		)

	def add_documents(self, documents: list[Document]):
		"""Adds LangChain documents directly to the AWS database"""
		if not self.vectorstore:
			raise ValueError("Store not initialized. Call initialize_store first.")

		print(f"Pushing {len(documents)} chunks to AWS RDS...")
		self.vectorstore.add_documents(documents,chunk_size=1000)
