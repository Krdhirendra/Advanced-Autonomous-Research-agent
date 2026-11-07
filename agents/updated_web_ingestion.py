import asyncio
from time import perf_counter

from langchain_core.documents import Document as LangChainDoc

from agents.prompts import *  # existing project import
from RAG import chunks
from tools import tavily_search, extract_text


async def web_and_ingestion_agent(state: dict, embed_model, rds_wrapper, retriever=None):
    """Async version of web ingestion.

    Uses async/await for I/O-bound work:
    - web search
    - page extraction
    - local retrieval
    - batch upload to RDS
    """
    print("=== INGESTION AGENT: Searching and RAG processing (async) ===")

    queries = state.get("search_queries", {}).get("searches", [])
    web_strategies = [s for s in queries if s.get("preferred_tool") == "web_search"]
    local_strategies = [s for s in queries if s.get("preferred_tool") == "local_retrieve"]

    valid_docs = []
    failed_urls = []

    # 1) Local retrieval
    if local_strategies:
        if retriever is None:
            print("> Warning: local_retrieve requested but no retriever available. Skipping local retrieval.")
        else:
            async def run_local(strat):
                q = strat.get("query")
                try:
                    print(f"-> Local retrieval for: {q}")
                    docs = await asyncio.to_thread(retriever.advanced_hybrid_retrieval, q)
                    return [d for d in docs if hasattr(d, "page_content")]
                except Exception as e:
                    print(f"> Local retrieval failed for {q}: {e}")
                    return []

            local_results = await asyncio.gather(*(run_local(s) for s in local_strategies))
            for docs in local_results:
                valid_docs.extend(docs)

    # 2) Web search
    if web_strategies:
        links, response = await asyncio.to_thread(tavily_search, web_strategies, 3)

        async def extract_link(link):
            try:
                extracted_data = await asyncio.to_thread(extract_text, link)
                if extracted_data and extracted_data.get("page_content"):
                    return LangChainDoc(
                        page_content=extracted_data["page_content"],
                        metadata=extracted_data.get("metadata", {}),
                    ), None
            except Exception as e:
                print(f"> Extraction failed for {link}: {e}")
            return None, link

        results = await asyncio.gather(*(extract_link(link) for link in links))
        for doc, failed in results:
            if doc is not None:
                valid_docs.append(doc)
            elif failed is not None:
                failed_urls.append(failed)

    if not valid_docs:
        print("> No valid documents extracted. Routing back.")
        return {
            "gathered_chunks": [],
            "failed_urls": failed_urls,
            "missing_information": [],
        }

    # Chunking
    chnk_strt = perf_counter()
    chunker = chunks(vector_store=rds_wrapper.vectorstore)
    print(f"> Pushing {len(valid_docs)} source documents through the chunking pipeline...")
    chunkss = chunker.split_texts().child_splitter.split_documents(valid_docs)
    chnk_end = perf_counter()
    print(f"> Chunking completed in {chnk_end - chnk_strt} sec")

    # Embedding
    embd_strt = perf_counter()
    print("> Starting hosted HuggingFace API vector embedding...")
    texts_to_embed = [doc.page_content for doc in chunkss]
    embeddings = await asyncio.to_thread(embed_model.embed_documents, texts_to_embed)
    print(f"Successfully generated {len(embeddings)} vectors!")
    embd_end = perf_counter()
    print(f"Embedding completed in {embd_end - embd_strt} sec")

    # Upload embeddings
    print("> Starting AWS RDS async upload...")
    all_texts = texts_to_embed
    all_embeddings = embeddings
    all_metadatas = [doc.metadata for doc in chunkss]

    batch_size = 200
    batches = [
        {
            "texts": all_texts[i: i + batch_size],
            "embeddings": all_embeddings[i: i + batch_size],
            "metadatas": all_metadatas[i: i + batch_size],
        }
        for i in range(0, len(all_texts), batch_size)
    ]

    async def upload_batch_to_aws(batch_data):
        await asyncio.to_thread(
            rds_wrapper.vectorstore.add_embeddings,
            texts=batch_data["texts"],
            embeddings=batch_data["embeddings"],
            metadatas=batch_data["metadatas"],
        )

    print(f"> Firing {len(batches)} simultaneous upload requests to AWS...")
    upld_strt = perf_counter()
    await asyncio.gather(*(upload_batch_to_aws(batch) for batch in batches))
    upld_end = perf_counter()
    print(f"> Uploading completed in {upld_end - upld_strt} sec")

    return {
        "gathered_chunks": valid_docs,
        "failed_urls": failed_urls,
        "missing_information": [],
    }