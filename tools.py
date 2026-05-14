import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import ArxivRetriever


def load_env_variables(env_file=".env"):
    """Load Environment VAriables"""
    if not os.path.exists(env_file):
        print(f"\n==== Warning: {env_file} not found ====")
        return
    
    load_dotenv(env_file)
    print(f"=> Environment variables loaded from {env_file}")


def tavily_search(searches:list, max_results:int) -> set:
    """Tavily search for aquiring information from internet"""
    print('=> Using Tavily to Search Internet for relevant information....\n')
    links = set()

    for search in searches:
        query = search['query']
        tgt_domains = search['domains',[]]
        
        if not tgt_domains:
            print(f'=> Executing Open WEB Search for {query}....')
            tavily_tool = TavilySearchResults(max_results=3, search_depth="advanced", )
        else:
            print(f"=> Executing Targeted search for {query} from {tgt_domains} ....")
            tavily_tool = TavilySearchResults(max_results=3, search_depth="advanced", include_domains=tgt_domains)
        
        response = tavily_tool.invoke(query)
        
        for res in response:
            links.add(res['url'])

    return links


def arxiv_search(queries:list) -> set:

    retriever = ArxivRetriever(load_max_docs=2,get_full_documents=True)
    extracted_papers = []

    for query in queries:
        print("=> Executing Arxiv search for Academic Papers....")
        docs = retriever.invoke(query)

        for doc in docs:
            paper_data = {
                "source": doc.metadata.get('Entry ID','Unknown ArXiv ID'),
                "title": doc.metadata.get('Title', 'Unknown_title'),
                "content":doc.page_content
            }
            extracted_papers.append(paper_data)

    return extracted_papers

    
    
