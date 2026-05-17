import re
import os
import requests
from io import BytesIO
from pypdf import PdfReader
from bs4 import BeautifulSoup
from readability import Document
from dotenv import load_dotenv
from langchain_community.retrievers import ArxivRetriever
from langchain_community.tools.tavily_search import TavilySearchResults



def load_env_variables(env_file=".env"):
    """Load Environment VAriables"""
    if not os.path.exists(env_file):
        print(f"\n!!!! Warning: {env_file} not found !!!!")
        return
    
    load_dotenv(env_file)
    print(f"=> Environment variables loaded from {env_file}")


def tavily_search(searches:list, max_results:int=3) -> set:
    """Tavily search for aquiring information from internet"""
    print('=> Using Tavily to Search Internet for relevant information....\n')
    links = set()

    for search in searches:
        query = search['query']
        tgt_domains = search.get('target_domains',[])
        exclude_domains = search.get('exclude_domains',[])
        
        if not tgt_domains:
            print(f'=> Executing Open WEB Search for {query}....')
            tavily_tool = TavilySearchResults(max_results=max_results, search_depth="advanced", )
        else:
            print(f"=> Executing Targeted search for {query} from {tgt_domains} ....")
            tavily_tool = TavilySearchResults(max_results=max_results, search_depth="advanced", include_domains=tgt_domains,exclude_domains=exclude_domains)
        
        response = tavily_tool.invoke(query)
        
        for res in response:
            links.add(res['url'])

    return links


def arxiv_search(queries:list) -> list:

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

    

# ADD the extractor tools for html and pdf
def extract_text(url:str):
    header = {"User-Agent":"Mozilla/5.0"}

    try:
        response = requests.get(headers=header, url=url, timeout=10)
        content_type = response.headers.get("content-Type","")

        if "application/pdf" in content_type or url.endswith(".pdf"):
            exracted_txt = extract_pdf(response.content)
        elif "text/html" in content_type:
            exracted_txt = extract_html(response.content)
        else:
            exracted_txt = response.text

        return {'content':extract_text,
              'url':url}
        
    except Exception as e:
        print(f"\n!!!!! ERROR at extract_text !!!!!\n{e}\n")
        


def extract_pdf(binary_content):
    try:
        reader = PdfReader(BytesIO(binary_content))
        text = []

        for i,page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    clean_text = str(page_text).replace('\x00','')
                    if clean_text:
                        text.append(clean_text)
            except Exception as page_error:
                print(f"\nWarning: Could not extract page {i+1}. Skipping stream. Error: {page_error}\n")
                continue
        return " ".join(text)
     
    except Exception as e:
        print(f"\n!!!! ERROR at extrcat pdf !!!!\nCritical error reading PDF structure: {e}\n")



def extract_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    main_content = soup.find(id="bodyContent") or soup.find("main") or soup.find("article") or soup
    
    unwanted_tags = ['script', 'style', 'noscript', 'table', 'sup', 'footer']
    for tag in main_content(unwanted_tags):
        tag.decompose()
        
    # 3. Get the raw text with a separator
    text = main_content.get_text(separator=" ")
    
    # 4. Clean up whitespace and Wikipedia artifacts using RegEx
    text = re.sub(r'\[\s*edit\s*\]', '', text)  # Removes "[ edit ]" section links
    text = re.sub(r'\s+', ' ', text)           # Collapses multiple tabs, spaces, and \n into a single space
    
    return text.strip()
