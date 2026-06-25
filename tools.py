# --------------IMPORTING LIBRARIES--------------- #

import re
import os
import requests
from io import BytesIO
from pypdf import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_tavily import TavilySearch


# -------------- TOOLS --------------- #

def load_env_variables(env_file=".env"):
    """Tool to load Environment variables"""

    if not os.path.exists(env_file):
        print(f"\n!!!! Warning: {env_file} not found !!!!")
        return
    
    load_dotenv(env_file)
    print(f"> Environment variables loaded from {env_file}")


tavily_api = os.getenv('TAVILY_API_KEY')
def tavily_search(searches:list, max_results:int=3) -> set | list:
    """Tavily search for aquiring information from internet"""

    print('> Using Tavily to Search Internet for relevant information....\n')
    links = set()

    for search in searches:
        query = search['query']
        tgt_domains = search.get('target_domains',[])
        exclude_domains = search.get('exclude_domains',[])
        
        if not tgt_domains:
            print(f'> Executing Open WEB Search for {query}....')
            tavily_tool = TavilySearch(max_results=max_results, search_depth="advanced", )
        else:
            print(f"> Executing Targeted search for {query} from {tgt_domains} ....")
            tavily_tool = TavilySearch(max_results=max_results, search_depth="advanced", include_domains=tgt_domains,exclude_domains=exclude_domains)
        
        response = tavily_tool.invoke(query)
        
        for res in response['results']:
            links.add(res['url'])

    return links, response

   
def extract_text(url:str):
    """Decides and Uses the appropiate scarpper"""

    print(f'> Extracting texts from the {url}....')
    header = {"User-Agent":"Mozilla/5.0"}

    try:
        response = requests.get(headers=header, url=url, timeout=10)
        content_type = response.headers.get("content-Type","")

        if "application/pdf" in content_type or url.endswith(".pdf"):
            extracted_txt = extract_pdf(response.content)
        elif "text/html" in content_type:
            extracted_txt = extract_html(response.content)
        else:
            extracted_txt = response.text

        return {'page_content':extracted_txt,
                "metadata":{'url':url}}
        
    except Exception as e:
        print(f"> \n!!!!! ERROR at extract_text !!!!!\n{e}\n")
        


def extract_pdf(binary_content):
    """PDF scrapper: it scrapes the data from PDFs"""

    print(f'> extract_pdf in action....\n')
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
                print(f"\n> Warning: Could not extract page {i+1}. Skipping stream. Error: {page_error}\n")
                continue
        return " ".join(text)
     
    except Exception as e:
        print(f"\n> !!!! ERROR at extrcat pdf !!!!\nCritical error reading PDF structure: {e}\n")



def extract_html(html_content):
    """HTML scrapper: it scrapes the data from HTML sites, blog sites, web articles others"""

    print(f'> Extract_html in action....\n')
    soup = BeautifulSoup(html_content, "html.parser")
    
    main_content = soup.find(id="bodyContent") or soup.find("main") or soup.find("article") or soup
    
    unwanted_tags = ['script', 'style', 'noscript', 'table', 'sup', 'footer']
    for tag in main_content(unwanted_tags):
        tag.decompose()
        
    # Get the raw text with a separator
    text = main_content.get_text(separator=" ")
    
    # Clean up whitespace and Wikipedia artifacts using RegEx
    text = re.sub(r'\[\s*edit\s*\]', '', text)  # Removes "[ edit ]" section links
    text = re.sub(r'\s+', ' ', text) # Collapses multiple tabs, spaces, and \n into a single space
    
    return text.strip()
