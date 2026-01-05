import os
import requests
import json
import time
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
# from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from constants import *
import fitz  # PyMuPDF

MAX_PAGES = 500  # Adjust as needed

def extract_pdf_text(url):
    """Downloads PDF and extracts text."""
    if not os.path.exists(PDF_DIR): os.makedirs(PDF_DIR)
    path = os.path.join(PDF_DIR, url.split('/')[-1])
    try:
        with requests.get(url, timeout=10) as r:
            with open(path, 'wb') as f: f.write(r.content)
        text = ""
        with fitz.open(path) as doc:
            for page in doc: text += page.get_text()
        return text
    except Exception as e:
        print(f"PDF Error {url}: {e}")
        return ""

def should_scrape(url):
    parsed = urlparse(url)
    clean_url = url.split('#')[0].rstrip('/').lower()
    
    # 1. NEW: Check if any banned substring exists in the URL
    if any(substring in clean_url for substring in BANNED_SUBSTRINGS):
        print(f"Bypassing banned URL pattern: {url}")
        return False
        
    # 2. Check domain (include subdomains like docs)
    is_eluvio = parsed.netloc == BASE_URL or parsed.netloc.endswith("." + BASE_URL)
    
    # 3. Check for binary file extensions
    is_binary_file = parsed.path.lower().endswith(SKIP_EXTENSIONS)
    
    return is_eluvio and not is_binary_file

def scrape_eluvio(jsonfile):
    visited = set()
    to_visit = START_URLS.copy()
    count = 0

    with sync_playwright() as p:
        # Launch browser - headless=True for background, False to debug
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        # page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"})

        with open(jsonfile, 'w', encoding='utf-8') as f:
            while to_visit and count < MAX_PAGES:
                url = to_visit.pop(0).split('#')[0].rstrip('/')
                # Check if we've been here OR if it's a banned/invalid link
                if url in visited:
                    continue
                
                if not should_scrape(url):
                    # ADDED: Mark it as visited even if we don't scrape it
                    visited.add(url)
                    continue
                
                print(f"[{count+1}] Processing: {url}")

                try:
                    if url.lower().endswith('.pdf'):
                        content = extract_pdf_text(url)
                        title = f"PDF: {url.split('/')[-1]}"
                    else:
                        # Use networkidle to ensure JS hydration is complete
                        page.goto(url, wait_until="networkidle", timeout=10000)
                        title, content = page.title(), page.inner_text('body')

                        # Piercing Shadow DOM to find links
                        raw_links = page.locator('a[href]').evaluate_all("els => els.map(el => el.href)")
                        for link in raw_links:
                            full_url = urljoin(url, link).split('#')[0].rstrip('/')
                            if full_url not in visited and should_scrape(full_url):
                                to_visit.append(full_url)

                    f.write(json.dumps({"title": title, "url": url, "content": content.strip()}) + "\n")
                    visited.add(url)
                    count += 1
                except Exception as e:
                    print(f"Skipped {url}: {e}")

        browser.close()
    print(f"Done! Saved {count} pages to {OUTPUT_FILE}")

def scrape_to_file(file_path):
    # Define the schema for our searchable index
    schema = Schema(
        title=TEXT(stored=True),
        url=ID(stored=True, unique=True),
        content=TEXT(stored=True),
        source_type=TEXT(stored=True) # 'api' or 'documentation'
    )
    visited = set()
    to_visit = [START_URL]
    count = 0
    
    # Open file in append mode
    with open(file_path, 'w', encoding='utf-8') as f:
        print(f"Starting crawl. Saving to {file_path}...")
        
        while to_visit and count < 200: # Adjust limit as needed
            current_url = to_visit.pop(0)
            if current_url in visited: continue
            
            try:
                r = requests.get(current_url, timeout=5)
                soup = BeautifulSoup(r.text, 'html.parser')
                visited.add(current_url)
                
                # Extract main content
                main_content = soup.find('main') or soup.find('article')
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                    title = soup.title.string if soup.title else current_url
                    
                    # Create a data object
                    page_data = {
                        "title": title,
                        "url": current_url,
                        "content": text
                    }
                    
                    # Write as a single line in JSONL
                    f.write(json.dumps(page_data) + "\n")
                    count += 1
                    print(f"[{count}] Saved: {title}")

                # Find internal links
                for link in soup.find_all('a', href=True):
                    full_url = requests.compat.urljoin(DOCS_BASE_URL, link['href'])
                    if full_url.startswith(DOCS_BASE_URL) and full_url not in visited:
                        to_visit.append(full_url)
                        
            except Exception as e:
                print(f"Error on {current_url}: {e}")

    print("Scraping finished.")



def create_db(data_dir, db_file):
    # 1. Load the data from your JSONL file
    print(f"Loading data from {data_dir}...")
    documents = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # We wrap the content in a LangChain Document object
            documents.append(Document(
                page_content=data['content'],
                metadata={"title": data['title'], "url": data['url']}
            ))

    # 2. Chunking (600 characters with 100 overlap is ideal for technical docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,
                                                   chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Setup BGE-M3 using the new LangChain-HuggingFace class
    print(f"Initializing {EMBEDDING_MODEL}...")
    model_kwargs = {'device': 'cuda'} # Use 'cuda' if you have a GPU
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # NOTE: BGE-M3 works best in LangChain with an empty query instruction
        # query_encoder_kwargs={"instruction": ""} 
    )

    # 4. Create the Vector DB
    print("Creating local Vector Database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_file
    )
    print(f"Vector Database created and saved to {db_file}")
    return vector_db


def generate_phrase():
    input_query = input('Ask me a question: ')
    retrieved_knowledge = retrieve(input_query)

    print('Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question.
    Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''
    stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    # print the response from the chatbot in real-time
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def search_index(ix, query_str, limit):
    print(f"\n--- Searching for: '{query_str}' ---")
    results_array = []
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query, limit=limit)
        
        if not results:
            print("No results found.")
            return

        for hit in results:
            if hit.score < 1.5:
                continue # Ignore low-relevance results to prevent hallucinations
            chunk = hit['content']
            score = hit.score
            
            results_array.append((chunk, score))
    return results_array

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    # temporary list to store (chunk, similarity) pairs
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
        # sort by similarity in descending order, because higher similarity means more relevant chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        # finally, return the top N most relevant chunks
    return similarities[:top_n]

def setup_index():
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)
        return create_in(INDEX_DIR, schema)
    return open_dir(INDEX_DIR)

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

if __name__ == "__main__":
    # if not os.path.exists(JSONFILE):
        # scrape_to_file(JSONFILE)
    scrape_eluvio(JSONFILE)
    # else:
        # print(f"{OUTPUT_FILE} already exists. Skipping scrape.")

    vector_db = create_db(JSONFILE, DB_FILE)

    # for query in questions:
    #     results = vector_db.similarity_search_with_score(query, k=5)
    #     augment_llm(query, results)
