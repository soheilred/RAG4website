import os
import requests
import json
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
    if not os.path.exists(JSONFILE):
        scrape_to_file(JSONFILE)
    else:
        print(f"{OUTPUT_FILE} already exists. Skipping scrape.")

    vector_db = create_db(JSONFILE, DB_FILE)

    # for query in questions:
    #     results = vector_db.similarity_search_with_score(query, k=5)
    #     augment_llm(query, results)
