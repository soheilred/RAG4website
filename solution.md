# Eluvio RAG Challenge: Implementation Strategy

## Overall RAG Strategy
My solution follows a standard "Retriever-Reader" architecture. The Retriever identifies the most relevant snippets of Eluvio's technical documentation, and the Reader (Ollama: llama3.2:3b) synthesizes these snippets into a concise, grounded answer. I implemented a Semantic Search approach to ensure that the agent understands the "intent" behind a user's question rather than just matching keywords.

## Scraping & Data Ingestion

The ingestion process utilizes a Python scraper built with BeautifulSoup4. The scraper targets the main Eluvio homepage and recursively crawls the documentation paths at docs.eluv.io. All scraped data is first normalized and saved into a `eluvio_raw_data.jsonl` file to serve as a persistent "Source of Truth."

## Indexing & Retrieval Optimization
To ensure high-quality retrieval, I moved beyond basic keyword search to a vector-based system.

**Embedding Model**: I chose the BAAI/bge-m3 model. It is an accurate local model that offers a robust performance in technical retrieval tasks.

**Chunking Strategy**: I used the RecursiveCharacterTextSplitter with a chunk size of 600 characters and an overlap of 100 characters. This preserves enough context for the LLM while keeping the information granular enough for precise retrieval.

**Vector Database**: ChromaDB was used for local storage, allowing the index to be loaded instantly by the Flask server without re-processing the documentation. Using LangChain's Chroma class simplifies the process; it utilizes the embedding function you provided during initialization to convert your text prompt into a numeric vector. This vector is then compared against the existing document vectors in the database to find the k most similar matches.

## Challenges & Lessons Learned

**Lexical vs. Semantic Gap**: Early versions using keyword matching (Whoosh) failed to answer natural language questions (e.g., "What is the difference between a Validator Node and a Content Node in the Eluvio Fabric?"). Switching to vector embeddings resolved this by allowing the system to understand synonyms and technical concepts. Using transformer-based embeddings, the vector representations of queries and text chunks carry semantic meaning; therefore, finding concepts that are closely related but not represented by the same words becomes possible.

**Resource Management**: Large embedding models (like BGE-M3) occasionally stalled during download. Using a smaller, highly-optimized model like e5-small is an alternative that can provide a smoother experience on laptop hardware. However, I used BGE-M3 for this project.

## Setup and Execution
### Install Dependencies

Install all required Python libraries, including LangChain, ChromaDB, and the embedding models, using pip:

```bash
pip install -r requirements.txt
```

### Build the Local Index

Run the indexing script. This will scrape the Eluvio public site, process the data into semantic chunks, and build the local vector database in the `DB_VEC` directory:

```bash
python index.py
```

### Start the Local Server

Once the index is built, start the Flask server to expose the API. The server will load the vector database into memory and connect to your local Ollama instance:

```bash
python server.py
```

### Query the Agent

With the server running, you can send a POST request to the /ask endpoint:

```bash
curl -X POST "http://localhost:5000/ask" \
     -H "Content-Type: application/json" \
     -d '{"prompt":"What does Eluvio do?"}'
```

## Showcased Results

Query: "What does Eluvio do?"

Response: Based on the provided context, it appears that Eluvio is a platform that provides a Content Fabric, which is a suite of features for managing and distributing content. The platform offers various capabilities, including:\n\n* Just-in-time capabilities\n* Live ingest\n* Cloud origin\n* Live transcoding\n* Content management\n* Encryption/DRM (digital rights management)\n* Program sequencing\n* Rights and avails controls\n* Direct-to-consumer streaming\n* Static content distribution\n\nEluvio's platform uses open APIs and a programmable approach, allowing for customized experiences. It is designed to provide a cost-effective alternative to traditional content delivery technologies, with lower latency and a wider range of creative possibilities.\n\nAdditionally, Eluvio operates an Ethereum-style blockchain that uses the Clique protocol for consensus, which enables features such as validator setup and node provider management.

Query: "What is mainnet?"

Response: "Based on the provided context, it appears that \"mainnet\" refers to a specific network in the blockchain space. In this case, mainnets are mentioned as follows:\n\n* Ethereum Mainnet (e.g., eip155:1/)\n* Polygon Mainnet (eip155:137/)\n* Solana Mainnet (solana:4sGjMW1sUnHzSxGspuhpqLDx6wiyjNtZ/)\n* Flow Mainnet (flow:mainnet/)\n\nA mainnet typically refers to the primary or official network of a blockchain, where the majority of transactions and interactions take place. In contrast, other networks might be referred to as \"testnets\" or \"sidechains,\" which are used for testing, development, or secondary functionality.\n\nIn the context of NFTs and access control, mainnets seem to be used to specify the chain on which a particular NFT is stored."

