# Eluvio RAG Challenge: Implementation Strategy

## Overall RAG Strategy
My solution follows a standard "Retriever-Reader" architecture. The Retriever identifies the most relevant snippets of Eluvio's technical documentation, and the Reader (Ollama: llama3.2:3b) synthesizes these snippets into a concise, grounded answer. I implemented a Semantic Search approach to ensure that the agent understands the "intent" behind a user's question rather than just matching keywords.

## Scraping & Data Ingestion

The ingestion process utilizes a Python scraper built with BeautifulSoup4.

Source Targeting: The scraper targets the main Eluvio homepage and recursively crawls the documentation paths at docs.eluv.io.

Unified Storage: All scraped data is first normalized and saved into a eluvio_raw_data.jsonl file to serve as a persistent "Source of Truth."

## Indexing & Retrieval Optimization
To ensure high-quality retrieval, I moved beyond basic keyword search to a vector-based system.

**Embedding Model**: I chose the BAAI/bge-m3 model. It is an accurate local model that outperforms much larger transformers in technical retrieval tasks.

**Chunking Strategy**: I used the RecursiveCharacterTextSplitter with a chunk size of 600 characters and an overlap of 100 characters. This preserves enough context for the LLM while keeping the information granular enough for precise retrieval.

**Vector Database**: ChromaDB was used for local storage, allowing the index to be loaded instantly by the Flask server without re-processing the documentation.

## Challenges & Lessons Learned

**Lexical vs. Semantic Gap**: Early versions using keyword matching (Whoosh) failed to answer natural language questions (e.g., "how to protect content"). Switching to vector embeddings resolved this by allowing the system to understand synonyms and technical concepts.

**Resource Management**: Large embedding models (like BGE-M3) occasionally stalled during download or exhausted RAM. Using a smaller, highly-optimized model like e5-small is an alternative that can provide a smoother experience on laptop hardware.

## Showcased Results

Query: "What does Eluvio do?"

Response: Based on the provided context, it appears that Eluvio is a platform that provides a Content Fabric, which is a suite of features for managing and distributing content. The platform offers various capabilities, including:\n\n* Just-in-time capabilities\n* Live ingest\n* Cloud origin\n* Live transcoding\n* Content management\n* Encryption/DRM (digital rights management)\n* Program sequencing\n* Rights and avails controls\n* Direct-to-consumer streaming\n* Static content distribution\n\nEluvio's platform uses open APIs and a programmable approach, allowing for customized experiences. It is designed to provide a cost-effective alternative to traditional content delivery technologies, with lower latency and a wider range of creative possibilities.\n\nAdditionally, Eluvio operates an Ethereum-style blockchain that uses the Clique protocol for consensus, which enables features such as validator setup and node provider management.

Query: "What is the Chain ID for the mainnet?"

Response: "The specific Chain ID for the Eluvio Mainnet is 955305."
