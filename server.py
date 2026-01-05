from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from flask import Flask, request, jsonify
from constants import *
import ollama


def load_vec_db(model_name, db_file):
    # 1. Define the SAME embedding model as before
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Point to the existing directory
    # No 'from_documents' or 'from_texts'—just instantiate the class
    vector_db = Chroma(
        persist_directory=db_file, 
        embedding_function=embeddings
    )

    # 3. Verify it loaded correctly
    count = vector_db._collection.count()
    print(f"Successfully loaded {count} chunks from the local database.")

    return vector_db


vector_db = load_vec_db(EMBEDDING_MODEL, DB_FILE)
app = Flask(__name__)

def ask_agent(prompt: str) -> str:
    """
    Hybrid RAG handler with query routing
    """

    route = route_query(prompt)
    print(f"[Router] Query classified as: {route}")

    # 1️. GENERAL knowledge → no retrieval
    if route == "GENERAL":
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are a knowledgeable assistant. "
                        "Answer naturally and directly. "
                        "Do not mention internal context or sources."
                    )
                },
                {'role': 'user', 'content': prompt},
            ],
            stream=True,
        )

    # 2️. ELUVIO-specific → pure RAG
    elif route == "ELUVIO":
        results = vector_db.similarity_search_with_score(prompt, k=5)
        stream = augment_llm(prompt, results)

    # 3️. HYBRID → structured answer
    else:  # HYBRID
        results = vector_db.similarity_search_with_score(prompt, k=5)
        stream = augment_llm_hybrid(prompt, results)

    # Collect streaming response
    resp = ""
    for chunk in stream:
        resp += chunk['message']['content']

    return resp

def augment_llm(input_query, retrieved_knowledge):

    instruction_prompt = f'''
    You are an Eluvio product expert helping customers understand the platform.
    Answer naturally and directly.
    Do not mention the existence of retrieved context or documents.

    Eluvio context:
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

    return stream

def augment_llm_hybrid(input_query, retrieved_knowledge):
    """
    Hybrid answer: general knowledge first, then Eluvio-specific
    """

    instruction_prompt = f"""
    You are an Eluvio product expert.

    First, briefly explain the general concept needed to answer the question.
    Then, explain how it specifically relates to Eluvio.

    Use only the Eluvio-related context below for the Eluvio-specific part.
    Do not invent Eluvio details.
    Do not mention internal context or documents.

    Eluvio context:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    """

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    return stream

def route_query(query: str) -> str:
    """
    Classify query into: GENERAL, ELUVIO, or HYBRID
    """
    router_prompt = f"""
    Classify the following user query into exactly one category:
    - GENERAL: general knowledge, not specific to Eluvio
    - ELUVIO: specifically about Eluvio, its product, team, or technology
    - HYBRID: general concept + Eluvio-specific usage

    Respond with ONLY one word: GENERAL, ELUVIO, or HYBRID.

    Query: "{query}"
    """

    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'user', 'content': router_prompt}],
        stream=False,
    )

    return response['message']['content'].strip().upper()


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")

    response_text = ask_agent(prompt)

    return jsonify({"response": response_text})

if __name__ == "__main__":

    # test_queries = [
    #     "What does eluvio do?",
    #     "What is the difference between a Validator Node and a Content Node in the Eluvio Fabric?",
    #     "What role does the Key Service (KMS) node play in content security?",
    #     "What are the hardware requirements (CPU and RAM) for running a Full Fabric Node?",
    #     "Explain the relationship between a Title Master and a Title Mezzanine object.",
    #     "What is an 'Offering' in the context of an Eluvio Mezzanine, and what does it configure?",
    #     "Does the Content Fabric store pre-generated files for every resolution, or is transcoding done on-the-fly?",
    #     "How does Eluvio use proxy re-encryption to protect content from the infrastructure?",
    #     "What are the fees and royalties structure for secondary NFT sales in an Eluvio Marketplace?",
    #     "What happens to an NFT's special operations (like opening a pack) if it is transferred to Ethereum or Polygon?"
    #     ]

    # print("\n--- Running Startup Test Queries ---")
    # for q in test_queries:
    #     try:
    #         # We call the handler directly to test the full pipeline
    #         # Note: Using 'query: ' prefix if you used the E5 model
    #         response = ask_agent(f"query: {q}") 
    #         print(f"PASS\nresponse: {response}")
    #     except Exception as e:
    #         print(f"FAIL | Query: {q} | Error: {e}")
    
    # print(" Tests Complete. Starting Server... \n")

    app.run(host="127.0.0.1", port=5000)
