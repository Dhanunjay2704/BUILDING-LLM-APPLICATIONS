# -*- coding: utf-8 -*-
"""
RAG-DocuChat - VS Code Version

Install dependencies:
    pip install langchain langchain-community langchain-text-splitters \
                langchain-huggingface langchain-chroma langchain-google-genai \
                sentence-transformers pypdf chromadb python-dotenv
"""

import os
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. EMBEDDINGS + PERSISTENT VECTOR STORE
#    - First run  → loads PDF, splits, embeds, saves to disk
#    - Later runs → skips all of the above, loads existing DB directly
# ══════════════════════════════════════════════════════════════════════════════

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# !! Update this path to wherever your PDF is stored locally
FILE_PATH = "attention_is_all_you_need.pdf"
DB_PATH   = "./langchain_chroma_db"

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if not os.path.exists(DB_PATH):
    # ── First run: build the vector store from scratch ──────────────────────
    print("No existing DB found. Building vector store from scratch...")

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load
    print(f"Loading document: {FILE_PATH}")
    loader = PyPDFLoader(FILE_PATH)
    doc = loader.load()
    print(f"Loaded {len(doc)} pages.")

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(doc)
    print(f"Paper split into {len(all_splits)} sub-documents.")

    # Embed & store
    vector_store = Chroma(
        collection_name="Gen_AI_research_material",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Vector store created and saved. ({len(document_ids)} chunks stored)")

else:
    # ── Subsequent runs: load existing DB ───────────────────────────────────
    print("Existing DB found. Loading vector store (skipping PDF processing)...")
    vector_store = Chroma(
        collection_name="Gen_AI_research_material",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    print("Vector store loaded successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, k: int = 2):
    """
    Retrieve the top-k most relevant document chunks for a given query.

    Args:
        query: The user's question.
        k:     Number of chunks to retrieve.

    Returns:
        docs_content:   Concatenated string of source + content for each chunk.
        retrieved_docs: List of raw LangChain Document objects.
    """
    retrieved_docs = vector_store.similarity_search(query, k=k)

    docs_content = ""
    for doc in retrieved_docs:
        docs_content += f"Source : {doc.metadata}\n"
        docs_content += f"Content: {doc.page_content}\n\n"

    return docs_content, retrieved_docs


# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERATION (RAG PIPELINE)
# ══════════════════════════════════════════════════════════════════════════════

from langchain.chat_models import init_chat_model

print("Initializing Gemini model...")
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)


def docu_chat(user_query: str) -> dict:
    """
    Full RAG pipeline: retrieve relevant context, then generate an answer.

    Args:
        user_query: The question to answer.

    Returns:
        dict with keys:
            - answer      : The model's generated answer.
            - source_docs : The retrieved LangChain Document objects.
            - context_used: The raw context string passed to the model.
    """
    context, source_docs = retrieve_context(user_query)

    system_message = (
        "You are a helpful chatbot. "
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information.\n\n"
        f"{context}\n\n"
        "Answer the question ONLY using the provided context. "
        "If the answer is not in the context, say 'I don't know'."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_query}
    ]

    response = model.invoke(messages)

    return {
        "answer":       response.content,
        "source_docs":  source_docs,
        "context_used": context
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def interactive_chat():
    """Run an interactive chat loop in the terminal."""
    print("\n" + "=" * 60)
    print("  DocuChat - Ask anything about your document")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter a question.")
            continue

        print("\nSearching document...")
        response = docu_chat(user_input)

        print(f"\nAssistant: {response['answer']}")

        print("\n[Sources used]")
        for i, doc in enumerate(response["source_docs"], 1):
            print(f"  {i}. Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    interactive_chat()