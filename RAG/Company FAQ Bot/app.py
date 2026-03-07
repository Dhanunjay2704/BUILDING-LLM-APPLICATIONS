

import os
import shutil
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

# !! Put all your FAQ files (PDF / TXT / CSV) in this folder
FAQ_DIR  = "./faq_docs"
DB_PATH  = "./faq_chroma_db"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Load all documents from the FAQ folder
# ══════════════════════════════════════════════════════════════════════════════

def load_faq_documents():
    """
    Load all supported FAQ files from FAQ_DIR.
    Supports: .pdf, .txt, .csv
    Returns a combined list of LangChain Document objects.
    """
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader,
    )

    if not os.path.exists(FAQ_DIR):
        os.makedirs(FAQ_DIR)
        print(f"Created '{FAQ_DIR}' folder. Please add your FAQ files there and re-run.")
        exit()

    all_docs   = []
    file_count = 0

    for filename in os.listdir(FAQ_DIR):
        filepath = os.path.join(FAQ_DIR, filename)
        ext      = filename.lower().split(".")[-1]

        try:
            if ext == "pdf":
                loader = PyPDFLoader(filepath)
            elif ext == "txt":
                loader = TextLoader(filepath, encoding="utf-8")
            elif ext == "csv":
                loader = CSVLoader(filepath)
            else:
                print(f"  Skipping unsupported file: {filename}")
                continue

            docs = loader.load()
            all_docs.extend(docs)
            file_count += 1
            print(f"  Loaded: {filename} ({len(docs)} pages/sections)")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    if not all_docs:
        print(f"No supported files found in '{FAQ_DIR}'. Add .pdf, .txt, or .csv files.")
        exit()

    print(f"\nTotal: {file_count} file(s) loaded → {len(all_docs)} document sections.")
    return all_docs


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Check if FAQ folder has changed since last DB build
# ══════════════════════════════════════════════════════════════════════════════

def get_faq_file_list() -> set:
    """Return a set of filenames currently in FAQ_DIR."""
    if not os.path.exists(FAQ_DIR):
        return set()
    return set(os.listdir(FAQ_DIR))

def save_file_snapshot(files: set):
    import json
    with open("./faq_snapshot.json", "w") as f:
        json.dump(sorted(list(files)), f)

def load_file_snapshot() -> set:
    import json
    if os.path.exists("./faq_snapshot.json"):
        with open("./faq_snapshot.json", "r") as f:
            return set(json.load(f))
    return set()


# ══════════════════════════════════════════════════════════════════════════════
# 1. EMBEDDINGS + PERSISTENT VECTOR STORE
#    - First run       → loads all FAQ docs, splits, embeds, saves to disk
#    - Later runs      → skips processing, loads existing DB
#    - New file added  → auto-detects change, rebuilds DB
# ══════════════════════════════════════════════════════════════════════════════

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

current_files  = get_faq_file_list()
snapshot_files = load_file_snapshot()
db_exists      = os.path.exists(DB_PATH)
files_changed  = current_files != snapshot_files

if db_exists and not files_changed:
    # ── No changes: load existing DB ─────────────────────────────────────────
    print("Existing DB found. No new files detected. Loading vector store...")
    vector_store = Chroma(
        collection_name="company_faq",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    print("Vector store loaded successfully.")

else:
    # ── New/changed files: rebuild DB ────────────────────────────────────────
    if db_exists and files_changed:
        print("FAQ files changed. Rebuilding vector store...")
        shutil.rmtree(DB_PATH)
    else:
        print("No existing DB found. Building vector store from scratch...")

    # Load all FAQ documents
    print(f"\nLoading FAQ documents from '{FAQ_DIR}'...")
    all_docs = load_faq_documents()

    # Split into chunks
    # Larger overlap helps preserve Q&A context across chunk boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )
    all_splits = text_splitter.split_documents(all_docs)
    print(f"Split into {len(all_splits)} chunks.")

    # Embed & store
    vector_store = Chroma(
        collection_name="company_faq",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Vector store created and saved. ({len(document_ids)} chunks stored)")

    # Save snapshot of current files
    save_file_snapshot(current_files)


# ══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, k: int = 4) -> tuple:
    """
    Retrieve the top-k most relevant FAQ chunks for a given query.
    k=4 helps cover answers that may be spread across multiple FAQ sections.
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

print("\nInitializing Gemini model...")
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)


def faq_chat(user_query: str) -> dict:
    """
    Full RAG pipeline for Company FAQ Q&A.
    Retrieves relevant FAQ sections and generates a helpful answer.
    """
    context, source_docs = retrieve_context(user_query)

    system_message = (
        "You are a professional and friendly company HR/support assistant. "
        "Employees are asking you questions about company policies, procedures, "
        "benefits, IT support, and other internal topics. "
        "Use ONLY the following FAQ content to answer the question. "
        "Be clear, professional, and concise. "
        "If the answer involves multiple steps or points, format them as a numbered list. "
        "Do not make up any information not present in the FAQ documents.\n\n"
        f"{context}\n\n"
        "If the answer is not found in the provided FAQ content, say: "
        "'I could not find this information in the company FAQ. "
        "Please contact HR or your manager for assistance.'"
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
    """Run an interactive company FAQ chat loop in the terminal."""
    print("\n" + "=" * 60)
    print("  Company FAQ ChatBot")
    print("  Ask anything about company policies & procedures")
    print("  Examples:")
    print("    - How many leave days do I get per year?")
    print("    - What is the work from home policy?")
    print("    - How do I apply for reimbursement?")
    print("    - What are the IT helpdesk hours?")
    print("    - How do I reset my company password?")
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

        print("\nSearching FAQ...")
        response = faq_chat(user_input)

        print(f"\nAssistant: {response['answer']}")

        print("\n[Sources used]")
        for i, doc in enumerate(response["source_docs"], 1):
            source = doc.metadata.get("source", "Unknown")
            page   = doc.metadata.get("page", None)
            loc    = f"Page {page}" if page is not None else "Section unknown"
            print(f"  {i}. {os.path.basename(source)} — {loc}")


if __name__ == "__main__":
    interactive_chat()