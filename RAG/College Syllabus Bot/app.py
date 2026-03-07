
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

# !! Update this to your syllabus PDF path
FILE_PATH = "syllabus.pdf"
DB_PATH   = "./syllabus_chroma_db"

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
    print(f"Loading syllabus: {FILE_PATH}")
    loader = PyPDFLoader(FILE_PATH)
    doc = loader.load()
    print(f"Loaded {len(doc)} pages.")

    for page in doc:
        print("PAGE TEXT:", page.page_content[:200])

    # Split
    # Smaller chunk size works better for syllabus docs (short, structured content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(doc)
    print(f"Syllabus split into {len(all_splits)} sub-documents.")

    # Embed & store
    vector_store = Chroma(
        collection_name="college_syllabus",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Vector store created and saved. ({len(document_ids)} chunks stored)")

else:
    # ── Subsequent runs: load existing DB ───────────────────────────────────
    print("Existing DB found. Loading vector store (skipping PDF processing)...")
    vector_store = Chroma(
        collection_name="college_syllabus",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    print("Vector store loaded successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, k: int = 4):
    """
    Retrieve the top-k most relevant syllabus chunks for a given query.
    k=4 gives broader coverage for syllabus queries like "list all units".
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


def syllabus_chat(user_query: str) -> dict:
    """
    Full RAG pipeline for syllabus Q&A.
    Retrieves relevant syllabus sections and generates a focused answer.
    """
    context, source_docs = retrieve_context(user_query)

    system_message = (
        "You are a helpful college academic assistant. "
        "A student is asking questions about their course syllabus. "
        "Use ONLY the following syllabus content to answer the question. "
        "Be clear, structured, and student-friendly in your response. "
        "If the answer involves a list of topics or units, format them as a numbered list. "
        "Don't make up any information not present in the syllabus.\n\n"
        f"{context}\n\n"
        "If the answer is not found in the syllabus content provided, "
        "say 'This information is not available in the syllabus.'"
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
    """Run an interactive syllabus chat loop in the terminal."""
    print("\n" + "=" * 60)
    print("  Syllabus ChatBot - Ask anything about your syllabus")
    print("  Examples:")
    print("    - What are the topics in Unit 3?")
    print("    - What is the exam pattern?")
    print("    - List all reference books.")
    print("    - How many units are there?")
    print("    - What is covered in the first unit?")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! Good luck with your studies!")
            break

        if not user_input:
            print("Please enter a question.")
            continue

        print("\nSearching syllabus...")
        response = syllabus_chat(user_input)

        print(f"\nAssistant: {response['answer']}")

        print("\n[Sources used]")
        for i, doc in enumerate(response["source_docs"], 1):
            print(f"  {i}. Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    interactive_chat()