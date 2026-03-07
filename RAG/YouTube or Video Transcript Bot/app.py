# -*- coding: utf-8 -*-
"""
YouTube Transcript ChatBot - VS Code Version

Ask anything about a YouTube video using its transcript.

Install dependencies:
    pip install langchain langchain-community langchain-text-splitters \
                langchain-huggingface langchain-chroma langchain-google-genai \
                sentence-transformers chromadb python-dotenv \
                youtube-transcript-api
"""

import os
import re
import json
import shutil
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

DB_PATH      = "./youtube_chroma_db"
DB_META_FILE = "./youtube_chroma_db_meta.json"  # stores which video the DB belongs to


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Extract YouTube Video ID from URL
# ══════════════════════════════════════════════════════════════════════════════

def extract_video_id(url: str) -> str:
    """
    Extract the video ID from various YouTube URL formats.
    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Fetch Transcript from YouTube
# ══════════════════════════════════════════════════════════════════════════════

def fetch_transcript(video_id: str) -> str:
    """
    Fetch the transcript for a YouTube video using youtube-transcript-api.
    Returns the full transcript as a single string.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    print(f"Fetching transcript for video ID: {video_id} ...")

    yt = YouTubeTranscriptApi()
    transcript = yt.fetch(video_id)

    full_transcript = " ".join(chunk.text for chunk in transcript)

    print(f"Transcript fetched. ({len(full_transcript)} characters)")
    return full_transcript


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Save & Load DB metadata (which video is currently stored)
# ══════════════════════════════════════════════════════════════════════════════

def save_db_meta(video_id: str, url: str):
    with open(DB_META_FILE, "w") as f:
        json.dump({"video_id": video_id, "url": url}, f)

def load_db_meta() -> dict:
    if os.path.exists(DB_META_FILE):
        with open(DB_META_FILE, "r") as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ASK FOR URL + EMBEDDINGS + PERSISTENT VECTOR STORE
#    - If same video URL → load existing DB (skip re-processing)
#    - If new video URL  → delete old DB, fetch new transcript, rebuild DB
# ══════════════════════════════════════════════════════════════════════════════

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Ask user for YouTube URL ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  YouTube ChatBot")
print("=" * 60)
YOUTUBE_URL = input("\nEnter YouTube video URL: ").strip()

if not YOUTUBE_URL:
    raise ValueError("No URL entered. Please restart and provide a valid YouTube URL.")

video_id = extract_video_id(YOUTUBE_URL)
print(f"Video ID detected: {video_id}")

# ── Load embedding model ─────────────────────────────────────────────────────
print("\nLoading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── Check if existing DB matches the entered video ───────────────────────────
db_meta        = load_db_meta()
db_video_id    = db_meta.get("video_id", "")
db_exists      = os.path.exists(DB_PATH)
same_video     = db_exists and (db_video_id == video_id)

if same_video:
    # ── Same video: load existing DB ─────────────────────────────────────────
    print(f"Existing DB found for this video. Loading vector store...")
    vector_store = Chroma(
        collection_name="youtube_transcript",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    print("Vector store loaded successfully. (Skipping transcript fetch)")

else:
    # ── New video: wipe old DB and rebuild ────────────────────────────────────
    if db_exists:
        print(f"Different video detected (was: {db_video_id}). Rebuilding DB...")
        shutil.rmtree(DB_PATH)
        if os.path.exists(DB_META_FILE):
            os.remove(DB_META_FILE)
    else:
        print("No existing DB found. Building vector store from scratch...")

    # Fetch transcript
    transcript = fetch_transcript(video_id)

    # Wrap in LangChain Document
    docs = [Document(
        page_content=transcript,
        metadata={"source": YOUTUBE_URL, "video_id": video_id}
    )]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Transcript split into {len(all_splits)} chunks.")

    # Embed & store
    vector_store = Chroma(
        collection_name="youtube_transcript",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Vector store created and saved. ({len(document_ids)} chunks stored)")

    # Save metadata so next run can detect same video
    save_db_meta(video_id, YOUTUBE_URL)


# ══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, k: int = 4) -> tuple:
    """
    Retrieve the top-k most relevant transcript chunks for a given query.
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


def video_chat(user_query: str) -> dict:
    """
    Full RAG pipeline for YouTube transcript Q&A.
    """
    context, source_docs = retrieve_context(user_query)

    system_message = (
        "You are a helpful video content assistant. "
        "A user is asking questions about a YouTube video based on its transcript. "
        "Use ONLY the following transcript content to answer the question. "
        "Be clear and concise. If the answer involves steps or a list, format it clearly. "
        "Don't make up any information not present in the transcript.\n\n"
        f"{context}\n\n"
        "If the answer is not found in the transcript content provided, "
        "say 'This topic was not covered in the video.'"
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
    """Run an interactive YouTube transcript chat loop in the terminal."""
    print("\n" + "=" * 60)
    print("  Ask anything about the video!")
    print("  Examples:")
    print("    - What is this video about?")
    print("    - Summarize the key points.")
    print("    - What did the speaker say about X?")
    print("    - List the main topics covered.")
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

        print("\nSearching transcript...")
        response = video_chat(user_input)

        print(f"\nAssistant: {response['answer']}")

        print("\n[Sources used]")
        for i, doc in enumerate(response["source_docs"], 1):
            vid = doc.metadata.get("video_id", "N/A")
            print(f"  {i}. Video ID: {vid} — {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    interactive_chat()