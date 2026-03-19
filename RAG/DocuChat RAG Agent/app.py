

import os
import re
import logging
import warnings

# ── Suppress noisy logs ────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()   # reads GEMINI_API_KEY and TAVILY_API_KEY from .env

# ─────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────
W = 62   # box width

def banner(text: str):
    print(f"\n{'─' * W}\n  {text}\n{'─' * W}")

def status(text: str):
    print(f"  {text}")

def print_box(title: str, body: str):
    """Print a nicely bordered response box."""
    def row(content=""):
        inner = content.ljust(W - 2)
        print(f"│ {inner} │")

    print("┌" + "─" * W + "┐")
    row(f"🤖  {title}")
    print("├" + "─" * W + "┤")
    for line in body.split("\n"):
        while len(line) > W - 2:
            print(f"│ {line[:W-2]} │")
            line = "   " + line[W - 2:]
        row(line)
    print("└" + "─" * W + "┘")
    print()

def format_answer(text: str) -> str:
    """Clean up markdown for plain terminal output."""
    text = re.sub(r"\*\*(.*?)\*\*", lambda m: m.group(1).upper(), text)
    lines = text.strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if line.startswith(("* ", "- ")):
            out.append(f"  • {line[2:]}")
        else:
            out.append(line)
    return "\n".join(out)

# ─────────────────────────────────────────
# 1. PDF path
# ─────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

banner("📂  SETUP")
DEFAULT_PDF = "attention_is_all_you_need.pdf"
pdf_path = input(f"  PDF path (Enter = '{DEFAULT_PDF}'): ").strip() or DEFAULT_PDF

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

status(f"Loading: {pdf_path} ...")
doc = PyPDFLoader(pdf_path).load()
status(f"✅ {len(doc)} pages loaded.")

# ─────────────────────────────────────────
# 2. Chunk
# ─────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

all_splits = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(doc)
status(f"✅ Split into {len(all_splits)} chunks.")

# ─────────────────────────────────────────
# 3. Embeddings
# ─────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

status("Loading embedding model ...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
status("✅ Embedding model ready.")

# ─────────────────────────────────────────
# 4. Vector store
# ─────────────────────────────────────────
from langchain_chroma import Chroma

status("Building vector store ...")
vector_store = Chroma(
    collection_name="Gen_AI_research_material",
    embedding_function=embedding_model,
    persist_directory="./langchain_chroma_db",
)
ids = vector_store.add_documents(documents=all_splits)
status(f"✅ Vector store ready ({len(ids)} chunks indexed).")

# ─────────────────────────────────────────
# 5. Gemini model
# ─────────────────────────────────────────
from langchain.chat_models import init_chat_model

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or input("  Enter GEMINI_API_KEY: ").strip()
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required.")

status("Initializing Gemini model ...")
model = init_chat_model("google_genai:gemini-2.5-flash", api_key=GEMINI_API_KEY)
status("✅ Gemini model ready.")

# ─────────────────────────────────────────
# 6. Tools  — langchain_core only, no fragile agent imports
# ─────────────────────────────────────────
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or input("  Enter TAVILY_API_KEY: ").strip()
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is required.")

@tool
def retrieve_from_pdf(query: str) -> str:
    """Search the loaded research paper for content relevant to the query."""
    results = vector_store.similarity_search(query, k=3)
    out = ""
    for d in results:
        page = d.metadata.get("page", "?")
        out += f"[Page {page}]\n{d.page_content}\n\n"
    return out or "No relevant content found in the PDF."

_tavily = TavilySearch(
    max_results=3,
    search_depth="advanced",
    tavily_api_key=TAVILY_API_KEY,
)
_tavily.name = "web_search"

tools = [retrieve_from_pdf, _tavily]
tools_by_name = {t.name: t for t in tools}

# Bind tools so the model knows it can call them
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful research assistant with two tools:

* retrieve_from_pdf — finds information inside the loaded research paper.
* web_search        — searches the web for recent or external information.

Rules:
- Paper questions   -> use retrieve_from_pdf
- Recent / external -> use web_search
- Needs both        -> call both, then give a combined answer
- Always mention which page or source your answer comes from.
"""

# ─────────────────────────────────────────
# 7. Agent loop  (manual, version-stable)
# ─────────────────────────────────────────
def extract_text(content) -> str:
    """
    Safely extract plain text from a model response content field.
    Gemini can return a plain string OR a list of content blocks.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", str(block)))
        return "\n".join(parts).strip()
    return str(content).strip()


# Persistent conversation history — grows across turns
# Format: alternating HumanMessage / AIMessage (+ ToolMessages during tool calls)
conversation_history: list = []

MAX_HISTORY_TURNS = 20   # keep last N human+AI pairs to avoid token bloat


def trim_history(history: list, max_turns: int) -> list:
    """Keep only the most recent max_turns of human/AI exchanges."""
    # Count human messages as turn boundaries
    turn_indices = [i for i, m in enumerate(history) if isinstance(m, HumanMessage)]
    if len(turn_indices) > max_turns:
        cutoff = turn_indices[-max_turns]
        return history[cutoff:]
    return history


def run_agent(user_query: str, verbose: bool = False) -> str:
    """
    Tool-calling loop with persistent conversation memory.
    - conversation_history grows across turns so the model has full context.
    - System prompt is always prepended fresh (not stored in history).
    - Tool calls/results are stored temporarily per-turn only (not in history)
      to keep history clean: we only store HumanMessage + final AIMessage.
    """
    global conversation_history
    from langchain_core.messages import AIMessage

    # Append user message to persistent history
    conversation_history.append(HumanMessage(content=user_query))
    conversation_history = trim_history(conversation_history, MAX_HISTORY_TURNS)

    # Build message list for this turn: system + full history
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(conversation_history)
    response = None

    for _ in range(8):      # safety cap on tool-call iterations
        response = model_with_tools.invoke(messages)
        messages.append(response)   # append to local turn messages only

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            # Model finished — extract answer and save to persistent history
            answer = extract_text(response.content)
            conversation_history.append(AIMessage(content=answer))
            return answer

        # Execute all tool calls the model requested
        for tc in tool_calls:
            name  = tc["name"]
            args  = tc["args"]
            tc_id = tc["id"]

            if verbose:
                status(f"  🔧 [{name}]  args = {args}")

            fn = tools_by_name.get(name)
            try:
                result = fn.invoke(args) if fn else f"Unknown tool: {name}"
            except Exception as exc:
                result = f"Tool error: {exc}"

            if verbose:
                snippet = str(result)[:300].replace("\n", " ")
                status(f"  📄 Result snippet: {snippet}…")

            # Append tool result to local turn messages (not persistent history)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))

    # Fallback if we hit iteration limit
    fallback = extract_text(getattr(response, "content", "")) if response else ""
    if fallback:
        conversation_history.append(AIMessage(content=fallback))
    return "Reached iteration limit. " + fallback

status("✅ Agent ready.")

# ─────────────────────────────────────────
# 8. Interactive chat loop
# ─────────────────────────────────────────
verbose_mode = False

print()
print("╔" + "═" * W + "╗")
print("║" + "  🤖  RAG RESEARCH AGENT".center(W) + "║")
print("║" + f"  Paper: {os.path.basename(pdf_path)}".center(W) + "║")
print("║" + "  Commands: verbose on/off | clear history | quit".center(W) + "║")
print("╚" + "═" * W + "╝")
print()

while True:
    try:
        user_input = input("You › ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  👋 Goodbye!\n")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit", "q"):
        print("\n  👋 Goodbye!\n")
        break

    if user_input.lower() == "verbose on":
        verbose_mode = True
        status("🔍 Verbose ON — tool calls will be shown.\n")
        continue

    if user_input.lower() == "verbose off":
        verbose_mode = False
        status("🔇 Verbose OFF.\n")
        continue

    if user_input.lower() in ("clear history", "clear", "reset"):
        conversation_history.clear()
        status("🗑️  Conversation history cleared. Fresh start!\n")
        continue

    print("\n  ⏳ Thinking...\n")
    try:
        raw = run_agent(user_input, verbose=verbose_mode)
        print_box("Agent", format_answer(raw))
    except Exception as exc:
        status(f"❌ Error: {exc}\n")