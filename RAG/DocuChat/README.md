
# 🤖 Intelligent RAG Agent

> A decision-making AI agent that combines **document retrieval + real-time web search** to generate accurate, context-aware answers.

---

## 📌 Overview

This project implements an **Intelligent RAG (Retrieval-Augmented Generation) Agent** that goes beyond traditional RAG systems.

Instead of relying only on a document, the agent can:

* 📄 Retrieve information from a PDF (vector database)
* 🌐 Fetch real-time data from the web
* 🧠 Decide which tool(s) to use
* 🔀 Combine multiple sources into a single answer

---

## 🧠 Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│    RAG Agent     │   ← Decision Maker
└────────┬─────────┘
         │
 ┌───────┴──────────────┐
 │                      │
 ▼                      ▼
PDF Retriever        Web Search
(ChromaDB)           (Tavily API)
 │                      │
 └──────────┬───────────┘
            ▼
        Gemini LLM
            ▼
      Final Answer
```

---

## ⚡ Features

* 🤖 **Agent-based decision making** (tool selection)
* 📄 **Semantic PDF retrieval** using embeddings
* 🌐 **Real-time web search integration**
* 🔀 **Hybrid answers** (document + internet)
* 🧠 **Conversation memory (multi-turn chat)**
* 🧰 **Custom tool-calling loop (no fragile agent APIs)**
* 🖥️ **Terminal-based interactive interface**
* 🔍 **Verbose mode for debugging tool calls**

---

## 🧱 Tech Stack

* **Language**: Python
* **LLM**: Gemini (Google GenAI)
* **Framework**: LangChain (Core Tools)
* **Embeddings**: HuggingFace (MiniLM)
* **Vector Database**: ChromaDB
* **Search API**: Tavily

---

## 🚀 How It Works

1. User inputs a query
2. Agent analyzes intent
3. Agent selects appropriate tool(s):

   * 📄 PDF Retriever
   * 🌐 Web Search
   * 🔀 Both
4. Retrieved data is passed to Gemini
5. Final answer is generated with sources

---

## 📂 Project Structure

```
.
├── app.py              # Main agent implementation
├── langchain_chroma_db/     # Persistent vector database
├── .env                      # API keys
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone repo
cd rag-agent
```

---

### 2. Install dependencies

```bash
pip install langchain-community pypdf langchain-text-splitters \
langchain-huggingface sentence-transformers langchain-chroma \
chromadb langchain-google-genai langchain-core langchain \
langchain-tavily python-dotenv
```

---

### 3. Add API Keys

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

### 4. Run the Agent

```bash
python app.py
```

---

## 💬 Commands (Interactive Mode)

* `verbose on`  → Show tool calls
* `verbose off` → Hide tool calls
* `clear`       → Reset conversation history
* `quit`        → Exit the program

---

## 💡 Example Queries

```
Explain attention mechanism from the paper
What are the latest advancements in transformers?
Compare paper concepts with current AI trends
```

---

## 🔥 Key Insight

> Traditional RAG systems retrieve information.
> This agent **reasons, decides, and acts**.

