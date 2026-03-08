# 🤖 RAG ChatBot Collection

A collection of 4 Retrieval-Augmented Generation (RAG) chatbots built with LangChain, ChromaDB, HuggingFace Embeddings, and Google Gemini. Each bot is designed for a specific use case but shares the same underlying architecture.

---

## 📁 Project Structure

```
RAG/
├── DocuChat/
│   ├── app.py
│   ├── attention_is_all_you_need.pdf
│   ├── .env
│   └── langchain_chroma_db/         # auto-created on first run
│
├── SyllabusChat/
│   ├── syllabus_chat.py
│   ├── syllabus.pdf
│   ├── .env
│   └── syllabus_chroma_db/          # auto-created on first run
│
├── YouTubeChat/
│   ├── youtube_chat.py
│   ├── .env
│   ├── youtube_chroma_db/           # auto-created on first run
│   └── youtube_chroma_db_meta.json  # tracks current video
│
└── FAQChat/
    ├── faq_chat.py
    ├── .env
    ├── faq_docs/                    # put your FAQ files here
    ├── faq_chroma_db/               # auto-created on first run
    └── faq_snapshot.json            # tracks FAQ file changes
```

---

## 🏗️ Architecture

All 4 bots follow the same RAG pipeline:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              RETRIEVAL STAGE                    │
│                                                 │
│  Query → HuggingFace Embeddings                 │
│       → Similarity Search in ChromaDB           │
│       → Top-K relevant chunks returned          │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│              GENERATION STAGE                   │
│                                                 │
│  Retrieved chunks + User query                  │
│       → System prompt (context injected)        │
│       → Google Gemini 2.5 Flash                 │
│       → Final answer generated                  │
└─────────────────────────────────────────────────┘
    │
    ▼
Answer + Sources shown to user
```

---

## 🔧 Common Tech Stack

| Component        | Tool / Model                              |
|------------------|-------------------------------------------|
| Framework        | LangChain                                 |
| Embedding Model  | `sentence-transformers/all-MiniLM-L6-v2`  |
| Vector Database  | ChromaDB (persisted locally)              |
| LLM              | Google Gemini 2.5 Flash                   |
| PDF Loader       | PyPDFLoader (LangChain Community)         |
| Text Splitter    | RecursiveCharacterTextSplitter            |
| Env Management   | python-dotenv                             |

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone repo url
cd RAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-community langchain-text-splitters \
            langchain-huggingface langchain-chroma langchain-google-genai \
            sentence-transformers pypdf chromadb python-dotenv
```

For the **YouTube ChatBot**, install one extra package:

```bash
pip install youtube-transcript-api
```

For the **FAQ ChatBot**, install one extra package:

```bash
pip install unstructured
```

### 4. Set up your `.env` file

Create a `.env` file in each project folder:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get your free Gemini API key at: https://aistudio.google.com/app/apikey

---

## 🤖 Bot 1 — DocuChat (PDF Research Paper Bot)

Ask questions about any research paper or PDF document.

### Setup
- Place your PDF in the project folder
- Update `FILE_PATH` in `app.py` to match your PDF filename

### Run
```bash
python app.py
```

### Example Questions
```
You: What is the use of decoders in Transformers?
You: Explain the attention mechanism.
You: What datasets were used for training?
```

### How persistence works
| Run | Behaviour |
|-----|-----------|
| First run | Loads PDF → splits → embeds → saves to `langchain_chroma_db/` |
| Later runs | Loads existing DB instantly, skips PDF processing |

> To switch to a different PDF, delete the `langchain_chroma_db/` folder and re-run.

---

## 🎓 Bot 2 — Syllabus ChatBot

Ask questions about your college course syllabus.

### Setup
- Place your syllabus PDF in the project folder
- Update `FILE_PATH` in `syllabus_chat.py` to match your filename

### Run
```bash
python syllabus_chat.py
```

### Example Questions
```
You: What are the topics in Unit 3?
You: List all reference books.
You: What is the exam pattern?
You: How many units are there?
```

### Tuning differences vs DocuChat
| Parameter    | DocuChat | SyllabusChat | Reason                              |
|--------------|----------|--------------|-------------------------------------|
| `chunk_size` | 1000     | 500          | Syllabus content is short/structured|
| `k`          | 2        | 4            | Broader coverage for unit listings  |

---

## 📺 Bot 3 — YouTube Transcript ChatBot

Ask questions about any YouTube video using its transcript.

### Setup
No file setup needed — just paste a YouTube URL when prompted.

### Run
```bash
python youtube_chat.py
```

```
Enter YouTube video URL: https://www.youtube.com/watch?v=abc123
```

### Example Questions
```
You: What is this video about?
You: Summarize the key points.
You: What did the speaker say about neural networks?
You: List the main topics covered.
```

### How smart caching works
| Scenario              | Behaviour                                        |
|-----------------------|--------------------------------------------------|
| Same URL as last run  | Loads existing DB instantly                      |
| Different URL entered | Auto-deletes old DB → fetches new transcript → rebuilds |
| First run             | Fetches transcript → builds DB → saves           |

> **Note:** Only works on YouTube videos that have captions enabled (auto-generated or manual). Videos with captions disabled will raise a `TranscriptsDisabled` error.

---

## 🏢 Bot 4 — Company FAQ ChatBot

Ask questions about company policies, HR procedures, IT support, and more.

### Setup
1. Create a `faq_docs/` folder in the project directory
2. Add your FAQ files — supports `.pdf`, `.txt`, and `.csv`

```
faq_docs/
├── hr_policy.pdf
├── it_support.txt
└── benefits.csv
```

### Run
```bash
python faq_chat.py
```

### Example Questions
```
You: How many leave days do I get per year?
You: What is the work from home policy?
You: How do I apply for reimbursement?
You: What are the IT helpdesk hours?
You: How do I reset my company password?
```

### How auto-rebuild works
| Scenario                    | Behaviour                              |
|-----------------------------|----------------------------------------|
| No DB exists                | Loads all files → builds DB → saves    |
| Same files as last run      | Loads existing DB instantly            |
| New file added to faq_docs/ | Auto-detects → rebuilds DB             |

---

## 📊 Bot Comparison

| Feature              | DocuChat  | SyllabusChat | YouTubeChat       | FAQChat              |
|----------------------|-----------|--------------|-------------------|----------------------|
| Input source         | PDF file  | PDF file     | YouTube URL       | Folder of files      |
| Supported formats    | PDF       | PDF          | YouTube transcript| PDF, TXT, CSV        |
| chunk_size           | 1000      | 500          | 500               | 600                  |
| chunk_overlap        | 200       | 100          | 100               | 150                  |
| k (chunks retrieved) | 2         | 4            | 4                 | 4                    |
| DB rebuild trigger   | Manual    | Manual       | URL change        | File change detected |
| Extra dependency     | —         | —            | youtube-transcript-api | unstructured    |

---

## 🗂️ Persistent Vector Store

All bots save their ChromaDB to disk so **you only process your documents once**.

```
First run:   Load → Split → Embed → Save to disk  (slow, done once)
Later runs:  Load from disk                        (fast, instant)
```

To force a full rebuild, delete the respective DB folder:

```bash
# DocuChat
rm -rf langchain_chroma_db/

# SyllabusChat
rm -rf syllabus_chroma_db/

# YouTubeChat
rm -rf youtube_chroma_db/ youtube_chroma_db_meta.json

# FAQChat
rm -rf faq_chroma_db/ faq_snapshot.json
```

---

## 🔑 Environment Variables

| Variable         | Description                  | Required by    |
|------------------|------------------------------|----------------|
| `GEMINI_API_KEY` | Google Gemini API key        | All 4 bots     |

---

## 📌 Notes

- The HuggingFace embedding model (`all-MiniLM-L6-v2`) runs **locally** — no API key needed for embeddings.
- The `UNEXPECTED` warning for `embeddings.position_ids` on startup is **harmless** and can be ignored.
- Gemini 2.5 Flash is used for generation — fast, capable, and free-tier friendly.
- All bots answer **only from the provided documents** — they will not hallucinate or make up answers.

---

## 📜 License

MIT License — free to use and modify for personal or commercial projects.