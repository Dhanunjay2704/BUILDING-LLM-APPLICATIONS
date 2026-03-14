# Building LLM Applications

This workspace contains a comprehensive collection of AI-powered applications and assistants built using Python, LangChain, Gradio, and various LLM integrations. Explore different approaches to building conversational AI, RAG systems, and specialized assistants.

## 🏗️ Project Categories

### 🤖 AI Agents Platform
- **career-ai-platform/**: A comprehensive career guidance platform with 6 specialized AI agents for skill mapping, interview prep, salary insights, course finding, startup jobs, and skill comparison. Built with Gradio UI and LangChain agents.

### 🎯 Individual AI Assistants
- **Language Translator Assistant/**: AI-powered language translation tool
- **Question Generator Assistant/**: Generates questions for educational purposes
- **study-assistant/**: Academic study aid and learning assistant
- **Tone Modifier Assistant/**: Modifies text tone and style

### 🔍 RAG (Retrieval-Augmented Generation) Systems
- **RAG/**: Collection of RAG implementations including:
  - College Syllabus Bot (with ChromaDB)
  - Company FAQ Bot
  - DocuChat (document Q&A)
  - YouTube/Video Transcript Bot

### 🛠️ Tool Use & Function Calling
- **Tool use and function calling in LLM/**: Demonstrations of LLM tool integration including:
  - Currency Converter with Tool Calling
  - Weather Assistant with Tool Calling

### 🔗 LangChain Examples
- **Langchain/**: Basic LangChain implementations with different LLM providers

### 🤖 Additional AI Agents (Individual)
Located in `AI Agents/` folder:
- **CareerLens/**: Career-focused AI assistant
- **Course Finder Agent/**: Specialized course recommendation system
- **Interview Prep Agent/**: Interview preparation assistant
- **Salary insights agent/**: Salary analysis and insights
- **Skill Comparison Agent/**: Compare different technical skills
- **Startup jobs agent/**: Startup job search and insights

## 🚀 Quick Start

1. **Set up environment**:
   ```bash
   # Activate virtual environment
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac

   # Install dependencies (if needed)
   pip install -r requirements.txt  # For projects with requirements.txt
   ```

2. **Configure API keys**:
   - Copy `.env.example` to `.env` (if available)
   - Add your API keys for LLM providers (OpenAI, Google Gemini, etc.)

3. **Run an application**:
   ```bash
   cd "project-folder"
   python app.py
   ```

## 📋 Prerequisites

- Python 3.8+
- API keys for various LLM providers (OpenAI, Google Gemini, Anthropic, etc.)
- Required Python packages (listed in individual project READMEs or requirements.txt)

## 🏛️ Architecture Overview

Most applications follow this pattern:
- **Frontend**: Gradio for web UI, Streamlit for some projects
- **Backend**: Python with LangChain for LLM orchestration
- **LLMs**: Integration with multiple providers (OpenAI GPT, Google Gemini, Anthropic Claude, etc.)
- **Tools**: Custom tools for search, APIs, and data processing
- **Storage**: Vector databases like ChromaDB, FAISS for RAG applications

## 🤝 Contributing

Feel free to:
- Explore and modify existing code
- Add new AI assistants
- Improve documentation
- Share your own LLM application ideas

## 📚 Learning Resources

This workspace demonstrates:
- Conversational AI development
- RAG system implementation
- Tool integration with LLMs
- Multi-agent systems
- UI development with Gradio/Streamlit
- API integrations and data processing

Each project includes commented code to help you understand the implementation details.
