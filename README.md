# CrewAI RAG: Local PDF Q&A with Web Search Fallback
 
A Retrieval-Augmented Generation app that answers questions from your own PDFs using a **local LLM (via Ollama)**, and automatically falls back to a **live web search (Tavily)** when the PDF knowledge base doesn't have enough information. Multiple specialised agents (built with CrewAI handle retrieval, grading, web search, synthesis, and hallucination checking before you see a final answer.
 
## How it works
 
1. **Retriever Agent** - searches a persistent Chroma vector store of your uploaded PDFs.
2. **Grader Agent** - strictly judges whether the retrieved content is actually sufficient to answer the question.
3. **Web Search Agent** *(conditional)* - only runs if the grader says the PDF content is insufficient. Searches the web via Tavily to fill the gap.
4. **Synthesizer Agent** - drafts a concise answer from whatever context is available.
5. **Hallucination Checker Agent** - checks every claim in the draft against the retrieved context and flags anything unsupported.
6. **Synthesizer Agent (final pass)** - produces the corrected, final answer shown to the user.
All of this is wrapped in a Streamlit chat UI with a sidebar for uploading and managing PDFs.
 
## Tech stack
 
| Component | Tool |
|---|---|
| UI | Streamlit |
| Agent orchestration | CrewAI |
| LLM | Ollama (local, e.g. `llama3.1:8b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector store | Chroma |
| PDF loading / chunking | LangChain |
| Web search | Tavily |
 
## Prerequisites
 
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally, with the required models pulled:
```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
```
- A [Tavily](https://tavily.com/) API key (free tier available)
## Installation
 
```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
 
## Usage
 
```bash
streamlit run app.py
```
 
1. Open the sidebar and upload one or more PDFs.
2. Click **Add PDFs to database** - they'll be chunked and embedded into the local Chroma store.
3. Ask questions in the chat box. The agent crew will retrieve from your PDFs first, and only reach for the web if needed.
Re-uploading a PDF with the same filename replaces its previously indexed chunks, so you can safely re-index an updated document.
 
## Project structure
 
```
your-repo/
├── app.py                  # Streamlit app (main entry point)
├── RAG.ipynb               # Exploratory / step-by-step build notebook
├── doc.pdf                 # PDF file to be added to chroma vector base
├── chroma_db/              # Persisted vector store (auto-created)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example            # Template for required env vars
└── README.md
```

## Running with Docker
 
This spins up two containers: the Streamlit app, and Ollama itself (so you don't need Ollama installed on your host machine).
 
1. Make sure `.env` exists with the `TAVILY_API_KEY`, `OLLAMA_BASE_URL` is overridden automatically by Compose so you can leave it as-is.
2. Build and start everything:
```bash
   docker compose up --build
```
3. **Pull the models into the running Ollama container** (only needed once, they persist in a named volume):
```bash
   docker compose exec ollama ollama pull llama3.1:8b
   docker compose exec ollama ollama pull nomic-embed-text
```
4. Open [http://localhost:8501](http://localhost:8501).
