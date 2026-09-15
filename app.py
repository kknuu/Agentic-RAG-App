import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tavily import TavilyClient

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput

load_dotenv()

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "ollama/llama3.1:8b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "5"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    st.warning(
        "TAVILY_API_KEY is not set. Add it to your .env file to enable the "
        "web-search fallback — PDF-only retrieval will still work."
    )
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

DATA_DIR.mkdir(exist_ok=True)
CHROMA_PATH.mkdir(exist_ok=True)


# Chroma
@st.cache_resource
def get_vector_store():
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

db = get_vector_store()


def add_pdf_to_chroma(pdf_path: str) -> int:
    # Load PDF
    documents = PyPDFLoader(pdf_path).load()

    # Add source metadata
    source_name = Path(pdf_path).name

    for document in documents:
        document.metadata["source_document"] = source_name

    # Split PDF into chunks
    new_chunks = text_splitter.split_documents(documents)

    # Deterministic IDs based on source + chunk index + content hash
    chunk_ids = [
        hashlib.sha256(f"{source_name}-{i}-{chunk.page_content}".encode()).hexdigest()
        for i, chunk in enumerate(new_chunks)
    ]

    # Remove any existing chunks from this source before re-adding
    db.delete(where={"source_document": source_name})

    # Add chunks to Chroma
    db.add_documents(documents=new_chunks, ids=chunk_ids)
    
    return len(new_chunks)

def list_sources():
    try:
        raw = db.get(include=["metadatas"])
        counts = {}
        for metadata in raw.get("metadatas", []):
            if metadata:
                name = metadata.get("source_document", metadata.get("source", "Unknown"))
                counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception:
        return {}



# CrewAI

class RetrievalGrade(BaseModel):
    sufficient: bool = Field(description="True if the retrieved_content fully answers the user query, False otherwise.")

    retrieved_content: str = Field(description="The relevant text pulled from the PDF vector store, copied as-is. Empty string if nothing relevant was found.")

    reasoning: str = Field(description="One or two sentences on why the content is or isn't sufficient.")


def make_crew():
    llm = LLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    @tool("pdf_search_tool")
    def pdf_search_tool(query_text: str) -> str:
        """Search the PDF/RAG knowledge base for information relevant to the query."""
        results = db.similarity_search_with_score(query_text, k=5)
        
        # Prompt
        # Join the content of the retreived chunks into a single string to provide context for the LLM
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        return context_text
    
    @tool("web_search_tool")
    def web_search_tool(query: str) -> str:
        """Search the web for information relevant to the user's query."""

        response = tavily_client.search(query=query, max_results=3)

        return "\n\n".join(
            [
                f"Title: {result['title']}\n"
                f"Content: {result['content']}\n"
                f"URL: {result['url']}"
                for result in response["results"]
            ]
        )

    # Agents

    retriever_agent = Agent(
        role="PDF Retrieval Specialist",
        goal=("""Retrieve the most relevant passages from the PDF knowledge base for the user query: {query}.
                Return the retrieved text exactly as found, without summarizing or rephrasing it."""),
        backstory=("""You are a meticulous research assistant. You only report what the vector store actually returns, you never invent or paraphrase content, and you clearly say when nothing relevant came back."""),
        tools=[pdf_search_tool],
        llm=llm,
        verbose=False,
    )

    grader_agent = Agent(
        role="Retrieval Quality Grader",
        goal=("""Judge whether the content retrieved from the PDF is sufficient, on its own, to fully answer the user query: {query}.
                Be strict: if the content is only tangentially related, partial, or missing key details, mark it as insufficient."""),
        backstory=("""You are a skeptical fact-checker. You never assume missing information can be inferred. You output a clear, structured verdict every time."""),
        llm=llm,
        verbose=False,
    )

    web_search_agent = Agent(
        role="Web Research Specialist",
        goal=("""Search the web for information that fills the gaps left by the PDF retrieval for the query: {query}.
                Focus specifically on what was missing, not on re-covering ground already found in the PDF."""),
        backstory=("""You are a resourceful researcher who finds current, accurate information online and reports it plainly, citing where it came from when possible."""),
        tools=[web_search_tool],
        llm=llm,
        verbose=False,
    )

    synthesizer_agent = Agent(
        role="Response Synthesizer",
        goal=("""Combine all available retrieved information into one clear, concise answer to the user query: {query}."""),
        backstory=("""You are a skilled communicator who turns raw research notes into a clean, direct answer — no hedging, no filler."""),
        llm=llm,
        verbose=False,
    )

    hallucination_checker_agent = Agent(
            role="Hallucination Checker",
            goal=(
                "Check every factual claim in the proposed answer for the query: {query} "
                "against the retrieved context. Flag any claim that is not directly stated "
                "or clearly implied by the context — do not use outside knowledge to judge "
                "correctness, only whether the context supports the claim."
            ),
            backstory=(
                "You are an extremely literal fact-checker. You do not care whether a "
                "claim is true in the real world — you only care whether the provided "
                "context actually says it. If the context doesn't mention it, it's unsupported, "
                "even if you personally know it to be true."
            ),
            llm=llm,
            verbose=False,
        )



        # Tasks

    retrieval_task = Task(
        description=("""Search the PDF vector store for information relevant to the user query: {query}. Return the raw retrieved passages."""),
        expected_output="The relevant passages retrieved from the PDF, verbatim.",
        agent=retriever_agent,
    )

    grading_task = Task(
        description=("""Review the content retrieved by the previous task for the query: {query}. Decide whether it is sufficient to fully answer the query."""),
        expected_output="A structured grade with 'sufficient', 'retrieved_content', and 'reasoning' fields.",
        agent=grader_agent,
        context=[retrieval_task],
        output_pydantic=RetrievalGrade,
    )

    # Checks whether information is sufficient or not, if false then function returns True, meaning to do the websearch
    def needs_web_search(output: TaskOutput) -> bool:
        return output.pydantic.sufficient is False

    web_search_task = ConditionalTask(
        description=("""The PDF content was graded insufficient for the query: {query}. Search the web to find the missing information."""),
        expected_output="Additional information from the web that fills the gap.",
        agent=web_search_agent,
        context=[grading_task],
        condition=needs_web_search,
    )

    synthesis_task = Task(
        description=(
            "Using the PDF retrieval, grading, and any available web research, "
            "write the best concise answer to: {query}. If nothing relevant was "
            "found, say so explicitly."
        ),
        expected_output="A concise final answer grounded in the supplied context.",
        agent=synthesizer_agent,
        context=[grading_task, web_search_task],
    )

    hallucination_task = Task(
        description=(
            "Check the proposed answer for {query}. For every factual claim, "
            "determine whether it is supported by the supplied PDF/web context. "
            "List unsupported claims and explain what should be removed or corrected."
        ),
        expected_output=(
            "A grounded/ungrounded verdict with unsupported claims and reasoning."
        ),
        agent=hallucination_checker_agent,
        context=[grading_task, web_search_task, synthesis_task],
    )

    final_task = Task(
        description=(
            "Produce the final answer to {query}. Use the draft and hallucination "
            "check. Remove or correct unsupported claims. Answer directly and "
            "concisely. Do not mention internal agents or workflow."
        ),
        expected_output="The final grounded answer to the user.",
        agent=synthesizer_agent,
        context=[synthesis_task, hallucination_task],
    )

    return Crew(
        agents=[
            retriever_agent,
            grader_agent,
            web_search_agent,
            synthesizer_agent,
            hallucination_checker_agent,
        ],
        tasks=[
            retrieval_task,
            grading_task,
            web_search_task,
            synthesis_task,
            hallucination_task,
            final_task,
        ],
        process=Process.sequential,
        verbose=False,
    )


# Streamlit UI
st.set_page_config(page_title="CrewAI RAG", layout="wide",)

st.title("CrewAI RAG")
st.caption("Hello! How may I help you today?")

with st.sidebar:
    st.header("Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded files are chunked and added to the persistent Chroma database.",
    )

    if st.button("Add PDFs to database", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        else:
            progress = st.progress(0)
            for i, uploaded in enumerate(uploaded_files):
                safe_name = Path(uploaded.name).name
                target = DATA_DIR / safe_name
                target.write_bytes(uploaded.getbuffer())

                try:
                    with st.spinner(f"Indexing {safe_name}..."):
                        chunks = add_pdf_to_chroma(str(target))
                    st.success(f"{safe_name}: {chunks} chunks indexed.")
                except Exception as exc:
                    st.error(f"{safe_name}: {exc}")
                progress.progress((i + 1) / len(uploaded_files))

            st.cache_resource.clear()
            st.rerun()

    st.divider()

    sources = list_sources()
    st.subheader("Indexed PDFs")
    if sources:
        for source, count in sorted(sources.items()):
            st.write(f"**{source}**")
    else:
        st.info("No PDFs indexed yet.")

    st.divider()
    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption(f"Embeddings: `{EMBED_MODEL}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.status("Running CrewAI RAG...", expanded=True) as status:
            try:
                crew = make_crew()
                result = crew.kickoff(inputs={"query": query})
                answer = getattr(result, "raw", str(result))
                status.update(label="CrewAI RAG complete", state="complete")
            except Exception as exc:
                answer = f"RAG error: `{exc}`"
                status.update(label="CrewAI RAG failed", state="error")

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
