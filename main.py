import os
from dotenv import load_dotenv
load_dotenv()

# --- RENDER CHROMADB FIX ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LAZY LOADING MODELS ---
# We keep these empty until the user actually requests them
ai_models = {}

def load_models():
    if "embeddings" not in ai_models:
        print("Downloading & Loading AI Models... This takes a minute on the first run.")
        ai_models["embeddings"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        ai_models["llm"] = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ.get("GROQ_API_KEY"))
    return ai_models["embeddings"], ai_models["llm"]


prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chain():
    embeddings, llm = load_models()
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# --- Models ---
class TrainRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    question: str

# --- Routes ---
async def fetch_wikipedia_text(url: str) -> str:
    if "/wiki/" not in url:
        raise ValueError("Invalid Wikipedia URL")
    title = url.split("/wiki/")[-1].split("#")[0]
    title = title.split("?")[0]
    title = quote(title, safe="")

    api_url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&explaintext=1&redirects=1&titles=" + title

    headers = {
        "User-Agent": "rag-chatbot/1.0",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(api_url, headers=headers)
        r.raise_for_status()
        data = r.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    extract = page.get("extract", "")
    if not extract or len(extract.strip()) < 100:
        raise ValueError("Wikipedia API returned insufficient text")

    return extract


@app.post("/train")
async def train(req: TrainRequest):
    try:
        embeddings, llm = load_models() # Load models now that server is active

        if "wikipedia.org/wiki/" in req.url:
            text = await fetch_wikipedia_text(req.url)
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "en-US,en;q=0.9"
            }
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(req.url, headers=headers)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        if not text or len(text) < 100:
            return {"status": "error", "message": "Could not extract text from that URL."}

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents([text], metadatas=[{"source": req.url}])

        Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

        return {"status": "success", "chunks": len(chunks), "url": req.url}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        chain = get_chain()
        answer = chain.invoke(req.question)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}