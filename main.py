import os
from dotenv import load_dotenv
load_dotenv()

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500", 
        "http://localhost:5500"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ.get("GROQ_API_KEY"))

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
import httpx
from bs4 import BeautifulSoup

from urllib.parse import quote

async def fetch_wikipedia_text(url: str) -> str:
    # Extract title from Wikipedia URL
    if "/wiki/" not in url:
        raise ValueError("Invalid Wikipedia URL")
    title = url.split("/wiki/")[-1].split("#")[0]
    title = title.split("?")[0]
    title = quote(title, safe="")

    # Use MediaWiki API with plaintext extract
    api_url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&explaintext=1&redirects=1&titles="
        + title
    )

    headers = {
        "User-Agent": "rag-chatbot/1.0 (https://github.com/yourname/rag-chatbot; youremail@example.com)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(api_url, headers=headers)
        if r.status_code == 403:
            raise PermissionError(
                "Wikipedia API returned 403 Forbidden. Please ensure the User-Agent is set and the endpoint is not blocked from your network."
            )
        r.raise_for_status()
        data = r.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError("Wikipedia API did not return content")

    page = next(iter(pages.values()))
    extract = page.get("extract", "")
    if not extract or len(extract.strip()) < 100:
        raise ValueError("Wikipedia API returned insufficient text")

    return extract


@app.post("/train")
async def train(req: TrainRequest):
    try:
        if "wikipedia.org/wiki/" in req.url:
            text = await fetch_wikipedia_text(req.url)
        else:
            # Scrape using httpx instead of crawl4ai
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "en-US,en;q=0.9"
            }
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(req.url, headers=headers)
                response.raise_for_status()

            # Parse and extract clean text
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        if not text or len(text) < 100:
            return {"status": "error", "message": "Could not extract text from that URL."}

        # Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents([text], metadatas=[{"source": req.url}])

        # Embed + store
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