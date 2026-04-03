# 🧠 RAG Knowledge Bot

A lightweight, full-stack Retrieval-Augmented Generation (RAG) chatbot that learns from any website you feed it and answers questions based on that specific context. 

Built with a lightning-fast FastAPI backend, LangChain, ChromaDB, and a pure Vanilla JavaScript frontend.

### 🌐 Live Demo
**[Play with the Bot Here](https://krishnna-05.github.io/rag-chatbot/)**

*(Note: The backend is hosted on a free Render instance and spins down after 15 minutes of inactivity. It may take ~50 seconds to wake up on your first request!)*

---

## ✨ Features
* **Custom Knowledge Base:** Paste any Wikipedia link or standard article URL to instantly train the model on its contents.
* **Smart Retrieval:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and local vector search (ChromaDB) to find exact context.
* **Blazing Fast LLM:** Powered by Groq's `llama-3.3-70b-versatile` model for near-instant AI responses.
* **Zero-Build Frontend:** A completely native HTML/CSS/JS frontend. No `node_modules`, no build steps, just clean code.

---

## 🛠️ Tech Stack
**Frontend:**
* HTML5, CSS3, Vanilla JavaScript
* Hosted on: GitHub Pages

**Backend:**
* Python 3 & FastAPI
* LangChain & BeautifulSoup4 (Web Scraping/Chunking)
* ChromaDB (Local Vector Database)
* Groq Cloud API (LLM)
* Hosted on: Render

---

## 🚀 Run It Locally

Want to run this on your own machine? Follow these steps:

### 1. Clone the repository
git clone [https://github.com/krishnna-05/rag-chatbot.git](https://github.com/krishnna-05/rag-chatbot.git)
cd rag-chatbot

### 2. Set up the Python Backend
Create a virtual environment and install the required dependencies:
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt

### 3. Add your API Key
Create a .env file in the root directory and add your Groq API Key:
GROQ_API_KEY=gsk_your_api_key_here

### 4. Start the Server
Run the FastAPI backend using Uvicorn:
uvicorn main:app --reload --port 8000

###5. Launch the Frontend
Because the frontend is pure HTML/JS, simply open frontend/index.html in your web browser. Or, if using VS Code, use the "Live Server" extension.

(Note: If running locally, make sure to change the API_BASE variable in app.js back to http://localhost:8000)

## 📁 Project Structure
```text
rag-chatbot/
│
├── frontend/
│   ├── index.html     # Main UI
│   ├── app.js         # API logic and DOM manipulation
│   └── style.css      # Custom styling
│
├── main.py            # FastAPI backend and RAG pipeline
├── requirements.txt   # Python dependencies
├── .env               # API keys (Not pushed to GitHub)
└── .gitignore         # Hidden files and folders
