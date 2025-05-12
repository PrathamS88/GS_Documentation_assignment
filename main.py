import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from git import Repo
import shutil
import glob
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from google.generativeai import embed_content, GenerativeModel
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Load API Key
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

app = FastAPI()

# Clone Repo
def clone_repo(repo_url, branch, clone_dir="gs_docs"):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir, branch=branch)

# Get file hash
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Get documentation files
def get_doc_files(base_dir):
    patterns = ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.json", "**/*.yaml", "**/*.yml"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(f"{base_dir}/{pattern}", recursive=True))
    return files

# Embed with Gemini
def gemini_embed(text):
    result = embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

# Ingest docs to ChromaDB
def ingest_docs(doc_files, collection):
    for path in doc_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        h = file_hash(path)
        results = collection.get(where={"hash": h})
        if results["ids"]:
            continue
        collection.delete(where={"path": path})
        emb = gemini_embed(content)
        collection.add(
            documents=[content],
            embeddings=[emb],
            metadatas=[{"path": path, "hash": h}],
            ids=[h]
        )

# One-time setup
repo_url = "https://github.com/godspeedsystems/gs-documentation"
branch = "main"
clone_repo(repo_url, branch)
doc_files = get_doc_files("gs_docs")

# Setup vector store
client = chromadb.Client()
collection = client.get_or_create_collection("gs_docs")
ingest_docs(doc_files, collection)

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(
    collection_name="gs_docs",
    embedding_function=embedding_function,
    persist_directory=None
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = GoogleGenerativeAI(model="gemma-3-1b-it")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Request schema
class QueryInput(BaseModel):
    query: str

@app.post("/ask")
def ask_qna(data: QueryInput):
    result = qa_chain(data.query)
    sources = [doc.metadata["path"] for doc in result['source_documents']]
    return {
        "answer": result['result'],
        "sources": sources
    }
