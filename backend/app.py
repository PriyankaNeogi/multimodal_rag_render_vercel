# =========================
# ENVIRONMENT SETUP
# =========================
import os
import io
import base64
import logging
import fitz  # PyMuPDF

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

# =========================
# FASTAPI
# =========================
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# =========================
# LANGCHAIN
# =========================
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# 👉 Lightweight embeddings (MiniLM)
from langchain_community.embeddings import HuggingFaceEmbeddings

# 👉 Persistent vector DB
from langchain_community.vectorstores import Chroma

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# APP INIT
# =========================
app = FastAPI(title="Multimodal RAG (MiniLM + Groq)")

# =========================
# EMBEDDING MODEL (LIGHTWEIGHT)
# =========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTOR STORE (PERSISTENT)
# =========================
vector_store = Chroma(
    collection_name="multimodal_rag",
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Store images as base64 (optional future use)
image_store = {}

# =========================
# LLM (GROQ)
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# =========================
# REQUEST MODEL
# =========================
class QueryRequest(BaseModel):
    question: str

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# PDF UPLOAD
# =========================
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = file.file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        documents = []

        for page_num, page in enumerate(pdf):

            # -------- TEXT --------
            text = page.get_text()
            if text.strip():
                base_doc = Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "type": "text"
                    }
                )
                documents.extend(
                    splitter.split_documents([base_doc])
                )

            # -------- IMAGES (METADATA ONLY) --------
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]

                pil_image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                image_id = f"page_{page_num}_img_{img_index}"

                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")

                image_store[image_id] = base64.b64encode(
                    buffer.getvalue()
                ).decode()

                documents.append(
                    Document(
                        page_content=f"[IMAGE on page {page_num}]",
                        metadata={
                            "page": page_num,
                            "type": "image",
                            "image_id": image_id
                        }
                    )
                )

        pdf.close()

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No content found in PDF"
            )

        vector_store.add_documents(documents)
        vector_store.persist()

        return {
            "status": "PDF indexed",
            "chunks": len(documents)
        }

    except Exception as e:
        logger.exception("PDF upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# QUERY
# =========================
@app.post("/query")
def query_rag(request: QueryRequest):

    if vector_store._collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document indexed"
        )

    results = vector_store.similarity_search(
        request.question,
        k=5
    )

    text_context = []
    image_context = []

    for doc in results:
        if doc.metadata.get("type") == "text":
            text_context.append(
                f"[Page {doc.metadata['page']}]: {doc.page_content}"
            )
        else:
            image_context.append(
                f"Image present on page {doc.metadata['page']}"
            )

    prompt = f"""
You are a multimodal assistant.

TEXT CONTEXT:
{chr(10).join(text_context)}

IMAGE CONTEXT:
{chr(10).join(image_context)}

QUESTION:
{request.question}

Answer clearly and accurately.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# =========================
# LOCAL DEV
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
