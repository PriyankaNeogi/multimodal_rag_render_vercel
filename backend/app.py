# =========================
# ENVIRONMENT SETUP
# =========================
import os
import logging
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_grad_enabled(False)

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

# =========================
# CORE IMPORTS
# =========================
import io
import fitz  # PyMuPDF
import base64
from PIL import Image
from typing import Optional

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
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# =========================
# TRANSFORMERS (CLIP)
# =========================
from transformers import CLIPProcessor, CLIPModel

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# APP INIT
# =========================
app = FastAPI(title="Multimodal RAG (Groq + CLIP)")
device = "cpu"

# =========================
# GLOBAL MODELS (LAZY)
# =========================
clip_model: Optional[CLIPModel] = None
clip_processor: Optional[CLIPProcessor] = None

def load_clip():
    global clip_model, clip_processor

    if clip_model is None or clip_processor is None:
        logger.info("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=True
        )
        clip_model.to(device)
        clip_model.eval()

        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_fast=False
        )
        logger.info("CLIP loaded")

# =========================
# EMBEDDINGS
# =========================
def embed_text(text: str):
    load_clip()
    with torch.no_grad():
        inputs = clip_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        features = clip_model.get_text_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()

def embed_image(image: Image.Image):
    load_clip()
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt")
        features = clip_model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()

# =========================
# GLOBAL STORES
# =========================
vector_store: Optional[FAISS] = None
image_store = {}

# =========================
# GROQ LLM
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
# ROOT + HEALTH (RENDER FIX)
# =========================
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok"}

# =========================
# PDF UPLOAD
# =========================
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    global vector_store, image_store

    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF too large (max 5MB)")

    try:
        logger.info("PDF upload started")
        pdf_bytes = file.file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

        docs = []
        embeddings = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        for page_num, page in enumerate(pdf):

            # ---- TEXT ----
            text = page.get_text()
            if text.strip():
                temp_doc = Document(
                    page_content=text,
                    metadata={"page": page_num, "type": "text"}
                )
                chunks = splitter.split_documents([temp_doc])

                for chunk in chunks:
                    emb = embed_text(chunk.page_content)
                    docs.append(chunk)
                    embeddings.append(emb)

            # ---- IMAGES ----
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]

                    pil_image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    image_id = f"page_{page_num}_img_{img_index}"

                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")

                    image_store[image_id] = base64.b64encode(
                        buffered.getvalue()
                    ).decode()

                    emb = embed_image(pil_image)

                    image_doc = Document(
                        page_content=f"[IMAGE on page {page_num}]",
                        metadata={
                            "page": page_num,
                            "type": "image",
                            "image_id": image_id
                        }
                    )

                    docs.append(image_doc)
                    embeddings.append(emb)

                except Exception as img_err:
                    logger.error(f"Image error: {img_err}")

        pdf.close()

        vector_store = FAISS.from_embeddings(
            text_embeddings=[
                (doc.page_content, emb)
                for doc, emb in zip(docs, embeddings)
            ],
            embedding=None,
            metadatas=[doc.metadata for doc in docs]
        )

        logger.info("PDF indexed successfully")

        return {
            "status": "PDF indexed",
            "chunks": len(docs)
        }

    except Exception as e:
        logger.exception("PDF processing failed")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# QUERY
# =========================
@app.post("/query")
def query_rag(request: QueryRequest):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document indexed")

    query_embedding = embed_text(request.question)

    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=5
    )

    text_context = []
    image_context = []

    for doc in results:
        if doc.metadata["type"] == "text":
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
    return {"answer": response.content}

# =========================
# LOCAL DEV ONLY
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
