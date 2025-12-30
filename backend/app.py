# =========================
# ENVIRONMENT SETUP
# =========================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

# =========================
# CORE IMPORTS
# =========================
import io
import fitz
import torch
import base64
from PIL import Image

# =========================
# FASTAPI
# =========================
from fastapi import FastAPI, UploadFile, File
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
# APP INIT
# =========================
app = FastAPI(title="Multimodal RAG (Groq + CLIP)")
device = "cpu"

# =========================
# GLOBAL MODELS (LAZY LOADED)
# =========================
clip_model = None
clip_processor = None
vector_store = None
image_store = {}

def load_clip():
    global clip_model, clip_processor
    if clip_model is None:
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=True
        )
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_fast=False
        )

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
# LLM
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
# PDF UPLOAD
# =========================
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    global vector_store, image_store

    pdf = fitz.open(stream=file.file.read(), filetype="pdf")
    docs, embeddings = [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for page_num, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            temp = Document(page_content=text, metadata={"page": page_num, "type": "text"})
            for chunk in splitter.split_documents([temp]):
                docs.append(chunk)
                embeddings.append(embed_text(chunk.page_content))

        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                base = pdf.extract_image(img[0])
                pil = Image.open(io.BytesIO(base["image"])).convert("RGB")
                image_id = f"page_{page_num}_img_{img_index}"

                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                image_store[image_id] = base64.b64encode(buf.getvalue()).decode()

                docs.append(
                    Document(
                        page_content=f"[IMAGE on page {page_num}]",
                        metadata={"page": page_num, "type": "image", "image_id": image_id}
                    )
                )
                embeddings.append(embed_image(pil))
            except Exception as e:
                print(e)

    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip([d.page_content for d in docs], embeddings)),
        embedding=None,
        metadatas=[d.metadata for d in docs]
    )

    return {"status": "PDF indexed", "chunks": len(docs)}

# =========================
# QUERY
# =========================
@app.post("/query")
def query_rag(req: QueryRequest):
    if vector_store is None:
        return {"error": "No document indexed"}

    q_emb = embed_text(req.question)
    results = vector_store.similarity_search_by_vector(q_emb, k=5)

    text_ctx, image_ctx = [], []
    for d in results:
        if d.metadata["type"] == "text":
            text_ctx.append(f"[Page {d.metadata['page']}]: {d.page_content}")
        else:
            image_ctx.append(f"Image detected on page {d.metadata['page']}")

    prompt = f"""
TEXT CONTEXT:
{chr(10).join(text_ctx)}

IMAGE CONTEXT:
{chr(10).join(image_ctx)}

QUESTION:
{req.question}
"""

    return {"answer": llm.invoke(prompt).content}

# =========================
# HEALTH
# =========================
@app.get("/")
def health():
    return {"status": "ok"}
