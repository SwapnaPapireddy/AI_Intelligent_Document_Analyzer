"""
rag.py
------
Core engine for the AI Intelligent Document Analyzer.

Pipeline:
  1. File Parser        -> load_pdf_text() / load_image_text()
  2. OCR/Vision Layer    -> ocr_pdf() / ocr_image()  (auto-fallback if PDF has no extractable text)
  3. Vector Store        -> build_vector_store()  (for chat / Q&A over the document)
  4. AI Classification   -> detect_document_type()
  5. Summarization        -> generate_report()  (summary + entities + risk flags)
  6. Structured Output    -> Pydantic schema enforcement + JSON cleanup
  7. Chat / RAG           -> ask_question()
"""

import os
import io
import json
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---- LangChain / RAG imports ----
# NOTE: LangChain has been splitting sub-packages out of the core `langchain`
# package (text splitters, embeddings, vector stores). Depending on which
# version pip resolves, the import path differs. We try the new path first
# and fall back to the old one so this works across versions.
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

# ---- OCR / Vision imports ----
from pypdf import PdfReader
from PIL import Image
import pytesseract

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# ---- Structured output ----
from pydantic import BaseModel, Field


# =========================================================
# 1. STRUCTURED OUTPUT SCHEMA
# =========================================================

class DocumentReport(BaseModel):
    document_type: str = Field(description="e.g. Contract, Invoice, Scanned Form, Other")
    summary: str = Field(description="3-6 sentence plain-language summary")
    key_entities: dict = Field(
        default_factory=dict,
        description="Named entities like parties, dates, amounts, invoice numbers, clauses"
    )
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Potential risks, red flags, missing clauses, unusual terms"
    )
    confidence: str = Field(default="medium", description="low / medium / high")


# =========================================================
# 2. FILE PARSER + OCR/VISION LAYER
# =========================================================

def _pdf_has_selectable_text(pdf_path: str, min_chars: int = 40) -> bool:
    """Quick check: does the PDF already contain machine-readable text,
    or is it a scanned image that needs OCR?"""
    try:
        reader = PdfReader(pdf_path)
        total_chars = 0
        for page in reader.pages[:3]:  # sample first few pages
            text = page.extract_text() or ""
            total_chars += len(text.strip())
        return total_chars >= min_chars
    except Exception:
        return False


def ocr_pdf(pdf_path: str) -> str:
    """OCR a scanned PDF page-by-page using pdf2image + pytesseract."""
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError(
            "pdf2image not installed, or poppler is missing on this machine. "
            "Install poppler-utils (see README) to enable scanned-PDF OCR."
        )
    pages = convert_from_path(pdf_path)
    text_chunks = []
    for i, page_img in enumerate(pages):
        text_chunks.append(pytesseract.image_to_string(page_img))
    return "\n".join(text_chunks)


def ocr_image(image_path: str) -> str:
    """OCR a single image (jpg/png) using pytesseract."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


def load_pdf_text(pdf_path: str) -> str:
    """Load text from a PDF, auto-falling back to OCR if it's scanned."""
    if _pdf_has_selectable_text(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)
    else:
        return ocr_pdf(pdf_path)


def load_document_text(file_path: str) -> str:
    """Dispatch based on file extension. Handles PDFs and common image types."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf_text(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        return ocr_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# =========================================================
# 3. VECTOR STORE (for chat / RAG)
# =========================================================

def build_vector_store(text: str, persist_directory: Optional[str] = None):
    """Split raw text into chunks and build a Chroma vector store for Q&A."""
    if not text or not text.strip():
        raise ValueError("No text extracted from document — cannot build vector store.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    kwargs = {"documents": docs, "embedding": embeddings}
    if persist_directory:
        kwargs["persist_directory"] = persist_directory

    vector_db = Chroma.from_documents(**kwargs)
    return vector_db


# =========================================================
# 4. LLM HELPER
# =========================================================

def get_llm(temperature: float = 0.0):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")
    model_name = "gemini-3.6-flash"
    print(f"[rag.py] get_llm() called — requesting Gemini model: {model_name}")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Add it to Streamlit Cloud Secrets."
        )
    return ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,
        model=model_name,
        temperature=temperature,
    )


def _extract_text(content) -> str:
    """Normalize response.content across providers.

    Groq always returns a plain string. Gemini (via langchain_google_genai)
    can return either a plain string or a list of content parts (each part
    being a string or a dict with a 'text' key). This flattens either shape
    into a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content) if content is not None else ""


# =========================================================
# 5. DOCUMENT TYPE CLASSIFICATION
# =========================================================

def detect_document_type(text: str, llm=None) -> str:
    """Classify the document into Contract / Invoice / Scanned Form / Other."""
    llm = llm or get_llm()
    prompt = (
        "Classify the following document into exactly ONE label from this list: "
        "Contract, Invoice, Scanned Form, Legal Notice, Report, Other.\n"
        "Respond with ONLY the label, nothing else.\n\n"
        f"Document (truncated):\n{text[:3000]}"
    )
    response = llm.invoke(prompt)
    label = _extract_text(response.content).strip().split("\n")[0]
    return label


# =========================================================
# 6. SUMMARY + ENTITY EXTRACTION + RISK FLAGGING (STRUCTURED JSON)
# =========================================================

def _clean_json_block(raw: str) -> str:
    """Strip markdown code fences etc. before json.loads."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


REPORT_PROMPT = PromptTemplate(
    input_variables=["document_type", "text"],
    template="""You are a document intelligence engine. Analyze the document below
(classified as: {document_type}) and respond with ONLY a valid JSON object —
no preamble, no markdown fences, no explanation.

JSON schema:
{{
  "document_type": string,
  "summary": string (3-6 sentences, plain language),
  "key_entities": object (relevant fields such as parties, dates, amounts,
        invoice_number, contract_term, payment_terms — include only what is present),
  "risk_flags": array of strings (missing clauses, unusual terms, ambiguous language,
        compliance concerns, overdue payments, etc. Empty array if none found),
  "confidence": "low" | "medium" | "high"
}}

Document text (truncated to relevant portion):
---
{text}
---

JSON:"""
)


def generate_report(text: str, document_type: Optional[str] = None, llm=None) -> DocumentReport:
    """Runs summarization + entity extraction + risk flagging in one structured call."""
    llm = llm or get_llm()
    document_type = document_type or detect_document_type(text, llm=llm)

    prompt = REPORT_PROMPT.format(document_type=document_type, text=text[:6000])
    response = llm.invoke(prompt)
    raw = _clean_json_block(_extract_text(response.content))

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: LLM didn't return clean JSON — wrap what we have
        data = {
            "document_type": document_type,
            "summary": raw[:800],
            "key_entities": {},
            "risk_flags": ["Could not parse structured output — raw model response used as summary."],
            "confidence": "low",
        }

    # Ensure required keys exist even if the model omitted some
    data.setdefault("document_type", document_type)
    data.setdefault("summary", "")
    data.setdefault("key_entities", {})
    data.setdefault("risk_flags", [])
    data.setdefault("confidence", "medium")

    return DocumentReport(**data)


# =========================================================
# 7. CHAT / RAG Q&A
# =========================================================

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are answering questions about a document, using only the
context excerpts below. If the answer isn't in the context, say so clearly —
do not make anything up.

Context excerpts:
---
{context}
---

Question: {question}

Answer:"""
)


def ask_question(question: str, vector_db, llm=None, k: int = 4) -> str:
    """Answer a question using retrieval-augmented generation over the uploaded document.

    Implemented manually (retrieve -> build prompt -> generate) rather than via
    langchain.chains.RetrievalQA, since that helper's import path and behavior
    has changed across LangChain versions and is being phased out.
    """
    if vector_db is None:
        return "Please upload and process a document first."

    llm = llm or get_llm()

    try:
        # similarity_search works across Chroma versions; retriever.invoke is
        # the newer API but we fall back if it's unavailable.
        try:
            retriever = vector_db.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(question)
        except Exception:
            retrieved_docs = vector_db.similarity_search(question, k=k)

        if not retrieved_docs:
            return "I couldn't find anything relevant to that in the document."

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = QA_PROMPT.format(context=context[:6000], question=question)

        response = llm.invoke(prompt)
        return _extract_text(response.content).strip() or "I couldn't find an answer in the document."

    except Exception as e:
        return f"Error answering question: {e}"
