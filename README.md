# 📑 AI Intelligent Document Analyzer

An AI-powered document intelligence app built with **Streamlit**. Upload a contract, invoice, or scanned document — get a structured report (summary, key entities, risk flags) automatically, then chat with the document using retrieval-augmented generation (RAG).

Built to **Help Students and any one**.

---

## 🚀 Live Demo

👉 [Try the app here](https://requirementstxt-gr29btzgwrwu2beacrt6az.streamlit.app/)



---

## ✨ Features

- **Multi-format ingestion** — PDF, PNG, JPG, JPEG
- **Automatic OCR fallback** — if a PDF has no extractable text (i.e. it's a scan), it's routed through OCR automatically
- **Document classification** — labels the upload as Contract, Invoice, Scanned Form, Legal Notice, Report, or Other
- **Structured report generation** — a single LLM call returns:
  - Plain-language summary (3–6 sentences)
  - Key entities (parties, dates, amounts, invoice numbers, clauses, etc.)
  - Risk flags (missing clauses, unusual terms, ambiguous language, overdue payments, etc.)
  - A confidence rating (low / medium / high)
- **Downloadable JSON report** for every processed document
- **Chat with your document** — ask follow-up questions, answered strictly from the document's content via a manually implemented retrieve → prompt → generate RAG pipeline (no `RetrievalQA`, since that helper's behavior and import path has shifted across LangChain versions)

---

## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (`gemini-3.6-flash`) via `langchain-google-genai` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | Chroma |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| PDF Parsing | `PyPDFLoader` (pypdf) |
| OCR | `pytesseract` + `pdf2image` (poppler) |
| Structured Output | Pydantic |

---

## 📂 Project Structure

```
.
├── app.py          # Streamlit UI — upload, report tab, chat tab
├── rag.py          # Core engine — parsing, OCR, vector store, LLM calls, RAG
├── .env            # API keys (not committed)
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

Minimum dependencies (add a `requirements.txt` with these if you don't have one yet):

```
streamlit
python-dotenv
langchain-community
langchain-text-splitters
langchain-chroma
langchain-huggingface
langchain-google-genai
langchain-core
pypdf
pillow
pytesseract
pdf2image
pydantic
```

### 2. System-level dependencies (for OCR)

- **Tesseract OCR** — required by `pytesseract`
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: [installer here](https://github.com/UB-Mannheim/tesseract/wiki)
- **Poppler** — required by `pdf2image` for scanned-PDF OCR
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt install poppler-utils`
  - Windows: [poppler binaries](https://github.com/oschwartz10612/poppler-windows/releases)

### 3. Configure environment variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/).

> **Note:** the app checks which Gemini models your key has access to may vary — if you hit a `404 model not found` error, check the exact model string enabled for your key and update `model_name` in `get_llm()` inside `rag.py`.

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Upload a PDF or image in the main panel.
2. Wait for text extraction (OCR runs automatically for scanned documents), classification, and report generation.
3. View the structured report — summary, key entities, and risk flags — in the **Document Report** tab, and download it as JSON if needed.
4. Switch to the **Chat with Document** tab to ask follow-up questions grounded in the uploaded document.

---

## 🔧 Notes on Design Choices

- **Manual RAG pipeline**: `ask_question()` implements retrieve → build prompt → generate directly, rather than using `langchain.chains.RetrievalQA`, since that helper's import path and behavior has changed across LangChain versions and is being phased out.
- **Provider-agnostic response parsing**: `_extract_text()` normalizes `response.content` across providers — some (like Gemini) can return a list of content parts instead of a plain string.
- **OCR auto-fallback**: `_pdf_has_selectable_text()` samples the first few pages of a PDF; if there isn't enough machine-readable text, the file is routed through OCR instead of failing.

---

## 📄 License

**educational and learning purposes only**.
