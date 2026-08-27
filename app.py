import os
import json
import tempfile

import streamlit as st

import rag

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Intelligent Document Analyzer",
    layout="wide"
)

st.title("📑 AI Intelligent Document Analyzer")
st.write("Upload a contract, invoice, or scanned document. Get a structured report, then chat with it.")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ How it works")
    st.write("""
    1. Upload a PDF or image (scanned docs use OCR automatically)
    2. The document is classified (Contract / Invoice / Scanned Form / etc.)
    3. Get a summary, key entities, and risk flags as structured JSON
    4. Ask follow-up questions in the chat tab — answers are grounded
       in the document via RAG (retrieval-augmented generation)
    """)
    st.divider()
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("⚠️ GEMINI_API_KEY not found. Add it to your .env file.")
    else:
        st.success("✅ GEMINI_API_KEY loaded")

# =========================================================
# SESSION STATE
# =========================================================
for key, default in {
    "doc_text": "",
    "vector_db": None,
    "report": None,
    "messages": [],
    "processed_filename": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None and uploaded_file.name != st.session_state.processed_filename:

    st.success("✅ File uploaded")
    st.write(f"📌 **Filename:** {uploaded_file.name}  |  📦 **Size:** {uploaded_file.size / 1024:.2f} KB")

    with st.spinner("Extracting text (OCR fallback if needed)..."):
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            extracted_text = rag.load_document_text(tmp_path)
        except Exception as e:
            st.error(f"Text extraction failed: {e}")
            extracted_text = ""
        finally:
            os.unlink(tmp_path)

    if not extracted_text.strip():
        st.error("No text could be extracted from this document.")
    else:
        st.session_state.doc_text = extracted_text
        st.success(f"📄 Extracted {len(extracted_text)} characters")

        with st.spinner("Classifying document and generating report..."):
            try:
                llm = rag.get_llm()
                doc_type = rag.detect_document_type(extracted_text, llm=llm)
                report = rag.generate_report(extracted_text, document_type=doc_type, llm=llm)
                st.session_state.report = report
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.session_state.report = None

        with st.spinner("Building vector index for chat..."):
            try:
                st.session_state.vector_db = rag.build_vector_store(extracted_text)
            except Exception as e:
                st.error(f"Vector store build failed: {e}")
                st.session_state.vector_db = None

        st.session_state.messages = []
        st.session_state.processed_filename = uploaded_file.name

# =========================================================
# TABS: REPORT + CHAT
# =========================================================
tab_report, tab_chat = st.tabs(["📊 Document Report", "💬 Chat with Document"])

# ---------------- REPORT TAB ----------------
with tab_report:
    report = st.session_state.report

    if report is None:
        st.info("Upload a document to generate its structured report.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"Document Type: {report.document_type}")
            st.write("**Summary**")
            st.write(report.summary)

            st.write("**Key Entities**")
            if report.key_entities:
                st.json(report.key_entities)
            else:
                st.write("_No entities extracted._")

        with col2:
            st.write("**Risk Flags**")
            if report.risk_flags:
                for flag in report.risk_flags:
                    st.warning(f"⚠️ {flag}")
            else:
                st.success("No risk flags detected.")

            st.write("**Confidence**")
            st.write(report.confidence.capitalize())

        st.divider()
        report_json = json.dumps(report.model_dump(), indent=2)
        st.download_button(
            label="⬇️ Download Structured Report (JSON)",
            data=report_json,
            file_name=f"{(st.session_state.processed_filename or 'document').rsplit('.', 1)[0]}_report.json",
            mime="application/json",
        )
        with st.expander("Raw JSON"):
            st.code(report_json, language="json")

# ---------------- CHAT TAB ----------------
with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the uploaded document...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        if st.session_state.vector_db is None:
            answer = "⚠️ Please upload and process a document first."
        else:
            with st.spinner("Thinking..."):
                answer = rag.ask_question(question, st.session_state.vector_db)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})