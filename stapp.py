import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
import uuid

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# ---------- PDF Processing Functions ----------

def extract_text_from_pdfs(pdf_files):
    """Extract text from uploaded PDF files."""
    text = ""
    for file in pdf_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except Exception as e:
                st.warning(f"⚠️ Could not read a page in the PDF: {e}")
    if not text.strip():
        st.error("❌ No extractable text found in the uploaded PDF.")
    return text


def split_text_into_chunks(text):
    """Split extracted text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def create_vector_store(chunks):
    """Embed text chunks and save them to a temporary session folder."""
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Create unique folder for each user session
    session_folder = os.path.join("temp_faiss", str(uuid.uuid4()))
    os.makedirs(session_folder, exist_ok=True)

    vector_store.save_local(session_folder)
    st.session_state["vector_path"] = session_folder
    return vector_store


# ---------- QA Chain ----------

def build_qa_chain():
    prompt_template = """
    You are a helpful assistant. Answer the question based on the provided context.
    If the context does not contain the answer, say "I don't know."
    
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3)
    return create_stuff_documents_chain(llm=model, document_variable_name="context", prompt=prompt)


def answer_user_question(question):
    if "vector_path" not in st.session_state:
        st.error("⚠️ You must process a PDF first before asking questions.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    try:
        vector_db = FAISS.load_local(st.session_state["vector_path"], embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        st.error("❌ Could not find the stored vector data. Please re-upload your PDF.")
        return

    docs = vector_db.similarity_search(question)
    chain = build_qa_chain()
    response = chain.invoke({"context": docs, "question": question})
    st.write("🧠 **Answer:**", response)


# ---------- Streamlit UI ----------

def main():
    st.set_page_config("PDF Chat Assistant")
    st.title("📄💬 Chat With Your PDF")
    st.caption("Upload a PDF and ask up to 3 questions using Google's Gemini model.")

    # Initialize session state
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    with st.sidebar:
        st.header("📂 Upload & Process")
        pdf_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

        if st.button("🚀 Process PDF"):
            if pdf_files:
                with st.spinner("Reading and indexing your PDF..."):
                    # Clear previous vector store
                    st.session_state.pop("vector_path", None)

                    text = extract_text_from_pdfs(pdf_files)
                    if text.strip():
                        chunks = split_text_into_chunks(text)
                        create_vector_store(chunks)
                        st.success("✅ PDF processed successfully! You can now ask questions.")
                        st.session_state.question_count = 0
            else:
                st.warning("Please upload at least one PDF file to continue.")

    # Question input area
    question = st.text_input("🔍 Ask a question about your PDF:")

    if question:
        if st.session_state.question_count < 3:
            answer_user_question(question)
            st.session_state.question_count += 1
            st.info(f"Question {st.session_state.question_count}/3 used.")
        else:
            st.warning("⛔ You’ve reached your limit of 3 questions for this session.")

if __name__ == "__main__":
    main()
