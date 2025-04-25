import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
import uuid
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate    

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API Key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def get_pdf_text(pdf_file):
    text = ""
    for file in pdf_file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except Exception as e:
                st.warning(f"Could not read a page in the PDF: {e}")
    if not text.strip():
        st.error("No extractable text found in the uploaded PDF.")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Clear old vector store
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    # Save new one
    tmpdir = os.path.join("temp_faiss", str(uuid.uuid4()))
    os.makedirs(tmpdir, exist_ok=True)
    vector_store.save_local(tmpdir)

    st.session_state["faiss_path"] = tmpdir
    return vector_store


def get_conversational_chain():
    prompt_template = """You are a helpful assistant. Answer the question as detailed as possible based on the context provided.
    Make sure to provide all the relevant information from the context.
    If the context does not contain the answer, say "I don't know".
    Do not provide the wrong context or make up information.
    Context: \n {context} \n
    Question: \n {question} \n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3)
    return create_stuff_documents_chain(llm=model, document_variable_name="context", prompt=prompt)


def user_input(user_question):
    if "faiss_path" not in st.session_state:
        st.error("No PDF has been processed in this session. Please upload a PDF first.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    try:
        new_db = FAISS.load_local(st.session_state["faiss_path"], embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        st.error("Vector store not found. Please process the PDF first.")
        return
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"context": docs, "question": user_question})
    st.write("Response:", response)


def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("📄 Chat with Multiple PDF Using Gemini & LangChain")

    # Helpful hint for mobile users
    st.info("📱 On mobile? Tap the '≡' menu at the top-left to upload your PDF.")
    
    # Session state init
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    # Mobile-friendly PDF uploader
    with st.expander("📂 Upload PDF Here (Mobile Friendly)"):
        mobile_pdf = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True, key="mobile_upload")
        if st.button("🚀 Process PDF", key="mobile_process"):
            if mobile_pdf:
                with st.spinner("Processing..."):
                    text = get_pdf_text(mobile_pdf)
                    if text:
                        chunks = get_text_chunks(text)
                        get_vector_store(chunks)
                        st.success("✅ PDF processed successfully!")
                        st.session_state.question_count = 0
            else:
                st.warning("Please upload at least one PDF.")

    # Sidebar for desktop users
    with st.sidebar:
        st.title("Menu")
        pdf_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True, key="desktop_upload")
        if st.button("Process PDF", key="desktop_process"):
            if pdf_file:
                with st.spinner("Processing PDF..."):
                    text = get_pdf_text(pdf_file)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
                    st.success("PDF processed and vector store created.")
                    st.session_state.question_count = 0
            else:
                st.error("Please upload a PDF file.")

    # User question interface
    user_question = st.text_input("❓ Ask a question about your uploaded PDF:")

    if user_question:
        if st.session_state.question_count < 3:
            user_input(user_question)
            st.session_state.question_count += 1
            st.info(f"You have asked {st.session_state.question_count} of 3 allowed questions.")
        else:
            st.warning("⚠️ You’ve reached the limit of 3 questions per session.")


if __name__ == "__main__":
    main()
