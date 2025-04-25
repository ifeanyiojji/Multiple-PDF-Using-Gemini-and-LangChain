import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate    

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_file):
    text = ""
    for file in pdf_file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
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
    vector_store.save_local("faiss_index")
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
    
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    
    response = chain.invoke(
        {"input_documents": docs, "question": user_question}
        ,return_only_outputs=True)
    
    print("Response:", response)
    st.write("Response:", response['output_text'])
    
def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with Multiple PDF Using Gemini and LangChain")
        
    user_question = st.text_input("Ask a question about the PDF:")
        
    if user_question:
        user_input(user_question)
            
    with st.sidebar:
        st.title("Menu")
        pdf_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                text = get_pdf_text(pdf_file)
                chunks = get_text_chunks(text)
                vector_store = get_vector_store(chunks)
                st.success("PDF processed and vector store created.")
        else:
            st.error("Please upload a PDF file.")
                
if __name__ == "__main__":
    main()