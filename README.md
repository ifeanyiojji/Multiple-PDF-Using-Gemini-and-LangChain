# Multiple PDF Chatbot Using Gemini and LangChain

A chatbot application that enables interactive Q&A over multiple PDF documents using Gemini and LangChain. This project integrates advanced language models with document processing to deliver concise and relevant answers based on the content of uploaded PDFs.

---

## 🔍 Project Overview

This app allows users to upload multiple PDF files and chat with the system to get answers extracted from the documents. It leverages:

- **Gemini**: Advanced language model for generating context-aware responses.
- **LangChain**: Framework for managing chains of language model prompts and document retrieval.
- PDF processing and embedding for semantic search and retrieval.

---

## ⚙️ Features

- Supports uploading and querying multiple PDF documents simultaneously.
- Uses semantic search with embeddings for accurate information retrieval.
- Limits user to **3 tries per session** to encourage precise questioning and optimize resource usage.
- Real-world tested for robustness and practical use.

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ifeanyiojji/Multiple-PDF-Using-Gemini-and-LangChain.git
   cd Multiple-PDF-Using-Gemini-and-LangChain
    ```
2. Install Dependencies
   ```bash
    pip install -r requirements.txt
   ```
3. Make sure you have API keys and environment variables configured as needed for Gemini or LangChain (update .env or environment accordingly).


## 🚀 Usage

Run the main application:

```bash
streamlit run stapp.py
```

- Upload your PDF files using the interface.
- Ask questions related to the documents.
- You have 3 attempts to ask questions per session.
- The app will provide answers based on the PDF content using semantic search and    language modeling.

📄 ## File Structure
- `stapp.py` — Main Streamlit app script with the chatbot logic and UI.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.
- Other supporting files as needed.

⚠️ ## Limitations & Known Issues
-The 3 tries limit per session might restrict extended conversations.
-Depends on quality and clarity of PDF content for accuracy.
-No offline mode; requires internet connection for language model API.
-Currently optimized for English PDFs only.

🌟 ## Future Improvements
-Add user authentication and session management.
-Extend try limit or allow paid tiers for extended usage.
-Support additional document formats (Word, TXT).
-Improve multi-turn conversation context retention.
-Enhance UI/UX for smoother interaction.

👤 ## Author
Developed by Ifeanyi Ojji

📜 ## License
This project is licensed under the MIT License
