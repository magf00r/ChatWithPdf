# 📄 ChatWithPdf – PDF Q&A Chatbot (Streamlit + FastAPI + Groq + OCR)

This project lets you upload **native or scanned PDFs**, extract the content using **OCR or text parsing**, and ask natural language questions. It answers using **Groq’s LLaMA3-70B**, powered by **LangChain**, **LangGraph**, **FAISS**, and **Streamlit + FastAPI**.

---

## 🚀 Features

- 🧠 Answers questions from your PDF using Groq's LLM (LLaMA3-70B)
- 📄 Extracts text from scanned PDFs using Tesseract OCR
- 📚 Embeds text with SpaCy + FAISS for fast retrieval
- 💬 Chat interface with context & memory using LangGraph
- 🖥️ FastAPI backend + Streamlit frontend
- 📑 Generates a summary of the document

---

## 🛠 Tech Stack

| Layer       | Tech Used                       |
|-------------|----------------------------------|
| LLM         | Groq (LLaMA3-70B)                |
| Vectorstore | FAISS + SpaCy Embeddings         |
| Memory      | LangGraph + LangChain            |
| OCR         | pdf2image + pytesseract          |
| Frontend    | Streamlit                        |
| Backend     | FastAPI                          |


---

📦 Installation Guide

1. Clone the Repo

git clone https://github.com/magf00r/ChatWithPdf.git
cd ChatWithPdf


---

2. Install System Dependencies

🖼️ Poppler (for pdf2image)

🪟 Windows:

1. Download: https://github.com/oschwartz10612/poppler-windows/releases/


2. Extract and copy the path to the bin directory (e.g., C:\poppler\Library\bin)


3. In backend.py, set:



poppler_path = r"C:\\poppler\\Library\\bin"

🍏 macOS:

brew install poppler

🐧 Linux:

sudo apt install poppler-utils


---

🔍 Tesseract OCR

🪟 Windows:

1. Download: https://pypi.org/project/pytesseract/


2. In backend.py, set:



pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

🍏 macOS:

brew install tesseract

🐧 Linux:

sudo apt install tesseract-ocr


---

3. Create & Activate Virtual Environment

python -m venv venv
# Activate
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows


---

4. Install Python Dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm


---

5. Create .env File

# .env
GROQ_API_KEY=your_groq_api_key_here


▶️ Running the Application

Step 1: Start the FastAPI Backend

uvicorn backend:app --reload

> 📍 Runs at: http://localhost:8000


---

Step 2: Start the Streamlit Frontend

streamlit run frontend.py

> 📍 Opens at: http://localhost:8501


---

👨‍💻 Author

Magfoor Ahmad
📧 magfoor.ah@gmail.com


