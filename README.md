# 📄 PDF Q&A Chatbot (Streamlit + FastAPI + Groq + LangGraph + OCR)

An AI-powered chatbot that lets you upload a PDF (including scanned documents), ask questions in natural language, and get accurate answers from the document content — powered by **LangChain**, **Groq LLM**, **LangGraph**, and **FAISS**.

---

## 🔍 Features

- ✅ Upload **native or scanned PDFs**
- ✅ Extract text using **PyPDF2** and **OCR (Tesseract + pdf2image)**
- ✅ Build vector index using **FAISS** + **SpaCy embeddings**
- ✅ Ask natural language questions in **chat format**
- ✅ Responses generated using **Groq LLM (LLaMA3)** with **LangGraph memory**
- ✅ Summary generation of uploaded PDFs
- ✅ Streamlit-based interactive UI
- ✅ FastAPI-based backend

---

## 🛠️ Tech Stack

| Layer        | Tech Used                        |
|--------------|----------------------------------|
| 🧠 LLM       | Groq (LLaMA3 70B via LangChain)  |
| 🔗 Orchestration | LangChain, LangGraph         |
| 📚 Vector DB | FAISS + SpaCy embeddings         |
| 📄 PDF Parsing | PyPDF2 + Tesseract OCR         |
| 🌐 Backend   | FastAPI                          |
| 🖥️ Frontend  | Streamlit                        |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-qa-chatbot.git
cd pdf-qa-chatbot
