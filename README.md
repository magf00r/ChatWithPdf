# 📘 PDF Q&A Chatbot – FastAPI + Streamlit + Groq + LangChain

This project is a full-stack AI-powered application that allows users to:

- 📄 Upload a PDF document (e.g., guides, manuals)
- ❓ Ask natural language questions about the content
- 🧠 Get accurate answers using Generative AI
- 🗣️ Uses chat memory to remember previous interactions

---

## 🚀 Features

- 🔍 Extracts text from PDFs using **PyPDF2**
- 🧾 Fallback **OCR with Groq Scout** for scanned PDFs
- 🧠 Chat memory powered by **LangGraph + MemorySaver**
- 🤖 Uses **Groq LLaMA-3.3** and **Scout** for fast, accurate LLM responses
- 🧠 Semantic search using **Spacy Embeddings** + **FAISS**
- 💬 Interactive **Streamlit chat interface**
- ⚙️ Lightweight **FastAPI backend**
- 📂 Upload PDF and ask multiple questions
- 🧠 Maintains conversation context per document

---

## 🧰 Tech Stack

| Layer       | Stack                                |
|-------------|--------------------------------------|
| Frontend    | Streamlit                            |
| Backend     | FastAPI                              |
| Embeddings  | Spacy (`en_core_web_sm`)             |
| LLM         | Groq (`LLaMA-3.3`, `Scout`)           |
| Memory      | LangGraph + MemorySaver              |
| OCR         | Groq Scout (multimodal model)        |
| Vector DB   | FAISS                                |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/magf00r/ChatWithPdf.git
cd ChatWithPdf
