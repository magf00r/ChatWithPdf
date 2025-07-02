# 📘 PDF Q&A Chatbot – FastAPI + Streamlit + Groq + LangChain

This project is a full-stack AI-powered application that allows users to:

- Upload a PDF document (e.g., guides, manuals)
- Ask natural language questions about the content
- Get accurate answers from the document using LLMs
- Uses chat memory to remember previous interactions

---

## 🚀 Features

- 🔍 PDF text extraction with fallback to **OCR** via **Groq Scout**
- 🧠 Conversational memory using **LangGraph**
- ⚡ Vector search with **Spacy Embeddings** + **FAISS**
- 🤖 Answer generation using **Groq LLaMA 3.3**
- 📤 PDF upload and Q&A via **Streamlit frontend**
- ⚙️ Fast and responsive **FastAPI backend**

---

## 🧰 Tech Stack

| Layer       | Stack                      |
|-------------|----------------------------|
| Frontend    | Streamlit                  |
| Backend     | FastAPI                    |
| Embeddings  | Spacy (en_core_web_sm)     |
| LLM         | Groq (LLaMA-3.3 + Scout)   |
| Memory      | LangGraph + MemorySaver    |
| DB (Vector) | FAISS                      |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/magf00r/ChatWithPdf.git
cd ChatWithPdf
