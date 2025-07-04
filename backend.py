from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import os
from dotenv import load_dotenv
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver

# === Constants & Setup ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

FAISS_FOLDER = "faiss_index"
TESSERACT_PATH = "Path to TESSERACT"
POPPLER_PATH = "Path to POPPLER"

os.makedirs(FAISS_FOLDER, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Helper Functions ===

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2 or fallback to OCR."""
    try:
        reader = PdfReader(pdf_bytes)
        text = "".join(page.extract_text() or "" for page in reader.pages).strip()
        if text:
            return text
    except Exception as e:
        print(f"[PDF Extract] Error: {e}")

    try:
        images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
        return "\n".join([pytesseract.image_to_string(img) for img in images]).strip()
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return ""

def create_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_FOLDER)


def load_vectorstore() -> Optional[FAISS]:
    try:
        return FAISS.load_local(FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)
    except:
        return None


def setup_langgraph():
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": response}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# === Endpoints ===

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_pdf_text(content)

    if not text:
        return {"status": "error", "message": "Text extraction failed"}

    create_vectorstore(text)
    return {"status": "success", "message": "PDF processed successfully"}


@app.post("/ask/")
async def ask_question(
    question: str = Form(...),
    thread_id: str = Form(default="default_thread")
):
    vectorstore = load_vectorstore()
    if not vectorstore:
        return {"status": "error", "message": "No document found. Please upload a PDF first."}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt Logic
    system_prompt = (
                f"You are a helpful AI assistant.\n"
                f"Use the following context to answer user question:\n\n{context}\n\n"
                "You must only use the above context and the previous conversation history to answer.\n"
                "Do not use any outside knowledge, even if the answer seems obvious to you.\n"
                "If the question is unrelated and the context or memory doesn't have the answer, reply:\n"
                "'I'm sorry, I don't know that based on the document. Could you please provide the answer so I can remember it?'"
                )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    # Memory-aware conversation
    input_messages = [HumanMessage(content=question)]
    prompt_input = prompt_template.format_messages(messages=input_messages)

    app_graph = setup_langgraph()
    output = app_graph.invoke(
        {"messages": prompt_input},
        config={"configurable": {"thread_id": thread_id}}
    )

    final_response = output["messages"][-1].content
    return {"answer": final_response}
 
