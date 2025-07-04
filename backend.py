from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# FastAPI App
app = FastAPI()
FAISS_FOLDER = "faiss_index"
os.makedirs(FAISS_FOLDER, exist_ok=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR Setup
pytesseract.pytesseract.tesseract_cmd = "Path to Pytesseract"
POPPLER_PATH = "Path to poppler"

# LangChain components
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


# === OCR + Text Extraction ===
def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(pdf_bytes)
        text = "".join([page.extract_text() or "" for page in reader.pages]).strip()
        if text:
            return text
    except:
        pass
    # OCR fallback
    images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text.strip()


# === FAISS VectorStore ===
def create_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_FOLDER)


def load_vectorstore():
    return FAISS.load_local(FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)


# === LangGraph with Memory ===
def setup_langgraph(llm):
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": response}

    workflow.add_node("model", call_model)
    workflow.set_entry_point("model")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# === Upload PDF ===
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful AI assistant. Your task is to answer user question based on the given context:\n\n{context}\n\n"
                   f"If the question is not related to the document, reply: 'I'm sorry, I can only answer questions related to the provided document.'"),
        MessagesPlaceholder(variable_name="messages")
    ])

    input_messages = [HumanMessage(content=question)]
    prompt_input = prompt_template.format_messages(messages=input_messages)

    app_graph = setup_langgraph(llm)
    output = app_graph.invoke(
        {"messages": prompt_input},
        config={"configurable": {"thread_id": thread_id}}
    )

    response = output["messages"][-1]
    return {"answer": response.content}
 
