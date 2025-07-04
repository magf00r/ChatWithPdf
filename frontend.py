import streamlit as st
import requests
import random

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📓", layout="wide")
st.sidebar.header("📄 PDF Q&A (FastAPI + Streamlit)")
st.sidebar.write("It's a Gen-AI driven solution where you can chat with your document...")


LOADING_MESSAGES = [
    "Calculating your answer...",
    "Processing your request...",
    "Retrieving data, please wait...",
    "Loading, please hold on...",
    "Fetching information...",
    "Compiling your results...",
    "Please wait while we gather your data...",
    "Analyzing your query...",
    "Preparing your response...",
    "Loading data...",
    "Your request is being processed..."
]

# === Session State Setup ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# === Upload PDF ===
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    if uploaded_file.name != st.session_state.uploaded_filename:
        with st.spinner("📚 Indexing file and Storing in Vector DB..."):
            try:
                res = requests.post(
                    "http://localhost:8000/upload/",
                    files={"file": uploaded_file}
                )
                if res.ok:
                    result = res.json()
                    st.session_state.indexed = True
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.summary = result.get("summary", "")
                    st.sidebar.success(result.get("message", "PDF processed."))
                else:
                    st.error("❌ Failed to process PDF.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running. Please start the FastAPI server.")


# === Chat Interface ===
if st.session_state.indexed:
    user_input = st.chat_input("Ask a question based on the document...")
    
    # Always display the full chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
            
        answer = ""
        assistant = st.chat_message("assistant")
        
        with assistant:
            message_placeholder = st.empty()
            message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
            try:
                res = requests.post(
                    "http://localhost:8000/ask/",
                    data={
                        "question": user_input,
                        "filename": st.session_state.uploaded_filename,
                        "thread_id": "session_1"
                    }
                )
                if res.ok:
                    data = res.json()
                    answer = data.get("answer", "No answer found.")
                else:
                    answer = "❌ Error retrieving answer."
            except requests.exceptions.ConnectionError:
                answer = "❌ Backend connection error."
            message_placeholder.markdown(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
           
else:
    st.info("📂 Please upload a PDF file to begin.")

 
