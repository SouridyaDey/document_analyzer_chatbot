import os
import streamlit as st
from rag_pipeline import load_and_split, create_vectorstore, get_rag_chain, ask_question

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.title("Document Analyzer Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# If PDF uploaded, prepare RAG pipeline
if uploaded_file:
    pdf_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Loading and splitting PDF..."):
        docs = load_and_split(pdf_path)
    with st.spinner("Creating FAISS vector store..."):
        retriever = create_vectorstore(docs)
    with st.spinner("Preparing RAG pipeline..."):
        st.session_state.rag_chain = get_rag_chain(retriever)

    st.success("RAG pipeline is ready! Start chatting below")

# Chat interface
question = st.text_input("Ask a question:")

if question:
    if st.session_state.rag_chain:
        with st.spinner("Generating answer..."):
            answer = ask_question(st.session_state.rag_chain, question, st.session_state.chat_history)
    else:
        # If no PDF uploaded, answer on its own
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model="qwen3:8b", temperature=0.2)
        answer = llm.invoke(question).content

    # Update session history
    st.session_state.chat_history.append({"user": question, "bot": answer})

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"You: {chat['user']}")
    st.markdown(f"Bot: {chat['bot']}")

# Footer
st.markdown(
    """
    <br><br>
    <div style='text-align: center; color: gray;'>
        Created by <strong>Souridya Dey</strong>
    </div>
    """,
    unsafe_allow_html=True
)
