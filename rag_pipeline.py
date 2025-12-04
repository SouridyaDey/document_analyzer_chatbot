from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history')

def load_and_split(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def create_vectorstore(docs):
    embeddings = OllamaEmbeddings(model='embeddinggemma')
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def get_rag_chain(retriever):
    llm = OllamaLLM(model="qwen3:8b", temperature=0.2)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If the answer is not in context, you can answer on your own.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
""")

    retrieval_chain = (
        {
            'context': RunnableLambda(
                lambda x: "\n\n".join(
                    [doc.page_content for doc in retriever.get_relevant_documents(x['question'])]
                )
            ),
            'question': itemgetter('question'),
            'chat_history': RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"])
        }
        | prompt
        | llm
    )

    return retrieval_chain

def ask_question(chain, query: str, chat_history=None):
    """
    Ask a question using the RAG chain. If chat_history is provided, feed it to memory.
    """
    if chat_history:
        # Populate LangChain memory with previous messages
        memory.chat_memory.messages = []
        for chat in chat_history:
            memory.save_context({"question": chat["user"]}, {"answer": chat["bot"]})

    response = chain.invoke({'question': query})
    answer_text = response.content if hasattr(response, "content") else str(response)

    # Update memory
    memory.save_context({"question": query}, {"answer": answer_text})

    return answer_text
