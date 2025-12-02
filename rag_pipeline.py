from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

# For document ingestion and text splitting
def load_and_split():
    loader = PyMuPDFLoader('data/KEIPL Handbook.pdf')
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    return splitter.split_documents(docs)

docs = load_and_split()

# creating vector store
embeddings = OllamaEmbeddings(model='embeddinggemma')
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize LLM and its memory
llm = OllamaLLM(model="qwen3:8b", temperature=0.2)
memory = ConversationBufferMemory(return_messages = True, memory_key = 'chat_history')

# ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
                                          
You are a helpful assistant. Use the following context to answer the question. If the answer is not in context or the context is empty, you can answer on your own.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
                                          
                                          
""")

# Defining LCEL chain
retrieval_chain = (
    {
        'context': RunnableLambda(
            lambda x: "/n/n".join([doc.page_content for doc in retriever.get_relevant_documents(x['question'])])
        ),
        'question': itemgetter('question'),
        'chat_history': RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"])
    }
    | prompt
    | llm
)

def ask_question(query: str):
    response = retrieval_chain.invoke({'question': query})
    answer_text = response.content if hasattr(response, 'content') else str(response)
    memory.save_context({'question': query}, {'answer': answer_text})
    return answer_text
