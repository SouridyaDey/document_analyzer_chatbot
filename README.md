
# RAG Chatbot using LangChain and Ollama

This project implements a Retrieval-Augmented Generation (RAG) chatbot that can intelligently answer questions from uploaded PDFs using **LangChain**, **FAISS**, and **Ollama**.  
It integrates document loading, text chunking, embedding creation, and local model inference for efficient and context-aware responses with Qwen 3:8b LLM by Ollama.

---

## Features
- Upload PDF documents and query them conversationally  
- Uses **UnstructuredPDFLoader** for fast and accurate PDF parsing  
- Employs **FAISS** as the vector database for efficient similarity search  
- Powered by **Ollama** for local LLM inference  
- Designed with modular and privacy-conscious architecture
- Added Dockerfile for containerization

## Prerequisites and setup instructions
- Download and install Docker
- Install Ollama from *https://ollama.com/download*
- After installation. pull the qwen3:8b LLM by *ollama pull qwen3:8b*
- Run Ollama in a new terminal by *ollama serve*. Keep this terminal running.
- Pull the pre-built Docker image from DockerHub. The <username>/<Dockerimagename> is souridya7/ollama_rag_chatbot
- Run the Docker container by *docker run -it --network=host souridyadey/ollama-rag-chatbot* (For Linux)
