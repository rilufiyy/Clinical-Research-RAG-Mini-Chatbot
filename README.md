# 🧠 Clinical Research RAG Mini Chatbot

A mini Retrieval-Augmented Generation (RAG) chatbot designed to answer clinical psychology questions based on curated research articles, particularly focusing on **Nightmare Disorder**, **Imagery Rehearsal Therapy (IRT)**, and **Major Depressive Episode**.

This project demonstrates how combining vector-based retrieval with large language models can improve the accuracy, relevance, and reliability of AI-generated responses in sensitive domains such as mental health.

---

## 🚀 Project Overview

Traditional chatbots often generate answers based solely on pretrained knowledge, which can lead to hallucinations or outdated information.  
This project addresses that limitation by implementing a **Retrieval-Augmented Generation (RAG)** pipeline, where the model retrieves relevant research evidence before generating answers.

The chatbot:
- Retrieves context from clinical research documents
- Generates answers strictly based on retrieved evidence
- Avoids fabricating information outside the knowledge base

---

## 🏗️ System Architecture

1. **Document Loading**
   - Clinical research articles (PDF/text)
2. **Chunking**
   - Documents split into semantically meaningful chunks
3. **Embedding**
   - Text embeddings generated using HuggingFace sentence-transformers
4. **Vector Store**
   - FAISS used for efficient similarity search
5. **Retrieval**
   - Relevant chunks retrieved based on user queries
6. **Generation**
   - Groq-hosted LLM synthesizes answers using retrieved context only

---

## 🧩 Tech Stack

- **Programming Language**: Python
- **LLM**: Groq (LLaMA 3.1 models)
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Database**: FAISS
- **Framework**: LangChain (LCEL)
- **Document Loader**: PyMuPDF
- **Environment Management**: python-dotenv

---

## 📂 Project Structure

```text
.
├── data/                         # Clinical research documents
├── faiss_irt_nightmare_depression/  # Saved FAISS vector store
├── chatbot.py                    # Main RAG chatbot pipeline
├── knowledge_base_manager.py     # Knowledge base update & maintenance
├── .env                          # API keys (not committed)
├── requirements.txt
└── README.md
