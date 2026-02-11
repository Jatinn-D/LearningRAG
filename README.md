# SimpleRAG Explorer
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-rag-explorer.streamlit.app/)

A lightweight **Retrieval-Augmented Generation (RAG)** system built to provide an LLM with external knowledge. This project demonstrates how to turn a simple text file into a searchable "knowledge base" using vector embeddings and local AI models.

## 🚀 Overview
Standard LLMs are limited by their training data cutoff. This project solves that by:
1.  **Ingesting** local text files.
2.  **Vectorizing** the data into numerical embeddings.
3.  **Retrieving** relevant context using Cosine Similarity.
4.  **Augmenting** the model's response with grounded facts.

## 🛠️ Tech Stack
* **Language:** Python 🐍
* **AI Engine:** [Ollama](https://ollama.com/)
* **Embedding Model:** `bge-base-en-v1.5`
* **Language Model:** `Llama-3.2-1B-Instruct`
* **UI Framework:** Streamlit
* **Version Control:** Git & GitHub

## 📋 Features
* **Local Processing:** Everything runs on your machine using Ollama—no API keys required.
* **Dynamic Context:** Upload different text files to change the chatbot's knowledge base.
* **Similarity Search:** Uses mathematical vector comparison to find the most relevant information.

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Jatinn-D/LearningRAG.git](https://github.com/Jatinn-D/LearningRAG.git)
   cd LearningRAG

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Ensure Ollama is running and models are pulled:**
   ```bash
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

4. **Run the application:**
   ```bash
   streamlit run RAG.py

#
**Developed by Jatin;)**
