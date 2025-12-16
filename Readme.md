# Cyberpunk RAG-Powered AI Persona

The **Cyberpunk Oracle** is a Retrieval-Augmented Generation (RAG) system built on Streamlit, designed to function as an aggressive, street-smart AI character from the Cyberpunk universe (Night City). It provides highly constrained, context-aware answers to user queries using a local vector database of specialized lore, ensuring responses strictly adhere to the provided knowledge base and maintain a consistent, slang-heavy persona.

## Key Features

* **Custom RAG Pipeline:** Utilizes a custom, localized RAG architecture combining dense vector retrieval with a locally hosted Large Language Model (LLM).
* **Persona Enforcement:** Employs a highly constrained system prompt to ensure the LLM's entire output is immersed in Cyberpunk slang and attitude (`"choomba," "gonk," "preem,"` etc.).
* **Local LLM Integration:** Runs inference locally via **Ollama** using quantized models (e.g., Llama 3 8B), prioritizing speed and privacy.
* **Vector Search:** Implements **FAISS** for high-performance similarity search against a pre-computed index using `all-mpnet-base-v2` embeddings.
* **Conversational UI:** Provides a Streamlit-based interface with token-by-token streaming for real-time interaction and better perceived speed.

## Technology Stack

| Category | Component | Description |
| :--- | :--- | :--- |
| **Backend Framework** | Python 3.9+ | Primary programming language. |
| **User Interface** | Streamlit | Frontend for application deployment and conversational flow. |
| **LLM Orchestration** | LangChain | Used for constructing the RAG chain and managing prompt assembly. |
| **LLM Runtime** | Ollama | Platform for running local, open-source models (e.g., Llama 3). |
| **Vector Database** | FAISS | Efficient library for high-speed similarity search of dense vectors. |
| **Embedding Model** | `all-mpnet-base-v2` | Used for generating semantically accurate vector representations. |

## Setup and Installation

Follow these steps to get the Cyberpunk Oracle running locally on your machine.

### 1. Prerequisites

You must have the following installed:

* **Python 3.9+**
* **Ollama:** Download and install the application from the official site.

### 2. Model Installation (Ollama)

Open your terminal and pull the desired model. For best performance, use the quantized version:

```bash
ollama pull llama3:8b-instruct-q4_K_M

```
### 3. Clone and Environment Setup
```bash
# Clone the repository
git clone https://github.com/Rex1704/CyperpunkRAG.git
cd CyperpunkRAG

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate 

# Install required Python packages
pip install -r requirements.txt
```
### 4. Data and Indexes
For the RAG system to function, you must have the pre-computed vector indexes and metadata files in place.
Ensure the directory structure data/indexes/ exists.

Place all necessary knowledge base files (e.g., `".index"` and `".pkl"` files) into this folder.

### 5. Run the Application
Start the Streamlit application from your terminal:

```bash
streamlit run search_bot.py
```

## Customization and Development
### A. Performance
LLM Model: Modify the **OLLAMA_MODEL** variable in `search_bot.py` to test different quantized models from Ollama.

Slang Adherence: If the LLM loses its persona, strengthen the instructions and constraints in the `system_prompt` within the `_init_chain` method.

### B. Configuration

| Variable       | File                    | Description                                                  |
|----------------|-------------------------|--------------------------------------------------------------|
| `OLLAMA_MODEL`   | `search_bot.py` (Config)  | The local LLM model tag used by Ollama                        |
| `TOP_K_RETRIEVAL` | `search_bot.py` (Config)  | The number of documents retrieved from the index for context |
