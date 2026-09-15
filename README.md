\# 📄 Document QA Assistant



A local, privacy-friendly \*\*Document Question-Answering Assistant\*\* built with Retrieval-Augmented Generation (RAG).



Upload your documents and ask questions about their content. The system retrieves relevant passages using semantic + keyword hybrid search, reranks them using a Cross-Encoder, and generates a grounded answer using a local Qwen instruction-tuned LLM.



\*\*No paid API key is required.\*\*



\---



\## ✨ Features



\- 📄 Supports PDF, TXT and Markdown documents

\- 🧩 Automatic document chunking with overlap

\- 🔎 Semantic search using Hugging Face embeddings

\- 🔤 Keyword-based retrieval for exact term matching

\- 🔀 Hybrid retrieval combining semantic and keyword signals

\- 🎯 Cross-Encoder reranking for better relevance

\- 🤖 Local Qwen LLM for answer generation

\- 🛡️ Grounded answers using retrieved document context

\- 🚫 Refuses to answer when sufficient information is not found

\- 📚 Source and evidence display

\- 🔒 Local processing — documents and models stay on your machine

\- 💰 No paid LLM API required



\---



\## 🏗️ Architecture



```text

User Uploads

&#x20;    │

&#x20;    ▼

PDF / TXT / MD Extraction

&#x20;    │

&#x20;    ▼

Document Chunking

&#x20;    │

&#x20;    ▼

Hugging Face Embeddings

(all-MiniLM-L6-v2)

&#x20;    │

&#x20;    ▼

ChromaDB Vector Store

&#x20;    │

&#x20;    │

User Question

&#x20;    ▼

Hybrid Retrieval

Semantic + Keyword Search

&#x20;    │

&#x20;    ▼

Cross-Encoder Reranking

&#x20;    │

&#x20;    ▼

Top Relevant Context

&#x20;    │

&#x20;    ▼

Local Qwen LLM

Qwen2.5-1.5B-Instruct

&#x20;    │

&#x20;    ▼

Answer + Sources + Evidence


🧠 RAG Pipeline

1\. Document Ingestion



Uploaded PDF, TXT and Markdown files are parsed and converted into text.



2\. Chunking



Documents are split into overlapping chunks so relevant information can be retrieved efficiently.



3\. Embedding Generation



Each chunk is converted into a vector representation using:



sentence-transformers/all-MiniLM-L6-v2

4\. Vector Storage



Embeddings and metadata are stored locally using:



ChromaDB

5\. Hybrid Retrieval



The system combines:



Semantic similarity

Keyword overlap



This helps with both conceptual questions and exact terminology.



6\. Cross-Encoder Reranking



Retrieved candidates are reranked using:



cross-encoder/ms-marco-MiniLM-L-6-v2

7\. Grounded Generation



The top-ranked document chunks are passed to:



Qwen/Qwen2.5-1.5B-Instruct



The model is instructed to answer only from the retrieved document context.



8\. Evidence



The application displays the source document and retrieved evidence used for the answer.



🛠️ Tech Stack

Technology	Purpose

Python	Core application

Streamlit	Web UI

ChromaDB	Local vector database

Sentence Transformers	Text embeddings

Cross-Encoder	Document reranking

Qwen 2.5 1.5B	Local LLM

Hugging Face Transformers	LLM inference

PyPDF	PDF text extraction

PyTorch	Local model execution

📁 Project Structure

Document-QA-Assistant/

│

├── documents/

├── chroma\_db/

│

├── app.py

├── config.py

├── llm.py

├── qa.py

├── rag.py

├── retrieval.py

│

├── requirements.txt

├── README.md

└── .gitignore

Core Modules



app.py



Streamlit user interface, document upload, chat interface and source/evidence display.



config.py



Central configuration for models, storage paths, chunk size and collection settings.



rag.py



Document loading, chunking, embedding generation, ChromaDB indexing and semantic retrieval.



retrieval.py



Hybrid retrieval and Cross-Encoder reranking.



llm.py



Local Qwen model loading, prompt construction and grounded answer generation.



qa.py



Connects retrieval, reranking and LLM generation into the complete question-answering pipeline.



🚀 Getting Started

1\. Clone the Repository

git clone https://github.com/Aditya-Pratap-01/document-qa-assistant.git

cd document-qa-assistant

2\. Create a Virtual Environment

python -m venv .venv

3\. Activate the Environment

Windows PowerShell

.venv\\Scripts\\Activate.ps1

4\. Install Dependencies

pip install -r requirements.txt

5\. Run the Application

streamlit run app.py



The application will open in your browser.



📄 Supported Documents



The application currently supports:



.pdf

.txt

.md



Add your documents through the Streamlit interface and click Index Documents.



🔒 Privacy



This project is designed for local document processing.



Documents are indexed into a local ChromaDB database and the question-answering model runs locally.



No external LLM API key is required.



⚠️ Limitations



This is currently an MVP / portfolio implementation.



Potential future improvements include:



OCR support for scanned documents

Table-aware document extraction

Better chunking strategies

Query rewriting

Multi-query retrieval

BM25 + vector retrieval

Streaming LLM responses

Conversation memory

RAG evaluation metrics

GPU optimization

Multi-user support

🔮 Future Roadmap

Retrieval

BM25 + dense retrieval

Query expansion

Metadata filtering

Adaptive retrieval

Generation

Larger local instruction models

Streaming generation

Better citation attribution

Evaluation

Retrieval precision / recall

Answer faithfulness

Context relevance

Automated RAG evaluation

Product

Multi-document collections

Document management

User authentication

Chat history

Cloud deployment

👨‍💻 Author



Aditya Pratap



GitHub:

https://github.com/Aditya-Pratap-01



⭐ Support



If you find this project useful, consider giving the repository a star.



Built with Python, RAG and open-source AI.

