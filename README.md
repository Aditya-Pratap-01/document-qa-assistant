# 📄 Document QA Assistant

A local, privacy-friendly **Document Question-Answering Assistant** built with **Retrieval-Augmented Generation (RAG)**.

Upload your documents and ask questions about their content. The system retrieves relevant passages using **semantic + keyword hybrid search**, reranks them using a **Cross-Encoder**, and generates a grounded answer using a **local Qwen instruction-tuned LLM**.

> 🔒 No paid LLM API key is required.

---

## ✨ Features

- 📄 Supports PDF, TXT and Markdown documents
- 🧩 Automatic document chunking with overlap
- 🔎 Semantic search using Hugging Face embeddings
- 🔤 Keyword-based retrieval for exact term matching
- 🔀 Hybrid retrieval combining semantic and keyword signals
- 🎯 Cross-Encoder reranking for improved relevance
- 🤖 Local Qwen LLM for answer generation
- 🛡️ Grounded answers using retrieved document context
- 🚫 Refuses to answer when sufficient information is not found
- 📚 Displays source documents and retrieved evidence
- 🔒 Local processing for privacy
- 💰 No paid LLM API required

---

## 🏗️ Architecture

```text
┌──────────────────────┐
│     User Uploads     │
│     PDF / TXT / MD   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Document Extraction  │
│      & Chunking      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Hugging Face         │
│ Embeddings           │
│ all-MiniLM-L6-v2     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      ChromaDB        │
│    Vector Store      │
└──────────┬───────────┘
           │
           │
      User Question
           │
           ▼
┌──────────────────────┐
│   Hybrid Retrieval   │
│ Semantic + Keyword   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Cross-Encoder        │
│     Reranking        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Relevant Context    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Local Qwen LLM    │
│ Qwen2.5-1.5B-Instruct│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Answer + Sources +   │
│ Evidence             │
└──────────────────────┘

RAG Pipeline

1. Document Ingestion
Uploaded PDF, TXT and Markdown files are parsed and converted into text.

2. Chunking
Documents are split into overlapping chunks so relevant information can be retrieved efficiently.

Current configuration:
Chunk size: 800 characters
Overlap: 150 characters

3. Embedding Generation
Each document chunk is converted into a vector representation using:
sentence-transformers/all-MiniLM-L6-v2  

4. Vector Storage
Embeddings and document metadata are stored locally using:
ChromaDB

5. Hybrid Retrieval
The system combines two retrieval signals:
Semantic similarity
Keyword overlap
This helps retrieve both conceptually relevant passages and passages containing important exact terms.

6. Cross-Encoder Reranking
Retrieved candidates are reranked using:
cross-encoder/ms-marco-MiniLM-L-6-v2
This provides a second relevance evaluation before generation.

7. Grounded Generation
The highest-ranked document chunks are passed to:
Qwen/Qwen2.5-1.5B-Instruct
The model is instructed to answer only using the retrieved document context.

8. Evidence
The application displays the source document and retrieved text chunks used as evidence for the answer.

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
├── chroma_db/
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

Streamlit user interface, document upload, chat interface, and source/evidence display.
config.py
Central configuration for models, storage paths, chunk size, overlap, and ChromaDB collection settings.

rag.py
Handles:
Document loading
PDF text extraction
Text chunking
Embedding generation
ChromaDB indexing
Semantic retrieval
retrieval.py

Handles:

Keyword scoring
Hybrid retrieval
Cross-Encoder reranking
Final document selection
llm.py

Handles:

Local Qwen model loading
Prompt construction
Grounded answer generation
qa.py

Connects retrieval, reranking, and LLM generation into the complete question-answering pipeline.

Getting Started
1. Clone the Repository
git clone https://github.com/Aditya-Pratap-01/document-qa-assistant.git
cd document-qa-assistant
2. Create a Virtual Environment
python -m venv .venv
3. Activate the Environment
Windows PowerShell
.venv\Scripts\Activate.ps1
4. Install Dependencies
pip install -r requirements.txt
5. Run the Application
streamlit run app.py

The application will open in your browser.

Supported Documents

The application currently supports:
.pdf
.txt
.md

Upload your documents through the Streamlit interface and click Index Documents.

Privacy:
This project is designed for local document processing.
Documents are indexed into a local ChromaDB database, and the question-answering model runs locally.
No external LLM API key is required.

Limitations
This is currently an MVP / portfolio implementation.

Current limitations include:

Scanned PDFs requiring OCR are not directly supported
Complex tables may require improved extraction
CPU-based local LLM inference can be slower
Larger documents may require further retrieval optimization
Retrieval quality depends on document structure and chunking

Future Roadmap:
Retrieval
BM25 + dense retrieval
Query expansion
Query rewriting
Metadata filtering
Adaptive retrieval
Multi-query retrieval
Generation
Larger local instruction models
Streaming responses
Improved citation attribution
Better answer verification
Evaluation
Retrieval precision / recall
Context relevance
Answer faithfulness
Automated RAG evaluation
Product
Multi-document collections
Document management
Chat history
User authentication
GPU optimization
Cloud deployment

👨‍💻 Author
Aditya Pratap

GitHub:
https://github.com/Aditya-Pratap-01

Support
If you find this project useful, consider giving the repository a star.
Built with Python, RAG and open-source AI.