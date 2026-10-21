# Exam Master: Chat With Your PDFs

A Retrieval-Augmented Generation (RAG) app that answers questions using **only the PDFs you upload**, and shows the exact file and page each answer came from.

**Live demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app) <!-- replace with your deployed link -->

![App screenshot](screenshot.png) <!-- add a screenshot of the app to your repo and keep this filename -->

## What it does

- Upload one or more PDFs (lecture notes, textbooks, papers).
- Ask questions in plain English.
- Get an answer grounded in your documents, with **source citations** (file name and page number) and the exact passages the model used.
- If the answer isn't in your PDFs, the app says so instead of guessing.

## How it works

```
PDF upload → text extraction → chunking → embeddings → FAISS vector store
                                                              │
Question → retrieve top 4 relevant chunks → LLM (Groq) → answer + sources
```

1. **Extract:** pages are read from the uploaded PDFs with PyMuPDF.
2. **Chunk:** text is split into 1,000-character chunks with 150 characters of overlap, keeping each chunk's page number.
3. **Embed:** chunks are converted to vectors with a small local embedding model (FastEmbed, `BAAI/bge-small-en-v1.5`).
4. **Store:** vectors are kept in an in-memory FAISS index, separate for each user session.
5. **Retrieve and answer:** the 4 most relevant chunks are passed to an LLM through a LangChain retrieval chain, along with a prompt that tells the model to answer only from the provided context.

## Tech stack

| Part | Tool |
|---|---|
| Interface | Streamlit |
| Orchestration | LangChain |
| PDF parsing | PyMuPDF |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`) |
| Vector store | FAISS (in-memory) |
| LLM | Groq (`openai/gpt-oss-20b`) |

## Privacy and API keys

- **Your PDFs are processed in memory and are not stored.** Uploaded files are written to a temporary file for parsing and deleted immediately.
- **Bring your own Groq key.** Paste a free key from [console.groq.com](https://console.groq.com) into the sidebar. It is used only for your session and is not saved.

## Run it locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the small embedding model, so it may take a moment to start.

## Deploy your own copy

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, choose your fork, and set `app.py` as the main file.
4. Deploy. No secrets are needed because each user supplies their own Groq key.

## Project structure

```
.
├── app.py              # Streamlit app: upload, retrieval chain, UI
├── requirements.txt    # Python dependencies
└── README.md
```

## Known limitations

- **Scanned PDFs are not supported.** Image-only PDFs have no extractable text, and the app will tell you so.
- **No conversation memory.** Each question is answered on its own, without earlier questions as context.
- **Session-based storage.** Your documents are cleared when you refresh the page or close the tab.
- **Rate limits.** Answers depend on Groq's free-tier limits for your key.

## Ideas for improvement

- OCR support for scanned PDFs
- Chat history and follow-up questions
- A small evaluation set to measure answer accuracy
- Reranking and experiments with chunk size and embedding models
- Persistent storage so users can return to their documents

## Author

**Shanmendra Sai Gajula**
B.Tech student, KL University
[LinkedIn](https://www.linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)