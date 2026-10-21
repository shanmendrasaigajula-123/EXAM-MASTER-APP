import streamlit as st
import tempfile, os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq

st.set_page_config(page_title="Exam Master")
st.title("Your Exam Master...:)")

# Each user brings their own Groq key; it is kept only for their session
with st.sidebar:
    groq_api_key = st.text_input(
        "Your Groq API key", type="password",
        help="Create a free key at console.groq.com. It is used only for your session and not saved."
    )
    st.caption("Your PDFs are processed in memory and not stored.")

if not groq_api_key:
    st.info("Enter your Groq API key in the sidebar to start. You can get a free one at console.groq.com.")
    st.stop()


@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.file_names = []


def build_store(files, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            pages = PyMuPDFLoader(path).load()
        finally:
            os.remove(path)
        for p in pages:
            p.metadata["source"] = f.name
        chunks += splitter.split_documents(pages)
    if not chunks:
        raise ValueError("No readable text found. Scanned (image-only) PDFs are not supported yet.")
    return FAISS.from_documents(chunks, embeddings)


files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

names = [f.name for f in files] if files else []
if files and names != st.session_state.file_names:
    try:
        with st.spinner("Reading your PDFs..."):
            st.session_state.vectorstore = build_store(files, get_embeddings())
            st.session_state.file_names = names
    except ValueError as e:
        st.error(str(e))

question = st.text_input("Ask a question")

if question and st.session_state.vectorstore:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a Q&A assistant who answers questions using only the user's uploaded PDFs.
        Give accurate answers without making things up. If the answer is not in the
        context, say you could not find it in the uploaded PDFs.
        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    try:
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": question})
    except Exception:
        st.error("Something went wrong. Check that your Groq API key is valid and you haven't hit its rate limit.")
        st.stop()

    st.write(response["answer"])

    st.subheader("Sources")
    for d in response["context"]:
        st.caption(f"{d.metadata.get('source', 'Unknown PDF')}, page {d.metadata.get('page', 0) + 1}")
        with st.expander("View passage"):
            st.write(d.page_content)