import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# 1. Sidebar: upload PDF and choose model
st.sidebar.title("📄 PDF to Q&A App")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
model_name = st.sidebar.selectbox("Choose an LLM", ["google/flan-t5-small", "google/flan-t5-base"])

# 2. On upload: read, chunk, embed, index
if uploaded_file:
    with st.spinner("Reading PDF... 📖"):
        reader = PdfReader(uploaded_file)
        raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(raw_text)

    # Embeddings & vector store
    with st.spinner("Embedding the document... 🔍"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(docs, embeddings)

    # LLM client
    llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature":0})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    st.success("✅ Document processed! You can now ask questions.")

    # 3. User query
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Thinking... 🤔"):
            answer = qa.run(question)
        st.markdown(f"**Answer:** {answer}")
else:
    st.write("Upload a PDF to begin.")

# 4. Footer
st.markdown("---")
st.markdown("Built with ❤️ and Streamlit + LangChain + FAISS.")
