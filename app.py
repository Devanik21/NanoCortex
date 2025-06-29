import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="PDF Q&A App",
    page_icon="📄",
    layout="wide"
)

class SimplePDFQA:
    def __init__(self):
        self.text_chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
    
    def chunk_text(self, text, chunk_size=500):
        """Split text into chunks for better processing"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Split into sentences roughly
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def train_on_text(self, text):
        """Create TF-IDF vectors from text chunks"""
        self.text_chunks = self.chunk_text(text)
        
        if not self.text_chunks:
            return False
            
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform text chunks
        self.tfidf_matrix = self.vectorizer.fit_transform(self.text_chunks)
        return True
    
    def answer_question(self, question, top_k=3):
        """Find most relevant text chunks and create answer"""
        if not self.vectorizer or not self.tfidf_matrix.shape[0]:
            return "No document loaded. Please upload a PDF first."
        
        # Transform question using same vectorizer
        question_vector = self.vectorizer.transform([question])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        if similarities[top_indices[0]] < 0.1:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Combine top chunks for answer
        relevant_chunks = [self.text_chunks[i] for i in top_indices if similarities[i] > 0.1]
        
        answer = "Based on the document:\n\n"
        for i, chunk in enumerate(relevant_chunks[:2], 1):
            answer += f"{i}. {chunk}\n\n"
        
        return answer

# Initialize the QA system
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = SimplePDFQA()
    st.session_state.document_loaded = False

# App header
st.title("📄 Simple PDF Q&A App")
st.markdown("Upload a PDF document and ask questions about its content!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                # Extract text from PDF
                text = st.session_state.qa_system.extract_text_from_pdf(uploaded_file)
                
                if text:
                    # Train the system on the text
                    success = st.session_state.qa_system.train_on_text(text)
                    
                    if success:
                        st.session_state.document_loaded = True
                        st.success("✅ PDF processed successfully!")
                        st.info(f"Document contains {len(st.session_state.qa_system.text_chunks)} text chunks")
                    else:
                        st.error("Failed to process the PDF content")
                else:
                    st.error("Could not extract text from PDF")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    if st.session_state.document_loaded:
        st.success("Document is ready! Ask any question about the content.")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="What is this document about?",
            height=100
        )
        
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("Finding answer..."):
                answer = st.session_state.qa_system.answer_question(question)
                
                st.subheader("📝 Answer:")
                st.write(answer)
    else:
        st.info("👈 Please upload and process a PDF document first using the sidebar.")

with col2:
    st.header("ℹ️ How it works")
    st.markdown("""
    1. **Upload PDF**: Choose your PDF file
    2. **Process**: Click 'Process PDF' to extract and analyze text
    3. **Ask Questions**: Type questions about the document
    4. **Get Answers**: The app finds relevant sections and provides answers
    
    **Note**: This uses text similarity matching to find relevant content from your PDF.
    """)
    
    if st.session_state.document_loaded:
        st.header("📊 Document Stats")
        st.metric("Text Chunks", len(st.session_state.qa_system.text_chunks))
        
        # Show sample of first chunk
        if st.session_state.qa_system.text_chunks:
            with st.expander("Preview first chunk"):
                st.text(st.session_state.qa_system.text_chunks[0][:200] + "...")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Simple PDF Q&A System")
