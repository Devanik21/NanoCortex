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
    
    def chunk_text(self, text, chunk_size=300):
        """Split text into chunks for better processing"""
        # Clean text more thoroughly
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\(\)]', ' ', text)
        text = text.strip()
        
        if len(text) < 50:
            return [text] if text else []
        
        # Split into paragraphs first, then sentences
        paragraphs = text.split('\n')
        chunks = []
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) < 20:
                continue
                
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph.strip())
            else:
                # Split long paragraphs into sentences
                sentences = re.split(r'[.!?]+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len(current_chunk + sentence) < chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 30]
        
        return chunks if chunks else [text[:500]]  # Fallback
    
    def train_on_text(self, text):
        """Create TF-IDF vectors from text chunks"""
        self.text_chunks = self.chunk_text(text)
        
        if not self.text_chunks:
            return False
            
        # Create TF-IDF vectorizer with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            strip_accents='unicode'
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
        
        # Lower threshold and provide more context
        max_similarity = similarities[top_indices[0]]
        
        if max_similarity < 0.05:  # Lower threshold
            # If no good matches, provide a general summary
            return f"I couldn't find specific information matching your question. Here's what the document contains:\n\n{self.text_chunks[0][:300]}..."
        
        # Combine top chunks for answer
        relevant_chunks = []
        for i in top_indices:
            if similarities[i] > 0.02:  # Even lower threshold
                relevant_chunks.append((self.text_chunks[i], similarities[i]))
        
        if not relevant_chunks:
            return f"Here's a sample from the document:\n\n{self.text_chunks[0][:400]}..."
        
        answer = f"Based on the document (confidence: {max_similarity:.3f}):\n\n"
        for i, (chunk, score) in enumerate(relevant_chunks[:2], 1):
            answer += f"**Section {i}** (relevance: {score:.3f}):\n{chunk}\n\n"
        
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
        
                if question:
                    with st.spinner("Finding answer..."):
                        answer = st.session_state.qa_system.answer_question(question)
                        
                        st.subheader("📝 Answer:")
                        st.write(answer)
                        
                        # Debug info
                        with st.expander("🔍 Debug Info"):
                            st.write(f"Total chunks: {len(st.session_state.qa_system.text_chunks)}")
                            st.write("Sample chunks:")
                            for i, chunk in enumerate(st.session_state.qa_system.text_chunks[:3]):
                                st.write(f"**Chunk {i+1}**: {chunk[:100]}...")
        
        # Quick question buttons
        st.subheader("💡 Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("What is this about?"):
                with st.spinner("Finding answer..."):
                    answer = st.session_state.qa_system.answer_question("What is this document about? What is the main topic?")
                    st.subheader("📝 Answer:")
                    st.write(answer)
        
        with col2:
            if st.button("Summarize"):
                with st.spinner("Finding answer..."):
                    answer = st.session_state.qa_system.answer_question("summarize main points key information")
                    st.subheader("📝 Answer:")
                    st.write(answer)
        
        with col3:
            if st.button("Key details"):
                with st.spinner("Finding answer..."):
                    answer = st.session_state.qa_system.answer_question("important details key facts main information")
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
