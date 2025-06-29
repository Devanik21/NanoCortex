import streamlit as st
import PyPDF2
import numpy as np
import re
from collections import Counter
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="PDF Tiny Language Model",
    page_icon="🧠",
    layout="wide"
)

class TinyLanguageModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.vocab_size = 0
        self.max_sequence_length = 50
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.text_chunks = []
        self.is_trained = False
        
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
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Clean text
        text = re.sub(r'[^\w\s\.\,\?\!\:\;\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        
        # Split into chunks for training
        sentences = re.split(r'[.!?]+', text)
        self.text_chunks = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return text
    
    def build_vocabulary(self, text):
        """Build vocabulary from text"""
        words = text.split()
        word_counts = Counter(words)
        
        # Keep most common words
        most_common = word_counts.most_common(2000)
        
        # Build word-to-index mapping
        self.word_to_idx = {'<UNK>': 0, '<START>': 1, '<END>': 2}
        self.idx_to_word = {0: '<UNK>', 1: '<START>', 2: '<END>'}
        
        for i, (word, _) in enumerate(most_common):
            self.word_to_idx[word] = i + 3
            self.idx_to_word[i + 3] = word
        
        self.vocab_size = len(self.word_to_idx)
        
    def text_to_sequences(self, text):
        """Convert text to sequences of token indices"""
        words = text.split()
        sequences = []
        
        for i in range(len(words) - self.max_sequence_length):
            sequence = []
            for j in range(i, i + self.max_sequence_length + 1):
                word = words[j] if j < len(words) else '<END>'
                sequence.append(self.word_to_idx.get(word, 0))
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def create_model(self):
        """Create a simple LSTM language model"""
        model = Sequential([
            Embedding(self.vocab_size, 100, input_length=self.max_sequence_length),
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(128, dropout=0.2),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, text, epochs=10):
        """Train the language model"""
        # Preprocess text and build vocabulary
        clean_text = self.preprocess_text(text)
        self.build_vocabulary(clean_text)
        
        if self.vocab_size < 50:
            return False, "Not enough vocabulary to train model"
        
        # Create sequences
        sequences = self.text_to_sequences(clean_text)
        
        if len(sequences) < 10:
            return False, "Not enough training data"
        
        # Prepare training data
        X = sequences[:, :-1]  # Input sequences
        y = sequences[:, -1]   # Target words
        
        # Create and train model
        self.model = self.create_model()
        
        # Train with progress bar
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
        return True, history
    
    def generate_text(self, seed_text, max_length=100):
        """Generate text using the trained model"""
        if not self.is_trained or not self.model:
            return "Model is not trained yet!"
        
        # Prepare seed
        words = seed_text.lower().split()
        sequence = []
        
        for word in words[-self.max_sequence_length:]:
            sequence.append(self.word_to_idx.get(word, 0))
        
        # Pad if necessary
        while len(sequence) < self.max_sequence_length:
            sequence.insert(0, 0)
        
        generated = words.copy()
        
        for _ in range(max_length):
            # Predict next word
            x = np.array([sequence])
            predictions = self.model.predict(x, verbose=0)[0]
            
            # Sample from top predictions
            top_indices = np.argsort(predictions)[-5:]
            probabilities = predictions[top_indices]
            probabilities = probabilities / np.sum(probabilities)
            
            next_idx = np.random.choice(top_indices, p=probabilities)
            next_word = self.idx_to_word.get(next_idx, '<UNK>')
            
            if next_word in ['<END>', '<UNK>']:
                break
                
            generated.append(next_word)
            
            # Update sequence
            sequence = sequence[1:] + [next_idx]
        
        return ' '.join(generated)
    
    def answer_question(self, question):
        """Answer questions using both retrieval and generation"""
        if not self.text_chunks:
            return "No document loaded!"
        
        # Simple retrieval first
        question_words = set(question.lower().split())
        best_chunk = ""
        best_score = 0
        
        for chunk in self.text_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words.intersection(chunk_words))
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
        
        if best_score > 0:
            # Use the best chunk as context and generate response
            if self.is_trained:
                context = best_chunk[:100]  # First part as seed
                generated = self.generate_text(context, max_length=50)
                return f"Based on the document:\n\n{generated}"
            else:
                return f"Found relevant information:\n\n{best_chunk}"
        else:
            return "Could not find relevant information in the document."

# Initialize the model
if 'tlm' not in st.session_state:
    st.session_state.tlm = TinyLanguageModel()
    st.session_state.document_loaded = False
    st.session_state.model_trained = False

# App header
st.title("🧠 PDF Tiny Language Model Trainer")
st.markdown("Upload a PDF and train a small neural language model on its content!")

# Sidebar for PDF upload and training
with st.sidebar:
    st.header("📁 Document & Training")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to train the model"
    )
    
    if uploaded_file is not None:
        if st.button("📄 Process PDF", type="primary"):
            with st.spinner("Extracting text from PDF..."):
                text = st.session_state.tlm.extract_text_from_pdf(uploaded_file)
                
                if text and len(text) > 100:
                    st.session_state.tlm.raw_text = text
                    st.session_state.document_loaded = True
                    st.success("✅ PDF processed successfully!")
                    st.info(f"Text length: {len(text)} characters")
                else:
                    st.error("Could not extract sufficient text from PDF")
    
    if st.session_state.document_loaded and not st.session_state.model_trained:
        st.header("🧠 Train Model")
        
        epochs = st.slider("Training Epochs", 5, 30, 15)
        
        if st.button("🚀 Train Language Model", type="primary"):
            with st.spinner(f"Training model for {epochs} epochs..."):
                progress_bar = st.progress(0)
                
                success, result = st.session_state.tlm.train_model(
                    st.session_state.tlm.raw_text, 
                    epochs=epochs
                )
                
                progress_bar.progress(1.0)
                
                if success:
                    st.session_state.model_trained = True
                    st.success("🎉 Model trained successfully!")
                    
                    # Show training stats
                    if hasattr(result, 'history'):
                        final_loss = result.history['loss'][-1]
                        final_acc = result.history['accuracy'][-1]
                        st.metric("Final Loss", f"{final_loss:.3f}")
                        st.metric("Final Accuracy", f"{final_acc:.3f}")
                else:
                    st.error(f"Training failed: {result}")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    if st.session_state.model_trained:
        st.header("🤖 AI Assistant")
        st.success("Model is trained and ready!")
        
        # Question answering
        question = st.text_area(
            "Ask a question about the document:",
            placeholder="What is the main topic discussed?",
            height=80
        )
        
        if st.button("🔍 Get Answer") and question:
            with st.spinner("Generating answer..."):
                answer = st.session_state.tlm.answer_question(question)
                st.subheader("💬 Answer:")
                st.write(answer)
        
        st.markdown("---")
        
        # Text generation
        st.subheader("✨ Generate Text")
        seed_text = st.text_input(
            "Enter seed text to continue:",
            placeholder="The document discusses..."
        )
        
        if st.button("📝 Generate") and seed_text:
            with st.spinner("Generating text..."):
                generated = st.session_state.tlm.generate_text(seed_text, max_length=80)
                st.subheader("🎯 Generated Text:")
                st.write(generated)
    
    elif st.session_state.document_loaded:
        st.info("📄 Document loaded! Now train the language model using the sidebar.")
    else:
        st.info("👈 Upload a PDF document first to get started.")

with col2:
    st.header("ℹ️ How it Works")
    
    if st.session_state.model_trained:
        st.markdown("""
        **🧠 Tiny Language Model Trained!**
        
        **Architecture:**
        - Embedding Layer (100D)
        - 2x LSTM Layers (128 units)
        - Dense Layers with Dropout
        - Vocabulary: {} words
        
        **Capabilities:**
        - Answer questions about content
        - Generate text in document style
        - Learned patterns from your PDF
        """.format(st.session_state.tlm.vocab_size))
        
        st.header("📊 Model Stats")
        if hasattr(st.session_state.tlm, 'vocab_size'):
            st.metric("Vocabulary Size", st.session_state.tlm.vocab_size)
            st.metric("Text Chunks", len(st.session_state.tlm.text_chunks))
            st.metric("Sequence Length", st.session_state.tlm.max_sequence_length)
    
    else:
        st.markdown("""
        **Training Process:**
        1. 📄 Upload PDF document
        2. 🔤 Extract and tokenize text
        3. 📚 Build vocabulary from content
        4. 🧠 Train LSTM neural network
        5. 🤖 Use for Q&A and text generation
        
        **Model Architecture:**
        - Embedding layer for word vectors
        - LSTM layers for sequence learning
        - Dense layers for prediction
        - Trained specifically on your document
        """)

# Footer
st.markdown("---")
st.markdown("🧠 **Tiny Language Model** • Trains a real neural network on your PDF content!")
