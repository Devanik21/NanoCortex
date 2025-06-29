import streamlit as st
import PyPDF2
import numpy as np
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Enhanced PDF Tiny Language Model",
    page_icon="🧠",
    layout="wide"
)

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[-1], dtype=torch.long)

class ImprovedTinyLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=3, dropout=0.3):
        super(ImprovedTinyLM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Larger embeddings for richer representations
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Multi-layer LSTM with better capacity
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-layer output head for better representations
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        
        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights properly
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        # Embedding with improved initialization
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Use the last output and apply layer normalization
        last_output = lstm_out[:, -1, :]
        last_output = self.layer_norm(last_output)
        
        # Multi-layer output head
        output = self.dropout1(last_output)
        output = F.relu(self.fc1(output))
        output = self.dropout2(output)
        output = self.fc2(output)
        
        return output, hidden

class EnhancedLanguageModel:
    def __init__(self):
        self.model = None
        self.vocab_size = 0
        self.max_sequence_length = 40  # Increased for better context
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.text_chunks = []
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'losses': [], 'accuracies': [], 'perplexities': []}
        
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
        """Enhanced text preprocessing"""
        # Better text cleaning while preserving structure
        text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\'\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve case for proper nouns but lowercase most content
        sentences = re.split(r'(?<=[.!?])\s+', text)
        processed_sentences = []
        
        for sentence in sentences:
            # Keep first word capitalized, lowercase the rest except proper nouns
            words = sentence.split()
            if words:
                processed_sentence = []
                for i, word in enumerate(words):
                    if i == 0:
                        processed_sentence.append(word.lower().capitalize())
                    elif word.isupper() and len(word) > 1:  # Keep acronyms
                        processed_sentence.append(word)
                    elif word[0].isupper() and len(word) > 2:  # Likely proper noun
                        processed_sentence.append(word)
                    else:
                        processed_sentence.append(word.lower())
                processed_sentences.append(' '.join(processed_sentence))
        
        # Split into chunks for training (better sentence boundary detection)
        self.text_chunks = [s.strip() for s in processed_sentences if len(s.strip()) > 15]
        
        return ' '.join(processed_sentences)
    
    def build_vocabulary(self, text):
        """Enhanced vocabulary building with better token handling"""
        words = text.split()
        word_counts = Counter(words)
        
        # Keep more words but filter very rare ones
        min_frequency = max(2, len(words) // 1000)  # Adaptive threshold
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= min_frequency}
        most_common = Counter(filtered_words).most_common(2500)  # Increased vocab size
        
        # Build word-to-index mapping with special tokens
        self.word_to_idx = {
            '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3, 
            '<PERIOD>': 4, '<COMMA>': 5, '<QUESTION>': 6
        }
        self.idx_to_word = {
            0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>',
            4: '<PERIOD>', 5: '<COMMA>', 6: '<QUESTION>'
        }
        
        # Add punctuation mapping
        punct_map = {'.': '<PERIOD>', ',': '<COMMA>', '?': '<QUESTION>'}
        
        for i, (word, _) in enumerate(most_common):
            if word not in punct_map:  # Don't double-add punctuation
                self.word_to_idx[word] = i + 7
                self.idx_to_word[i + 7] = word
        
        self.vocab_size = len(self.word_to_idx)
        
    def text_to_sequences(self, text):
        """Enhanced sequence generation with better context windows"""
        words = text.split()
        sequences = []
        
        # Create overlapping sequences with stride
        stride = self.max_sequence_length // 4  # 75% overlap for better learning
        
        for i in range(0, len(words) - self.max_sequence_length, stride):
            sequence = []
            for j in range(i, i + self.max_sequence_length + 1):
                if j < len(words):
                    word = words[j]
                    # Handle punctuation
                    if word.endswith('.'):
                        sequence.append(self.word_to_idx.get(word[:-1], 1))
                        sequence.append(self.word_to_idx.get('<PERIOD>', 4))
                    elif word.endswith(','):
                        sequence.append(self.word_to_idx.get(word[:-1], 1))
                        sequence.append(self.word_to_idx.get('<COMMA>', 5))
                    elif word.endswith('?'):
                        sequence.append(self.word_to_idx.get(word[:-1], 1))
                        sequence.append(self.word_to_idx.get('<QUESTION>', 6))
                    else:
                        sequence.append(self.word_to_idx.get(word, 1))
                else:
                    sequence.append(self.word_to_idx.get('<END>', 3))
                    
                if len(sequence) > self.max_sequence_length:
                    break
            
            if len(sequence) == self.max_sequence_length + 1:
                sequences.append(sequence)
        
        return sequences
    
    def calculate_perplexity(self, loss):
        """Calculate perplexity from cross-entropy loss"""
        return torch.exp(torch.tensor(loss)).item()
    
    def train_model(self, text, epochs=50, batch_size=16, learning_rate=0.001):
        """Enhanced training with better optimization"""
        # Preprocess text and build vocabulary
        clean_text = self.preprocess_text(text)
        self.build_vocabulary(clean_text)
        
        if self.vocab_size < 100:
            return False, "Not enough vocabulary to train model"
        
        # Create sequences
        sequences = self.text_to_sequences(clean_text)
        
        if len(sequences) < 50:
            return False, "Not enough training data"
        
        # Create dataset and dataloader
        dataset = TextDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize improved model
        self.model = ImprovedTinyLM(self.vocab_size).to(self.device)
        
        # Better optimization setup
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training loop with better metrics
        self.training_history = {'losses': [], 'accuracies': [], 'perplexities': []}
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100 * correct / total
            perplexity = self.calculate_perplexity(avg_loss)
            
            self.training_history['losses'].append(avg_loss)
            self.training_history['accuracies'].append(accuracy)
            self.training_history['perplexities'].append(perplexity)
            
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                st.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.is_trained = True
        return True, self.training_history
    
    def generate_text(self, seed_text, max_length=50, temperature=0.8, top_k=40, top_p=0.9):
        """Enhanced text generation with nucleus sampling"""
        if not self.is_trained or not self.model:
            return "Model is not trained yet!"
        
        self.model.eval()
        
        # Prepare seed with better preprocessing
        words = seed_text.strip().split()
        sequence = []
        
        # Convert seed to indices
        for word in words[-self.max_sequence_length:]:
            sequence.append(self.word_to_idx.get(word, 1))
        
        # Pad if necessary
        while len(sequence) < self.max_sequence_length:
            sequence.insert(0, 0)
        
        generated = words.copy()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Predict next word
                x = torch.tensor([sequence], dtype=torch.long).to(self.device)
                output, hidden = self.model(x, hidden)
                
                # Apply temperature
                logits = output[0] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(0, top_k_indices, top_k_logits)
                    logits = logits_filtered
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
                
                # Sample from distribution
                probabilities = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probabilities, 1).item()
                next_word = self.idx_to_word.get(next_idx, '<UNK>')
                
                # Handle special tokens
                if next_word == '<END>':
                    break
                elif next_word == '<UNK>' and len(generated) > len(words):
                    break
                elif next_word in ['<PERIOD>', '<COMMA>', '<QUESTION>']:
                    if generated and not generated[-1].endswith(('.', ',', '?')):
                        punct_map = {'<PERIOD>': '.', '<COMMA>': ',', '<QUESTION>': '?'}
                        generated[-1] += punct_map.get(next_word, '')
                else:
                    generated.append(next_word)
                
                # Update sequence
                sequence = sequence[1:] + [next_idx]
                
                # Stop at sentence boundaries occasionally
                if next_word == '<PERIOD>' and len(generated) > 20 and np.random.random() < 0.3:
                    break
        
        return ' '.join(generated)
    
    def answer_question(self, question):
        """Enhanced question answering with better context retrieval"""
        if not self.text_chunks:
            return "No document loaded!"
        
        # Enhanced retrieval with TF-IDF-like scoring
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        best_chunks = []
        
        for chunk in self.text_chunks:
            chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
            
            # Calculate similarity score
            intersection = question_words.intersection(chunk_words)
            union = question_words.union(chunk_words)
            
            # Jaccard similarity + keyword density
            jaccard = len(intersection) / len(union) if union else 0
            keyword_density = sum(chunk.lower().count(word) for word in question_words) / len(chunk.split())
            
            score = jaccard + keyword_density
            
            if score > 0.1:  # Threshold for relevance
                best_chunks.append((chunk, score))
        
        # Sort by relevance and take top chunks
        best_chunks.sort(key=lambda x: x[1], reverse=True)
        
        if best_chunks:
            # Use best chunk as context for generation
            context = best_chunks[0][0]
            
            if self.is_trained and self.model:
                # Generate response based on context
                seed = context.split()[:15]  # Use first part as seed
                generated = self.generate_text(' '.join(seed), max_length=60, temperature=0.7)
                
                return f"**Based on the document:**\n\n{generated}"
            else:
                return f"**Found relevant information:**\n\n{context}"
        else:
            if self.is_trained and self.model:
                # Generate based on question
                generated = self.generate_text(question, max_length=50, temperature=0.8)
                return f"**AI Generated Response:**\n\n{generated}"
            else:
                return "Could not find relevant information in the document."

# Initialize the enhanced model
if 'elm' not in st.session_state:
    st.session_state.elm = EnhancedLanguageModel()
    st.session_state.document_loaded = False
    st.session_state.model_trained = False

# App header
st.title("🧠 Enhanced PDF Tiny Language Model")
st.markdown("Advanced neural language model with improved architecture and training!")

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
            with st.spinner("Extracting and preprocessing text..."):
                text = st.session_state.elm.extract_text_from_pdf(uploaded_file)
                
                if text and len(text) > 200:
                    st.session_state.elm.raw_text = text
                    st.session_state.document_loaded = True
                    st.success("✅ PDF processed successfully!")
                    st.info(f"📊 Text length: {len(text):,} characters")
                    
                    # Show text preview
                    with st.expander("📖 Text Preview"):
                        st.text(text[:500] + "..." if len(text) > 500 else text)
                else:
                    st.error("Could not extract sufficient text from PDF")
    
    if st.session_state.document_loaded and not st.session_state.model_trained:
        st.header("🧠 Enhanced Training")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 20, 200, 80)
        with col2:
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        
        learning_rate = st.select_slider(
            "Learning Rate", 
            options=[0.01, 0.005, 0.002, 0.001, 0.0005], 
            value=0.002,
            format_func=lambda x: f"{x:.4f}"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            st.info("Enhanced model uses better architecture, improved preprocessing, and advanced sampling techniques automatically.")
        
        if st.button("🚀 Train Enhanced Model", type="primary"):
            with st.spinner(f"Training enhanced model for up to {epochs} epochs..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success, result = st.session_state.elm.train_model(
                    st.session_state.elm.raw_text, 
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
                
                if success:
                    st.session_state.model_trained = True
                    st.success("🎉 Enhanced model trained successfully!")
                    
                    # Show final training stats
                    final_loss = result['losses'][-1]
                    final_acc = result['accuracies'][-1]
                    final_perplexity = result['perplexities'][-1]
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Final Loss", f"{final_loss:.3f}")
                    with col_m2:
                        st.metric("Accuracy", f"{final_acc:.1f}%")
                    with col_m3:
                        st.metric("Perplexity", f"{final_perplexity:.1f}")
                    
                    # Plot enhanced training progress
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                    
                    ax1.plot(result['losses'], 'b-', linewidth=2)
                    ax1.set_title('Training Loss', fontweight='bold')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(result['accuracies'], 'g-', linewidth=2)
                    ax2.set_title('Training Accuracy', fontweight='bold')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy (%)')
                    ax2.grid(True, alpha=0.3)
                    
                    ax3.plot(result['perplexities'], 'r-', linewidth=2)
                    ax3.set_title('Perplexity', fontweight='bold')
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Perplexity')
                    ax3.grid(True, alpha=0.3)
                    
                    # Learning curve
                    smoothed_loss = np.convolve(result['losses'], np.ones(5)/5, mode='valid')
                    ax4.plot(smoothed_loss, 'purple', linewidth=2)
                    ax4.set_title('Smoothed Learning Curve', fontweight='bold')
                    ax4.set_xlabel('Epoch')
                    ax4.set_ylabel('Smoothed Loss')
                    ax4.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.error(f"Training failed: {result}")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    if st.session_state.model_trained:
        st.header("🤖 Enhanced AI Assistant")
        st.success("🧠 Enhanced neural model is trained and ready!")
        
        # Quick action buttons
        st.subheader("⚡ Quick Actions")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📋 Summarize"):
                with st.spinner("Generating summary..."):
                    answer = st.session_state.elm.answer_question("summarize main points key information")
                    st.write(answer)
        
        with col_b:
            if st.button("🔍 Main Topic"):
                with st.spinner("Finding main topic..."):
                    answer = st.session_state.elm.answer_question("what is this document about main topic")
                    st.write(answer)
        
        with col_c:
            if st.button("💡 Key Points"):
                with st.spinner("Extracting key points..."):
                    answer = st.session_state.elm.answer_question("important details key facts conclusions")
                    st.write(answer)
        
        st.markdown("---")
        
        # Question answering
        st.subheader("❓ Ask Questions")
        question = st.text_area(
            "Ask anything about the document:",
            placeholder="What are the main conclusions and findings?",
            height=80
        )
        
        if st.button("🔍 Get Answer") and question:
            with st.spinner("Generating enhanced answer..."):
                answer = st.session_state.elm.answer_question(question)
                st.write(answer)
        
        st.markdown("---")
        
        # Enhanced text generation
        st.subheader("✨ Enhanced Text Generation")
        seed_text = st.text_input(
            "Enter seed text:",
            placeholder="The research findings indicate that..."
        )
        
        col_gen1, col_gen2, col_gen3 = st.columns(3)
        with col_gen1:
            max_length = st.slider("Max Length", 30, 150, 80)
        with col_gen2:
            temperature = st.slider("Creativity", 0.3, 1.5, 0.8, 0.1)
        with col_gen3:
            top_k = st.slider("Diversity (Top-K)", 10, 100, 40)
        
        if st.button("📝 Generate Enhanced") and seed_text:
            with st.spinner("Generating enhanced text..."):
                generated = st.session_state.elm.generate_text(
                    seed_text, 
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
                st.subheader("🎯 Enhanced Generated Text:")
                st.write(generated)
    
    elif st.session_state.document_loaded:
        st.info("📄 Document loaded! Now train the enhanced language model using the sidebar.")
        
        # Show document stats
        if hasattr(st.session_state.elm, 'raw_text'):
            text = st.session_state.elm.raw_text
            st.subheader("📊 Document Statistics")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Characters", f"{len(text):,}")
            with col_stat2:
                st.metric("Words", f"{len(text.split()):,}")
            with col_stat3:
                st.metric("Sentences", f"{len(re.split(r'[.!?]+', text)):,}")
            with col_stat4:
                st.metric("Paragraphs", f"{len([p for p in text.split('\n\n') if p.strip()]):,}")
    else:
        st.info("👈 Upload a PDF document first to get started.")

with col2:
    st.header("🔬 Enhanced Features")
    
    if st.session_state.model_trained:
        st.markdown("""
        **🧠 Enhanced Neural Architecture:**
        
        **Improvements:**
        - 🔤 Larger Embeddings (128D)
        - 🧠 3-Layer LSTM (256 units)
        - 🎯 Multi-layer Output Head
        - 📊 Layer Normalization
        - 🎮 Better Weight Initialization
        
        **Training Enhancements:**
        - 📈 AdamW Optimizer + Scheduler
        - 🎯 Gradient Clipping
        - 🛑 Early Stopping
        - 📊 Perplexity Tracking
        
        **Generation Features:**
        - 🎲 Top-K + Top-P Sampling
        - 🌡️ Temperature Control
        - 📝 Better Punctuation
        - 🎯 Context-Aware Stopping
        """)
        
        st.header("📊 Model Statistics")
        if hasattr(st.session_state.elm, 'vocab_size'):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Vocabulary", st.session_state.elm.vocab_size)
                st.metric("Sequence Length", st.session_state.elm.max_sequence_length)
            with col_s2:
                st.metric("Text Chunks", len(st.session_state.elm.text_chunks))
                st.metric("Device", str(st.session_state.elm.device).upper())
    
    else:
        st.markdown("""
        **🚀 Enhanced PyTorch Architecture:**
        
        **Key Improvements:**
        - 🧠 Deeper LSTM (3 layers vs 2)
        - 📈 Larger embeddings & hidden size
        - 🎯 Multi-layer output head
        - 📊 Layer normalization
        - 🎮 Proper weight initialization
        
        **Better Training:**
        - 🚀 AdamW optimizer
        - 📈 Learning rate scheduling
        - 🎯 Gradient clipping
        - 🛑 Early stopping
        - 📊 Perplexity monitoring
        
        **Enhanced Generation:**
        - 🎲 Top-K sampling
        - 🌊 Nucleus (Top-P) sampling
        - 🌡️ Temperature scaling
        - 📝 Better text formatting
        
        **Better Text Processing:**
        - 🔤 Improved tokenization
        - 📖 Sentence boundary detection
        - 🎯 Context-aware chunking
        """)

# Footer
st.markdown("---")
st.markdown("🧠 **Enhanced PyTorch Tiny Language Model** • Professional-grade neural architecture!")
