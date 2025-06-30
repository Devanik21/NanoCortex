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

# Set page config
st.set_page_config(
    page_title="PDF Transformer Language Model",
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, max_seq_len=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        out = self.transformer(emb)
        out = self.norm(out[:, -1, :])
        logits = self.fc(out)
        return logits

class SmallLanguageModel: # Renamed class
    def __init__(self):
        self.model = None
        self.vocab_size = 0
        # Default configurable parameters
        self.max_sequence_length = 64 # Increased default
        self.max_vocab_size = 5000    # Increased default and made configurable limit
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.text_chunks = []
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        # Clean text - keep punctuation attached initially
        text = re.sub(r'([.,!?;:])', r' \1 ', text) # Add spaces around punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text) # Remove other special characters
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

        # Keep most common words up to max_vocab_size
        most_common = word_counts.most_common(self.max_vocab_size - 3) # Reserve space for special tokens

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

        # Use configurable max_sequence_length
        for i in range(len(words) - self.max_sequence_length):
            sequence = []
            for j in range(i, i + self.max_sequence_length + 1):
                word = words[j] if j < len(words) else '<END>'
                sequence.append(self.word_to_idx.get(word, 0))
            sequences.append(sequence)

        return sequences
    
    def train_model(self, text, epochs=15, batch_size=32, learning_rate=0.001, progress_bar=None, status_text=None):
        """Train the language model"""
        # Preprocess text and build vocabulary
        clean_text = self.preprocess_text(text)
        self.build_vocabulary(clean_text)

        if self.vocab_size < 50: # Still need a minimum vocabulary
            return False, "Not enough vocabulary to train model"

        # Create sequences
        sequences = self.text_to_sequences(clean_text)

        if len(sequences) < 20: # Still need minimum data
            return False, "Not enough training data"

        # Create dataset and dataloader
        dataset = TextDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the advanced model
        self.model = TransformerLM(
            self.vocab_size,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            max_seq_len=self.max_sequence_length,
            dropout=0.2
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        losses = []
        accuracies = []

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            self.model.train() # Set model to training mode

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100 * correct / total

            losses.append(avg_loss)
            accuracies.append(accuracy)

            # Update Streamlit UI elements if provided
            if progress_bar:
                progress_bar.progress((epoch + 1) / epochs)
            if status_text:
                status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

        self.is_trained = True
        return True, {'losses': losses, 'accuracies': accuracies}
    
    def generate_text(self, seed_text, max_length=50, temperature=0.8):
        """Generate text using the trained model"""
        if not self.is_trained or not self.model:
            return "Model is not trained yet!"
        
        self.model.eval()
        
        # Prepare seed
        # Preprocess seed text similarly to training data
        seed_text = self.preprocess_text(seed_text)
        words = seed_text.split()
        sequence = [self.word_to_idx.get(word, 0) for word in words[-self.max_sequence_length:]]
        
        # Pad if necessary
        while len(sequence) < self.max_sequence_length:
            sequence.insert(0, 0) # Pad with UNK token
        
        generated = words.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Predict next word
                x = torch.tensor([sequence], dtype=torch.long).to(self.device)
                output = self.model(x)
                
                # Apply temperature
                output = output / temperature
                probabilities = F.softmax(output, dim=-1)
                
                # Sample from distribution
                next_idx = torch.multinomial(probabilities, 1).item()
                next_word = self.idx_to_word.get(next_idx, '<UNK>')
                
                if next_word in ['<END>'] or len(generated) > max_length: # Stop on <END>
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
        # Preprocess question similarly
        question = self.preprocess_text(question)
        question_words = set(question.split())
        best_chunk = ""
        best_score = 0
        
        for chunk in self.text_chunks:
            # Preprocess chunk for comparison
            clean_chunk = self.preprocess_text(chunk)
            chunk_words = set(clean_chunk.split())
            overlap = len(question_words.intersection(chunk_words))
            # Simple score: overlap + count of question words present in chunk (case-insensitive)
            score = overlap + sum(1 for w in question_words if w in clean_chunk.split())
            if score > best_score:
                best_score = score
                best_chunk = chunk # Keep original chunk for context
        
        if best_score > 0:
            # Use the best chunk as context and generate response
            if self.is_trained and self.model:
                # Use a larger portion of the best chunk as seed, up to sequence length limit
                context_words = self.preprocess_text(best_chunk).split()
                seed_words = context_words[:self.max_sequence_length]
                seed_text = ' '.join(seed_words)

                generated = self.generate_text(seed_text, max_length=60) # Generate a bit longer answer
                return f"**Based on the document:**\n\n{generated}"
            else:
                return f"**Found relevant information:**\n\n{best_chunk}"
        else:
            if self.is_trained and self.model:
                # Generate based on question keywords if no relevant chunk found
                seed = ' '.join(list(question_words)[:min(5, len(question_words))]) # Use up to 5 keywords
                generated = self.generate_text(seed, max_length=40)
                return f"**AI Generated Response (based on keywords):**\n\n{generated}" # Clarify generation source
            else:
                return "Could not find relevant information in the document."


# Initialize the model - use SmallLanguageModel and slm session state key
if 'slm' not in st.session_state:
    st.session_state.slm = SmallLanguageModel()
    st.session_state.document_loaded = False
    st.session_state.model_trained = False

# App header
st.title("🧠 PDF Transformer Language Model (PyTorch)")
st.markdown("""
<div style="background-color:#181c20;padding:18px 24px 18px 24px;border-radius:10px;">
    <h2 style="color:#fff;">🚀 Welcome to the PDF Transformer Language Model App!</h2>
    <ul style="color:#b0b8c1;">
        <li><b>Upload any PDF</b> and instantly extract its text.</li>
        <li><b>Configure</b> vocabulary size, sequence length, and training parameters.</li>
        <li><b>Train</b> a modern <span style="color:#4fc3f7;">Transformer-based neural language model</span> on your document.</li>
        <li><b>Ask questions</b> about the document and get context-aware answers.</li>
        <li><b>Generate new text</b> in the style and context of your PDF.</li>
        <li>All computation runs <b>locally</b> on your device (CPU/GPU supported) – <span style="color:#4fc3f7;">no data leaves your computer</span>.</li>
    </ul>
    <p style="color:#b0b8c1;font-size:15px;">
        <b>How does it work?</b> This app uses PyTorch to build a Transformer neural network, similar to the architecture behind modern LLMs. It learns the patterns, style, and vocabulary of your PDF, enabling you to interact with your document in new ways.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for PDF upload and training
with st.sidebar:
    st.header("📁 Document & Training")
    st.markdown("""
    <div style="font-size:15px;background-color:#23272e;padding:10px 16px 10px 16px;border-radius:8px;color:#b0b8c1;">
    <b>Step 1:</b> <span style="color:#4fc3f7;">Upload a PDF</span>.<br>
    <b>Step 2:</b> <span style="color:#4fc3f7;">Configure</span> model and training.<br>
    <b>Step 3:</b> <span style="color:#4fc3f7;">Train</span> and <span style="color:#4fc3f7;">Explore</span>!
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to train the model"
    )
    
    if uploaded_file is not None:
        if st.button("📄 Process PDF", type="primary"):
            with st.spinner("Extracting text from PDF..."):
                # Use slm session state key
                text = st.session_state.slm.extract_text_from_pdf(uploaded_file)
                
                if text and len(text) > 100:
                    # Use slm session state key
                    st.session_state.slm.raw_text = text
                    st.session_state.document_loaded = True
                    st.session_state.model_trained = False # Reset trained status on new doc
                    st.success("✅ PDF processed successfully!")
                    st.info(f"📊 Text length: {len(text):,} characters")
                    
                    # Show text preview
                    with st.expander("📖 Text Preview"):
                        st.text(text[:500] + "..." if len(text) > 500 else text)
                else:
                    st.error("Could not extract sufficient text from PDF")
    
    if st.session_state.document_loaded and not st.session_state.model_trained:
        st.header("🧠 Train Model")
        
        # Add configurable parameters
        st.subheader("⚙️ Model Configuration")
        max_vocab_size = st.slider("Max Vocabulary Size", 1000, 10000, st.session_state.slm.max_vocab_size, 500)
        max_sequence_length = st.slider("Sequence Length", 30, 128, st.session_state.slm.max_sequence_length, 8)

        # Update model instance with chosen parameters
        st.session_state.slm.max_vocab_size = max_vocab_size
        st.session_state.slm.max_sequence_length = max_sequence_length


        st.subheader("🏋️ Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 5, 100, 30)
        with col2:
            batch_size = st.selectbox("Batch Size", [4,8,16, 32, 64], index=1)
        
        learning_rate = st.select_slider(
            "Learning Rate", 
            options=[0.01, 0.005, 0.001, 0.0005], 
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )
        
        if st.button("🚀 Train Language Model", type="primary"):
            with st.spinner(f"Preparing for training..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

            # Start training - use slm session state key
            success, result = st.session_state.slm.train_model(
                st.session_state.slm.raw_text,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_bar=progress_bar, # Pass progress bar
                status_text=status_text    # Pass status text
            )

            # Clear spinner and final status text after training
            status_text.empty()
            progress_bar.empty()

            if success:
                st.session_state.model_trained = True
                st.success("🎉 Model trained successfully!")

                # Show training stats
                final_loss = result['losses'][-1]
                final_acc = result['accuracies'][-1]
                st.metric("Final Loss", f"{final_loss:.3f}")
                st.metric("Final Accuracy", f"{final_acc:.1f}%")

                # Plot training progress
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

                ax1.plot(result['losses'])
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')

                ax2.plot(result['accuracies'])
                ax2.set_title('Training Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')

                st.pyplot(fig)
            else:
                st.error(f"Training failed: {result}")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    if st.session_state.model_trained:
        st.header("🤖 AI Assistant")
        st.success("🧠 Neural model is trained and ready!")
        st.markdown("""
        <div style="background-color:#23272e;padding:12px 18px 12px 18px;border-radius:8px;">
        <b style="color:#fff;">What can you do?</b>
        <ul style="color:#b0b8c1;">
            <li>📝 <b>Summarize</b> your document</li>
            <li>🔍 <b>Extract main topics</b> and <b>key points</b></li>
            <li>❓ <b>Ask custom questions</b> and get context-aware answers</li>
            <li>✨ <b>Generate new text</b> in the style of your PDF</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick action buttons
        st.subheader("⚡ Quick Actions")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📋 Summarize"):
                with st.spinner("Generating summary..."):
                    # Use slm session state key
                    answer = st.session_state.slm.answer_question("summarize main points key information")
                    st.write(answer)
        
        with col_b:
            if st.button("🔍 Main Topic"):
                with st.spinner("Finding main topic..."):
                    # Use slm session state key
                    answer = st.session_state.slm.answer_question("what is this document about main topic")
                    st.write(answer)
        
        with col_c:
            if st.button("💡 Key Points"):
                with st.spinner("Extracting key points..."):
                    # Use slm session state key
                    answer = st.session_state.slm.answer_question("important details key facts")
                    st.write(answer)
        
        st.markdown("---")
        
        # Question answering
        st.subheader("❓ Ask Questions")
        question = st.text_area(
            "Ask anything about the document:",
            placeholder="What are the main conclusions?",
            height=80
        )
        
        if st.button("🔍 Get Answer") and question:
            with st.spinner("Generating answer..."):
                # Use slm session state key
                answer = st.session_state.slm.answer_question(question)
                st.write(answer)
        
        st.markdown("---")
        
        # Text generation
        st.subheader("✨ Generate Text")
        seed_text = st.text_input(
            "Enter seed text:",
            placeholder="The main findings show that..."
        )
        
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            max_length = st.slider("Max Length", 20, 200, 80) # Increased max length for generation
        with col_gen2:
            temperature = st.slider("Creativity", 0.3, 1.5, 0.8, 0.1)
        
        if st.button("📝 Generate") and seed_text:
            with st.spinner("Generating text..."):
                # Use slm session state key
                generated = st.session_state.slm.generate_text(
                    seed_text, 
                    max_length=max_length,
                    temperature=temperature
                )
                st.subheader("🎯 Generated Text:")
                st.write(generated)
    
    elif st.session_state.document_loaded:
        st.info("📄 Document loaded! Now train the language model using the sidebar.")
        st.markdown("""
        <div style="background-color:#222b36;padding:10px 16px 10px 16px;border-radius:8px;">
        <b style="color:#fff;">Next:</b> Configure your model and start training.<br>
        <i style="color:#b0b8c1;">Tip: Larger vocabulary and sequence length = more expressive model, but slower training.</i>
        </div>
        """, unsafe_allow_html=True)
        
        # Show document stats - use slm session state key
        if hasattr(st.session_state.slm, 'raw_text'):
            text = st.session_state.slm.raw_text
            st.subheader("📊 Document Statistics")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Characters", f"{len(text):,}")
            with col_stat2:
                st.metric("Words", f"{len(text.split()):,}")
            with col_stat3:
                st.metric("Sentences", f"{len(re.split(r'[.!?]+', text)):,}")
    else:
        st.info("👈 Upload a PDF document first to get started.")
        st.markdown("""
        <div style="background-color:#23272e;padding:10px 16px 10px 16px;border-radius:8px;">
        <b style="color:#fff;">What is this app?</b><br>
        <span style="color:#b0b8c1;">
        This app lets you build a custom neural language model from any PDF, right in your browser.<br>
        <ul>
            <li>No cloud, no data sharing.</li>
            <li>Great for research, summarization, and creative writing.</li>
        </ul>
        </span>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.header("ℹ️ How it Works")
    st.markdown("""
    <div style="background-color:#181c20;padding:12px 18px 12px 18px;border-radius:8px;">
    <b style="color:#fff;">App Pipeline:</b>
    <ol style="color:#b0b8c1;">
        <li><b>PDF Extraction:</b> Reads and cleans your PDF text.</li>
        <li><b>Tokenization & Vocabulary:</b> Builds a custom vocabulary from your document.</li>
        <li><b>Sequence Creation:</b> Splits text into training sequences for the model.</li>
        <li><b>Transformer Training:</b> Trains a multi-layer Transformer neural network on your data.</li>
        <li><b>Interactive Inference:</b> Lets you ask questions and generate new text based on your PDF.</li>
    </ol>
    <b style="color:#fff;">Why Transformers?</b>
    <ul style="color:#b0b8c1;">
        <li>Multi-head self-attention for deep context understanding</li>
        <li>Positional encoding for word order awareness</li>
        <li>Layer normalization for stable, fast training</li>
        <li>Highly parallelizable and scalable</li>
    </ul>
    <b style="color:#fff;">Use Cases:</b>
    <ul style="color:#b0b8c1;">
        <li>Summarize research papers, contracts, or reports</li>
        <li>Extract main ideas from books or articles</li>
        <li>Generate creative writing in the style of your document</li>
        <li>Build custom Q&A bots for your own data</li>
    </ul>
    <b style="color:#fff;">Limitations:</b>
    <ul style="color:#b0b8c1;">
        <li>Model is trained from scratch on your PDF (not a general LLM)</li>
        <li>Performance depends on document size and your hardware</li>
        <li>Best for single-document, focused tasks</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.header("📊 Model Stats")
    if hasattr(st.session_state.slm, 'vocab_size'):
        st.metric("Vocabulary", st.session_state.slm.vocab_size)
        st.metric("Sequence Length", st.session_state.slm.max_sequence_length)
        st.metric("Text Chunks", len(st.session_state.slm.text_chunks))
        st.metric("Device", str(st.session_state.slm.device).upper())
    st.markdown("""
    <div style="background-color:#23272e;padding:12px 18px 12px 18px;border-radius:8px;">
    <b style="color:#fff;">Model Training Details:</b>
    <ul style="color:#b0b8c1;">
        <li>Trained on your device using PyTorch</li>
        <li>Optimized for speed and efficiency</li>
        <li>Supports mixed precision and distributed training (if available)</li>
    </ul>
    <b style="color:#fff;">Document Processing:</b>
    <ul style="color:#b0b8c1;">
        <li>Extracts text, builds vocabulary, and creates training sequences</li>
        <li>Uses advanced regex and NLP techniques for clean extraction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    # ...existing code...

# Footer
st.markdown("---")
st.markdown("""
<div style="font-size:15px;background-color:#181c20;padding:10px 16px 10px 16px;border-radius:8px;">
🧠 <b style="color:#fff;">PyTorch Transformer Language Model</b> • <i style="color:#b0b8c1;">Train, explore, and interact with your own documents using modern neural networks – all locally!</i><br>
<span style="color:#666;">Created for research, learning, and fun. No data leaves your device.</span>
</div>
""", unsafe_allow_html=True)
""", unsafe_allow_html=True)
