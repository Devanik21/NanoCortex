# 🧠 Enhanced PDF Tiny Language Model

A sophisticated neural language model built with PyTorch that trains on PDF documents and generates coherent, contextually relevant text. This enhanced version features advanced architecture, improved training techniques, and state-of-the-art text generation methods.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.0+-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

### 🧠 Advanced Neural Architecture
- **Multi-layer LSTM**: 3-layer LSTM with 256 hidden units
- **Rich Embeddings**: 128-dimensional word embeddings
- **Layer Normalization**: Improved training stability
- **Multi-layer Output Head**: Enhanced representation learning
- **Proper Weight Initialization**: Orthogonal and Xavier initialization

### 🎯 Enhanced Training
- **AdamW Optimizer**: Superior optimization with weight decay
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Automatic overfitting prevention
- **Perplexity Monitoring**: Better language model evaluation metrics

### ✨ Advanced Text Generation
- **Top-K Sampling**: Intelligent vocabulary pruning
- **Nucleus (Top-P) Sampling**: Dynamic probability-based selection
- **Temperature Control**: Fine-tuned creativity adjustment
- **Context-Aware Generation**: Natural sentence boundary detection
- **Enhanced Preprocessing**: Improved tokenization and formatting

### 📱 Interactive Web Interface
- **Streamlit UI**: User-friendly web application
- **Real-time Training**: Live training metrics and visualization
- **Interactive Q&A**: Document-based question answering
- **Text Generation**: Multiple generation modes and parameters
- **Training Visualization**: Loss, accuracy, and perplexity plots

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)

### Install Dependencies
```bash
pip install streamlit torch torchvision PyPDF2 numpy matplotlib
```

### Alternative: Using requirements.txt
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file:
```
streamlit>=1.28.0
torch>=1.9.0
PyPDF2>=3.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## 🚀 Quick Start

1. **Clone or download the enhanced model code**

2. **Run the Streamlit application**:
```bash
streamlit run enhanced_pdf_llm.py
```

3. **Upload a PDF document** using the sidebar file uploader

4. **Train the model** with your desired parameters:
   - Epochs: 50-100 (early stopping included)
   - Batch Size: 16 (recommended)
   - Learning Rate: 0.002 (optimal for most documents)

5. **Generate text and ask questions** once training is complete!

## 📖 Usage Guide

### Training Your Model

1. **Upload PDF**: Choose a PDF with substantial text content (minimum 1000 words recommended)

2. **Configure Training Parameters**:
   ```
   Epochs: 80 (default, with early stopping)
   Batch Size: 16 (balanced memory and performance)
   Learning Rate: 0.002 (adaptive scheduling)
   ```

3. **Monitor Training**: Watch real-time metrics:
   - Loss curve (should decrease)
   - Accuracy (should increase)
   - Perplexity (should decrease, <50 is good)

### Text Generation

1. **Seed Text**: Provide starting text relevant to your document
2. **Adjust Parameters**:
   - **Max Length**: 50-100 tokens
   - **Temperature**: 0.7-0.8 for balanced creativity
   - **Top-K**: 40 for good diversity

### Question Answering

Simply type questions about your document content. The model uses:
- **Semantic Retrieval**: Finds relevant document sections
- **Neural Generation**: Creates contextually appropriate responses

## 🏗️ Architecture Details

### Model Architecture
```python
ImprovedTinyLM(
  (embedding): Embedding(vocab_size, 128)
  (lstm): LSTM(128, 256, num_layers=3, batch_first=True, dropout=0.3)
  (layer_norm): LayerNorm(256)
  (fc1): Linear(256, 128)
  (fc2): Linear(128, vocab_size)
  (dropout1): Dropout(0.3)
  (dropout2): Dropout(0.3)
)
```

### Key Components

| Component | Description | Purpose |
|-----------|-------------|---------|
| **Embedding Layer** | 128D word representations | Rich semantic encoding |
| **3-Layer LSTM** | Bidirectional sequence modeling | Long-term dependency capture |
| **Layer Normalization** | Training stabilization | Faster convergence |
| **Multi-layer Head** | Non-linear output mapping | Better classification |
| **Dropout Layers** | Regularization | Overfitting prevention |

## 📊 Performance Optimization

### Training Tips
- **Batch Size**: Start with 16, increase if you have more GPU memory
- **Learning Rate**: 0.002 works well for most documents
- **Early Stopping**: Let the model stop automatically when it stops improving
- **Vocabulary Size**: Automatically optimized based on document size

### Generation Tips
- **Temperature**: 
  - 0.5-0.7: More focused, factual
  - 0.8-1.0: More creative, diverse
  - 1.0+: Highly creative, potentially less coherent
- **Top-K**: 20-60 range works best
- **Seed Text**: Use document-relevant starting text for better results

## 🔧 Advanced Configuration

### Custom Training Parameters
```python
model = EnhancedLanguageModel()
success, history = model.train_model(
    text=document_text,
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)
```

### Custom Generation
```python
generated_text = model.generate_text(
    seed_text="Your starting text",
    max_length=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)
```

## 📈 Performance Metrics

### Training Metrics
- **Loss**: Cross-entropy loss (lower is better)
- **Accuracy**: Next-word prediction accuracy
- **Perplexity**: Language model quality (lower is better)

### Quality Indicators
- **Perplexity < 50**: Good model performance
- **Accuracy > 60%**: Decent learning
- **Stable Loss Curve**: Proper convergence

## 🐛 Troubleshooting

### Common Issues

**1. "Not enough vocabulary" Error**
- **Solution**: Use a larger PDF with more diverse text content
- **Minimum**: ~1000 unique words recommended

**2. "Not enough training data" Error**
- **Solution**: Upload a longer document or reduce sequence length
- **Minimum**: Document should have at least 50 sentences

**3. Poor Generation Quality**
- **Solutions**:
  - Train for more epochs
  - Reduce temperature for more focused output
  - Use document-relevant seed text
  - Check if perplexity is reasonable (<100)

**4. Training Too Slow**
- **Solutions**:
  - Reduce batch size if memory issues
  - Reduce vocabulary size for faster training
  - Use GPU if available

**5. Model Overfitting**
- **Solutions**:
  - Early stopping is automatic
  - Increase dropout if needed
  - Use smaller learning rate

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4GB | 8GB+ |
| **GPU** | None (CPU works) | GTX 1060+ / RTX 2060+ |
| **Storage** | 1GB | 2GB+ |
| **Python** | 3.8+ | 3.9+ |

## 🤝 Contributing

We welcome contributions! Here are some areas where you can help:

- **Model Architecture**: Implement attention mechanisms, transformers
- **Training**: Add more advanced optimization techniques
- **UI/UX**: Improve the Streamlit interface
- **Documentation**: Enhance tutorials and examples
- **Testing**: Add unit tests and benchmarks

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd enhanced-pdf-llm

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black enhanced_pdf_llm.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Streamlit Team**: For the amazing web app framework
- **OpenAI**: For inspiration from GPT architectures
- **Hugging Face**: For transformer implementations and ideas

## 📚 References & Further Reading

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🔮 Future Enhancements

### Planned Features
- **Attention Mechanisms**: Add self-attention for better context modeling
- **Transformer Architecture**: Option to use transformer blocks
- **Multi-document Training**: Train on multiple PDFs simultaneously
- **Fine-tuning**: Pre-trained model fine-tuning capabilities
- **Export Options**: Save trained models for later use
- **API Endpoint**: REST API for programmatic access

### Experimental Features
- **RAG Integration**: Retrieval-Augmented Generation
- **Multi-modal**: Support for images in PDFs
- **Summarization**: Automatic document summarization
- **Classification**: Document classification capabilities

---

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Troubleshooting section** above
2. **Review the Usage Guide** for best practices
3. **Create an issue** on GitHub with:
   - Error message (if any)
   - Document characteristics (size, type)
   - Training parameters used
   - System specifications

---

**Made with ❤️ and 🧠 by the Enhanced PDF LLM Team**

*Transform your PDFs into intelligent, conversational AI models!*
