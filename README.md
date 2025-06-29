🧠 Enhanced PDF Tiny Language Model
This project presents an advanced Streamlit-based application for building, training, and interacting with a "Tiny Language Model" specifically designed to learn from the content of uploaded PDF documents. It features a robust PyTorch-backed neural network architecture, enhanced training methodologies, and sophisticated text generation capabilities, allowing users to summarize, ask questions, and generate text based on their documents.

✨ Features
PDF Text Extraction: Seamlessly extract text content from any uploaded PDF document.

Enhanced Text Preprocessing: Improved text cleaning, case preservation for proper nouns, and adaptive token handling for better vocabulary building.

Improved Tiny Language Model (PyTorch):

Neural Architecture:

Larger Embeddings: 128-dimensional embeddings for richer word representations.

Multi-Layer LSTM: A 3-layer LSTM with 256 hidden units for increased model capacity.

Multi-layer Output Head: A denser output layer for better representation learning.

Layer Normalization: Enhances training stability.

Better Weight Initialization: Orthogonal and Xavier uniform initialization for improved convergence.

Training Enhancements:

AdamW Optimizer: An advanced optimizer for better performance and regularization.

Learning Rate Scheduling: ReduceLROnPlateau for adaptive learning rate adjustment.

Gradient Clipping: Prevents exploding gradients during training.

Early Stopping: Automatically stops training when validation loss stops improving to prevent overfitting.

Perplexity Tracking: Monitors model performance more accurately.

Advanced Text Generation:

Temperature Control: Adjusts the randomness of generated text.

Top-K Sampling: Constrains the next word prediction to the k most probable words.

Top-P (Nucleus) Sampling: Selects words from the smallest set whose cumulative probability exceeds p.

Context-Aware Stopping: Improves the coherence and naturalness of generated sentences.

Better Punctuation Handling: More natural placement of punctuation in generated text.

Intelligent Question Answering: Retrieves relevant chunks from the document based on TF-IDF-like scoring and generates answers using the trained model.

Interactive UI with Streamlit: A user-friendly web interface for easy interaction.

Training Visualization: Plots for training loss, accuracy, and perplexity to monitor model learning.

🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/your-username/enhanced-pdf-tiny-lm.git
cd enhanced-pdf-tiny-lm

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Install dependencies:
Create a requirements.txt file in the root directory of your project with the following content:

streamlit
PyPDF2
numpy
torch
matplotlib

Then install them:

pip install -r requirements.txt

Running the Application
After installing the dependencies, you can run the Streamlit application:

streamlit run app.py

Replace app.py with the actual name of your main Streamlit script if it's different.

The application will open in your default web browser at http://localhost:8501.

💡 Usage
Upload PDF: In the sidebar, use the "Choose a PDF file" uploader to select your document. Click "Process PDF".

Train Model: Once the PDF is processed, adjust the Epochs, Batch Size, and Learning Rate in the sidebar, then click "Train Enhanced Model". The training progress and metrics will be displayed.

Interact with AI Assistant:

Quick Actions: Use the "Summarize", "Main Topic", or "Key Points" buttons for quick insights.

Ask Questions: Type your question in the text area and click "Get Answer" to query the document.

Generate Text: Provide a "seed text" and adjust Max Length, Creativity (Temperature), and Diversity (Top-K) to generate new text based on the document's content.

📂 Project Structure
.
├── app.py                  # Main Streamlit application code
├── requirements.txt        # Python dependencies
├── README.md               # This README file
└── .gitignore              # Files to ignore in Git

📈 Future Enhancements
Saving/Loading Models: Implement functionality to save trained models and load them later without re-training.

More Sophisticated Embeddings: Explore pre-trained word embeddings (e.g., Word2Vec, GloVe) or contextual embeddings (e.g., BERT, GPT-2) if computational resources allow.

Fine-tuning Pre-trained LMs: Instead of training from scratch, fine-tune a smaller pre-trained language model for better zero-shot performance.

Evaluation Metrics: Add more robust evaluation metrics for text generation (e.g., BLEU, ROUGE scores).

Multi-Document Processing: Allow training on multiple PDFs.

Interactive Training Progress: Update the Streamlit progress bar and status text more frequently during training.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details (if applicable, otherwise state it directly here).

Made with ❤️ for exploring the power of Tiny Language Models!
