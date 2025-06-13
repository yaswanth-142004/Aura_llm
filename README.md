# Aura_LLM

Aura_LLM is a passion project aimed at building a custom Large Language Model (LLM) from scratch using only PyTorch. The focus of this model is to explore, understand, and generate content related to spiritual knowledge and consciousness — an area referred to here as "Aura Knowledge."

## 🌟 Project Vision

The objective of Aura_LLM is to create a purpose-driven language model that specializes in:
- Understanding spiritual texts and philosophies.
- Generating insights rooted in mindfulness, meditation, and metaphysics.
- Providing intelligent, ethical, and spiritually-aware responses.

## 🔧 Tech Stack

- **Framework**: PyTorch (no Hugging Face or external LLM libraries)
- **Language**: Python
- **Training**: Built-from-scratch tokenizers, transformers, and training loop
- **Deployment**: Local experimentation (initially), later plans for minimal web interface

## 🧠 Model Architecture

- Custom tokenizer
- Transformer blocks (Multi-head self-attention + Feed-forward)
- Positional encoding
- LayerNorm, Dropout, Residual connections
- Optimizer: AdamW
- Loss: CrossEntropyLoss (for autoregressive LM)

### Key Components:
- **GPTModel**: Main model, built using multi-head attention layers and transformer blocks.
- **TransformerBlock**: Core building block of the model, consisting of attention and feed-forward layers.
- **MultiHeadAttention**: Implements multi-head self-attention with a causal mask for autoregressive tasks.
- **LayerNorm**: Applies layer normalization to stabilize training.
- **FeedForward**: Implements a simple two-layer fully connected network after attention.
- **GELU Activation**: Applied after the first linear transformation in the feed-forward layer.

## 📚 Dataset

The model will be further scaled and be trained on a curated dataset of:
- Modern spiritual discourses
- Meditative writings and consciousness research

## 🚀 Status

- [x] Project initialized
- [x] Tokenizer logic complete
- [x] Initial transformer blocks implemented
- [x] Dataset preprocessing pipeline
- [x] Training loop
- [ ] Evaluation on spiritual QA


## 🙏 Purpose

This model isn't just about NLP — it's about merging modern AI with timeless wisdom. Aura_LLM seeks to be a mindful machine: a guide, not a guru.

## 📂 Code Overview

The core of the model is implemented using PyTorch with the following components:
- **`GPTModel`**: A neural network model implementing the GPT architecture with layers for embedding, positional encoding, multi-head attention, and a feed-forward network.
- **`TransformerBlock`**: A block that combines attention mechanisms and feed-forward networks with residual connections.
- **`MultiHeadAttention`**: Implements the multi-head attention mechanism essential for the transformer architecture.
- **`LayerNorm`**: A custom layer normalization used throughout the model to stabilize training.
- **`GELU`**: A custom activation function used in the feed-forward layer to add non-linearity.
- **`GPTDataSetV1`**: A dataset class that prepares the training and validation data using a sliding window approach, ensuring sequences are tokenized correctly.
- **`train_model_simple`**: The main training loop function that handles model training with gradient accumulation.
- **`plot_losses`**: A function to visualize the training and validation loss during training epochs.
  
### Example Training Workflow

1. **Data Preparation**: The text data is split into training and validation sets, and a DataLoader is created for both.
2. **Model Training**: The model is trained for multiple epochs using the `train_model_simple` function, with regular evaluation on the validation set.
3. **Loss Tracking**: Training and validation losses are plotted to visualize the model's performance.
4. **Checkpointing**: Model and optimizer states are saved after training for future inference or fine-tuning.

### Code Snippet

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logit
```

### This project is a journey towards creating a language model that can generate spiritual insights. If you're interested in the intersection of AI and spirituality, feel free to reach out!

## 📚 Resources Used

This project utilizes a variety of resources, libraries, and tools to build the custom Large Language Model. Below are the key resources used:

### 1. **PyTorch**
   - **Link**: [https://pytorch.org/](https://pytorch.org/)
   - PyTorch is used as the core framework for building and training the model from scratch.

### 2. **Tiktoken**
   - **Link**: [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
   - A tokenization library for efficient tokenization based on GPT-2's tokenization process.

### 3. **Matplotlib**
   - **Link**: [https://matplotlib.org/](https://matplotlib.org/)
   - Used for plotting training and validation losses during the training process.

### 4. **Transformer Architecture**
   - **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - The foundational paper for understanding transformers, attention mechanisms, and their implementation in this model.

### 5. **GPT-2**
   - **Link**: [https://openai.com/research/language-unsupervised](https://openai.com/research/language-unsupervised)
   - GPT-2 serves as the architecture model inspiration for building this custom LLM.

### 6. **Hugging Face’s Transformer Documentation (For General Knowledge)**
   - **Link**: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
   - Although Hugging Face's Transformers library is not used directly, this resource helped in understanding transformer models and architecture.

### 7. **Python Official Documentation**
   - **Link**: [https://docs.python.org/](https://docs.python.org/)
   - For general Python syntax and functionality used across the project.


## *** Buildin LLM from scratch Book By 
- **Link**: [Building LLM From Scratch ](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167?sr=8-1&language=en_US&ref_=as_li_ss_tl)

