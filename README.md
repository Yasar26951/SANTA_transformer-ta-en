# SANTA_transformer-ta-en

the model state are in below link download it and load in inference.py  place near inference.py
https://drive.google.com/file/d/1uY4bGDsPm5oX2mjghsFV9gHqsYUeAzTw/view?usp=sharing

📘 English → Tamil Neural Machine Translation
(Custom Transformer Built From Scratch in PyTorch)

This project implements a Transformer-based Neural Machine Translation (NMT) model from scratch to translate English sentences into Tamil using PyTorch.

🚀 Project Overview

Built the entire Transformer architecture from scratch (no HuggingFace or seq2seq shortcuts)

Implemented Multi-Head Attention, Positional Encoding, Masking, and FeedForward layers manually

Trained on 30k English–Tamil sentence pairs from the HuggingFace NLPC-UOM dataset

Used SentencePiece tokenizer (8,000 vocab) for subword-level encoding

Trained in ~2.5 hours on Google Colab Free GPU

Shows strong performance on frequent corpus words despite the small dataset

📊 Dataset

Dataset: NLPC-UOM/English-Tamil-Parallel-Corpus

Size: ~30,000 sentence pairs

Languages: English → Tamil

Tokenizer: SentencePiece (vocab size = 8000)

Special Tokens: <pad>, <bos>, <eos>

🏗️ Model Architecture

Encoder: 6 layers

Decoder: 6 layers

Embedding Dimension: 512

Feedforward Dimension: 2048

Attention Heads: 8

Dropout: 0.15

Positional Encoding: Sinusoidal

Loss: CrossEntropy Loss (ignore PAD)

Optimizer: Adam (β1 = 0.9, β2 = 0.98, eps = 1e-9)

Learning Rate Scheduler: Noam Schedule (warmup = 10,000 steps)

📈 Training Results

Training completed in ~2.5 hours on free-tier Google Colab GPU.

🔹 Loss Summary
Epoch	Train Loss	Test Loss
1	7.08	7.08
6	3.76	4.56
15 (best)	2.67	4.22
🔹 Observations

Performs very well on high-frequency words

Struggles on rare/unseen words due to small dataset

Demonstrates the power and efficiency of the Transformer architecture

🧪 Inference Example
Input:  "The project supports rural development."
Output: "இத்திட்டம் கிராமப்புற மேம்பாட்டிற்கு ஆதரவாக செயல்படுகிறது."


Greedy decoding used for text generation

Supports CPU & CUDA devices

🗂️ Project Structure
project

model.py

train.py

inference.py
 
config.json

transformer_model_15.pt

spm_en.model

 spm_ta.model
 
 README.md


✨ Features

Transformer Encoder–Decoder fully implemented manually

Sinusoidal positional embeddings

Padding mask + Subsequent mask support

Tokenization with SentencePiece

Noam learning rate scheduler

Modular and extensible PyTorch codebase

📌 Future Improvements

Add Beam Search decoding

Train on larger parallel corpora

Fine-tune multilingual models (mBART, MarianMT)

Deploy model to Hugging Face Spaces

Add BLEU / chrF evaluation metrics



# 🔍 Current Observation & Ongoing Work

During training on a ~30k sentence parallel corpus, the 6-layer, 8-head Transformer model exhibited signs of overfitting, where training loss continued to decrease while validation performance stagnated and began to degrade, indicating partial memorization of the dataset. This suggests that the model capacity is relatively high compared to the dataset size. To address this, the model architecture and key hyperparameters are being tuned, including reducing model depth and adjusting regularization settings.

Ongoing Work: The model will be retrained after the current academic examination period with optimized hyperparameters and improved regularization strategies to enhance generalization and achieve more stable performance.
