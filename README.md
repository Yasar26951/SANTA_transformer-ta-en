# 🚀 SANTA_transformer-ta-en

## English → Tamil Neural Machine Translation

**Custom Transformer Built From Scratch in PyTorch**

---

## 📘 Project Description

This project implements a Transformer-based Neural Machine Translation (NMT) system from scratch to translate English sentences into Tamil. The entire architecture is manually built in PyTorch without relying on high-level frameworks such as HuggingFace Transformers or pre-built seq2seq libraries, enabling deep control and understanding of every component.

The goal of this project is to explore the effectiveness and limitations of the original Transformer architecture when trained on a relatively small real-world parallel corpus.

---

## 🔗 Pretrained Model

Download the trained model weights and place them in the same directory as `inference.py`:

Model Link:
SANTA_transformer-ta-en
[https://drive.google.com/file/d/1uY4bGDsPm5oX2mjghsFV9gHqsYUeAzTw/view?usp=sharing](https://drive.google.com/file/d/1uY4bGDsPm5oX2mjghsFV9gHqsYUeAzTw/view?usp=sharing)

---

## 🚀 Project Overview

* Built the entire Transformer architecture from scratch
* Implemented Multi-Head Attention, Positional Encoding, Masking, and FeedForward layers manually
* Trained on 30k English–Tamil sentence pairs
* Used SentencePiece tokenizer (8,000 vocab) for subword-level encoding
* Trained in ~2.5 hours on Google Colab Free GPU
* Demonstrates strong performance on frequent corpus words despite limited data

---

## 📊 Dataset

* Dataset: NLPC-UOM / English-Tamil-Parallel-Corpus
* Size: ~30,000 sentence pairs
* Languages: English → Tamil
* Tokenizer: SentencePiece
* Vocabulary Size: 8000
* Special Tokens: <pad>, <bos>, <eos>

---

## 🏗️ Model Architecture

| Component             | Value                            |
| --------------------- | -------------------------------- |
| Encoder Layers        | 6                                |
| Decoder Layers        | 6                                |
| Embedding Dimension   | 512                              |
| Feedforward Dimension | 2048                             |
| Attention Heads       | 8                                |
| Dropout               | 0.1                              |
| Positional Encoding   | Sinusoidal                       |
| Loss Function         | CrossEntropy (ignore PAD)        |
| Optimizer             | Adam (β1=0.9, β2=0.98, eps=1e-9) |
| Scheduler             | Noam (warmup = 10,000 steps)     |

---

## 📈 Training Results

Training completed in approximately 2.5 hours on Google Colab Free GPU.

### Loss Summary

| Epoch     | Train Loss | Validation Loss |
| --------- | ---------- | --------------- |
| 1         | 7.08       | 7.08            |
| 6         | 3.76       | 4.56            |
| 15 (Best) | 2.67       | 4.22            |

### Observations

* Strong performance on high-frequency words
* Degradation on rare or unseen tokens due to limited dataset size
* Highlights both the power and data dependency of Transformer architectures

---

## 🧪 Inference Example

**Input:**
The project supports rural development.

**Output:**
இத்திட்டம் கிராமப்புற மேம்பாட்டிற்கு ஆதரவாக செயல்படுகிறது.

* Decoding Strategy: Greedy Decoding
* Supports both CPU and CUDA devices

---

## 🗂️ Project Structure

```
project/
│
├── model.py
├── inference.py
├── config.json
├── transformer_model_15.pt
├── spm_en.model
├── spm_ta.model
├── README.md
```

---

## ✨ Key Features

* Transformer Encoder–Decoder fully implemented manually
* Sinusoidal positional embeddings
* Padding mask + Subsequent mask support
* SentencePiece tokenization
* Noam learning rate scheduler
* Modular and extensible PyTorch codebase
* GPU acceleration (CUDA)

---

## 🔍 Current Observation & Ongoing Work

During training on a ~30k sentence parallel corpus, the 6-layer, 8-head Transformer model exhibited signs of overfitting, where training loss continued to decrease while validation performance stagnated and began to degrade, indicating partial memorization of the dataset. This suggests that the model capacity is relatively high compared to the dataset size. To address this, the model architecture and key hyperparameters are being tuned, including reducing model depth and adjusting regularization settings.

**Ongoing Work:** The model will be retrained after the current academic examination period with optimized hyperparameters and improved regularization strategies to enhance generalization and achieve more stable performance.

---

## 📌 Future Improvements

* Implement Beam Search decoding
* Train on larger parallel corpora
* Fine-tune multilingual models (mBART, MarianMT)
* Deploy the system to Hugging Face Spaces
* Add BLEU / chrF evaluation metrics

---



This project demonstrates:

* Deep understanding of Transformer internals
* Research-oriented experimentation and analysis
* Ability to evaluate and improve model performance
* Strong foundation in NLP and Deep Learning



---

## 🚀 How to Run

### Inference

```bash
python inference.py --text "The project supports rural development."
```



---



