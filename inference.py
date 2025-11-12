import torch
import sentencepiece as spm
import json
from model import Transformer
# Load the trained model
sp_en = spm.SentencePieceProcessor()
sp_en.load("spm_en.model")
sp_ta = spm.SentencePieceProcessor()
sp_ta.load("spm_ta.model")
# Load config
with open("config.json", "r") as f:
    config = json.load(f)

def en_encode(text):
  BOS,EOS=1,2
  token_ids = sp_en.encode(text, out_type=int);
  return [BOS]+token_ids+[EOS]


def en_decode(token_ids):
  decoded = sp_en.decode(token_ids[1:-1])
  return decoded


def ta_encode(text):
  BOS,EOS=1,2
  token_ids = sp_ta.encode(text, out_type=int);
  return [BOS]+token_ids+[EOS]


def ta_decode(token_ids):
  decoded = sp_ta.decode(token_ids[1:])
  return decoded

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    dmodel=config["d_model"],
    head=config["num_heads"],
    d_ff=config["d_ff"],
    vocab_size=config["vocab_size"],
    device=device,
    droprate=config["dropout_rate"]
).to(device)

# Load weights
model.load_state_dict(torch.load("transformer_model_15.pt", map_location=device))
model.eval()

# Sample Tamil input

# Convert to tensors

def greedy_decode(model, src, max_len=50, sos_id=1, eos_id=2):
    model.eval()

    # Make batch dimension if needed
    if src.dim() == 1:
        src = src.unsqueeze(0)

    # Start target with <SOS>
    tgt = torch.tensor([[sos_id]], device=src.device)

    for _ in range(max_len):
        # Forward pass
        out = model(src, tgt)             # (B, tgt_len, vocab)

        # Get last token prediction
        next_token = torch.argmax(out[:, -1, :], dim=-1)   # (B,)

        # Append
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        # Stop at <EOS>
        if next_token.item() == eos_id:
            break

    return tgt[0]   # return sequence
src_text = "நீங்கள் எப்படி இருக்கிறீர்கள்?" 
l=input("enter the word--") # "How are you?" in Tamil
src_ids =en_encode((l).lower())  # Convert to token IDs
tgt_ids = [2]  # Assume 2 is <BOS> token
print(src_ids)
src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)


tgt_tensor=greedy_decode(model,src_tensor)
# Decode output
print(tgt_tensor)
translated =sp_ta.decode(tgt_tensor.tolist())
print("🔤 Translated:", translated)