
import torch.nn.functional as F
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ScaleDotProduct(nn.Module):
  def __init__(self,d_k,drop_rate=0.1):
    super(ScaleDotProduct,self).__init__()

    self.d_k=d_k
    self.drop_rate=drop_rate
    self.dropout=nn.Dropout(drop_rate)

  def forward(self,query,key,value,mask=None):

    score=torch.matmul(query,key.transpose(-1,-2))
    score=score/math.sqrt(self.d_k)
    if mask is not None:
      # Broadcast to match score: [batch_size, heads, seq_len, seq_len]
      score = score.masked_fill(mask, float('-inf'))

    score=F.softmax(score,dim=-1)
    score = torch.nan_to_num(score, nan=0.0)
    score=torch.matmul(self.dropout(score),value)
    return score


class MulitHeadAttention(nn.Module):
  def __init__(self,dmodel,head,drop=0.1):
    super(MulitHeadAttention,self).__init__()
    self.dmodel=dmodel
    self.d_k=dmodel//head
    self.head=head
    self.drop=drop
    self.query=nn.Linear(self.dmodel,self.dmodel)
    self.key=nn.Linear(self.dmodel,self.dmodel)
    self.value=nn.Linear(self.dmodel,self.dmodel)
    self.attention=ScaleDotProduct(self.d_k)
    self.Wo=nn.Linear(self.dmodel,self.dmodel)
    self.dropout=nn.Dropout(drop)

  def group_head(self,x):
    """ note i am only done this Combines multiple heads into a single embedding dimension
    # Input shape: (batch, seq_len, head, d_k)
    # Output shape: (batch, seq_len, head * d_k)"""
    return x.view(x.size(0),x.size(1),self.head*self.d_k)


  def group_split (self,x):
      """ return (batch,seq_len,head,d_k)"""
      return x.view(x.size(0),x.size(1),self.head,self.d_k)

  def forward(self, query, key, value, mask=None):
      Q = self.group_split(self.query(query))   # (B,L,H,d_k)
      K = self.group_split(self.key(key))       # (B,L,H,d_k)
      V = self.group_split(self.value(value))   # (B,L,H,d_k)

      # → (B,H,L,d_k)
      Q = Q.permute(0, 2, 1, 3).contiguous()
      K = K.permute(0, 2, 1, 3).contiguous()
      V = V.permute(0, 2, 1, 3).contiguous()

      # Attention
      out = self.attention(Q, K, V, mask=mask)   # (B,H,L,d_k)

      # (B,L,H,d_k)
      out = out.permute(0, 2, 1, 3).contiguous()

      #  (B,L,d_model)
      out = self.group_head(out)

      return self.Wo(self.dropout(out))


class FeedForward(nn.Module):
  def __init__(self,dmodel,d_ff,drop=0.1):
    super(FeedForward,self).__init__()
    self.dmodel=dmodel
    self.d_ff=d_ff
    self.drop=drop
    self.dropout=nn.Dropout(drop)
    self.layer1=nn.Linear(self.dmodel,self.d_ff)
    self.layer2=nn.Linear(self.d_ff,self.dmodel)



  def forward(self,x):
    x=F.relu(self.dropout(self.layer1(x)))
    x=self.layer2(x)
    return x



import math
class PositionalEmbedding(nn.Module):
  def __init__(self,d_model,max_len=300,drop=0.1):
    super(PositionalEmbedding,self).__init__()
    self.d_model=d_model
    self.maxlen=max_len
    self.dropout=nn.Dropout(drop)

    pos=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d_model,2).float()*(-1*math.log(10000))/self.d_model)
    pe=torch.zeros(self.maxlen,self.d_model)
    #torch.Size([256]) torch.Size([512, 1])
    pe[:,0::2]=torch.sin(pos/div)
    pe[:,1::2]=torch.cos(pos/div)
    #self.register_buffer('pe', self.pe.unsqueeze(0))
    #self.pe=self.pe.unsqueeze(0).to("cuda" if torch.cuda.is_available() else  "cpu")
    self.register_buffer("pe", pe.unsqueeze(0))
  def forward(self ,x):

    x=  x + self.pe[:,:x.size(1), :]
    return self.dropout(x)
  
class Encodelayer(nn.Module):
    def __init__(self,d_model,head,d_ff,dropout_rate=0.1):
        super(Encodelayer,self).__init__()
        self.d_model=d_model
        self.head=head
        self.d_ff=d_ff
        self.MHA=MulitHeadAttention(self.d_model,self.head,drop=dropout_rate)
        self.norm1=nn.LayerNorm(d_model, eps=1e-6)
        self.FF=FeedForward(self.d_model,self.d_ff,dropout_rate)
        self.norm2=nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self,x,mask=None):

        x1=self.MHA(x,x,x,mask)

        x=self.norm1(self.dropout(x1)+x)

        x1=self.FF(x)

        x=self.norm2(self.dropout(x1)+x)

        return x

class Encoder(nn.Module):
  def __init__(self,vocab_size,d_model,d_ff,head,drop_rate=0.15,max_len=5000):
    super(Encoder,self).__init__()
    self.vocab_size=vocab_size
    self.d_model=d_model
    self.d_ff=d_ff
    self.head=head
    self.embedding=nn.Embedding(self.vocab_size,self.d_model,padding_idx=0)
    self.position=PositionalEmbedding(self.d_model,max_len)
    self.layers= nn.ModuleList([Encodelayer(self.d_model,self.head,self.d_ff,dropout_rate=0.1) for _ in range(6)])
    self.norm=nn.LayerNorm(self.d_model)
  def forward(self,x,mask=None):
    x1=self.embedding(x)* math.sqrt(self.d_model)
    x=self.position(x1)
    for i in range(6):
      x=self.layers[i](x,mask)
    x=self.norm(x)
    return x
  


class DecoderLayer(nn.Module):
  def __init__(self,dmodel,head,d_ff,droprate=0.1):
    super(DecoderLayer,self).__init__()
    self.atten_layer=MulitHeadAttention(dmodel,head,droprate)
    self.norm1=nn.LayerNorm(dmodel)
    self.en_atten_layer=MulitHeadAttention(dmodel,head,droprate)
    self.norm2=nn.LayerNorm(dmodel)
    self.ff=FeedForward(dmodel,d_ff,droprate)
    self.norm3=nn.LayerNorm(dmodel)
    self.dropout=nn.Dropout(droprate)

  def forward(self,x,en_out,src_mask,tgt_mask):
    x1=self.atten_layer(x,x,x,mask=tgt_mask)
    x=self.norm1(self.dropout(x1)+x)
    x1=self.en_atten_layer(x,en_out,en_out,mask=src_mask)
    x=self.norm2(self.dropout(x1)+x)
    x1=self.ff(x)
    x=self.norm3(self.dropout(x1)+x)
    return x

class Decoder(nn.Module):
  def __init__(self,dmodel,head,d_ff,vocab_size,droprate=0.15):
    super(Decoder,self).__init__()
    self.dmodel=dmodel
    self.head=head
    self.vocab_size=vocab_size
    self.droprate=droprate
    self.embedding=nn.Embedding(self.vocab_size,self.dmodel,padding_idx=0)
    self.position=PositionalEmbedding(self.dmodel)
    self.layers=nn.ModuleList([DecoderLayer(self.dmodel,self.head,d_ff,droprate) for i in range(6)])
    self.norm=nn.LayerNorm(self.dmodel)
    self.dropout=nn.Dropout(droprate)



  def forward(self,x,ec_out,src_mask,tgt_mask):
    x=self.embedding(x)*math.sqrt(self.dmodel)
    x=self.position(x)

    for i in range(6):
      x=self.layers[i](x,ec_out,src_mask,tgt_mask)

    x=self.norm(x)
    return x
  

class Generator(nn.Module):
  def __init__(self,dmodel,vocab):
    super(Generator,self).__init__()
    self.layer=nn.Linear(dmodel,vocab)

  def forward(self,x):
    return self.layer(x)
  

import numpy as np
class Transformer(nn.Module):
  def __init__(self,dmodel,head,d_ff,vocab_size,device,droprate):
    super(Transformer,self).__init__()
    self.pad_idx = 0
    self.device=device
    self.encoder=Encoder(d_model=dmodel,head=head,vocab_size=vocab_size,d_ff=d_ff,drop_rate=droprate).to(device)
    self.decoder=Decoder(dmodel=dmodel,head=head,vocab_size=vocab_size,droprate=droprate,d_ff=d_ff).to(device)
    self.generator=Generator(dmodel=dmodel,vocab=vocab_size).to(device)

  def get_pad_mask(self, seq, pad_idx):
    # seq: (B, T)
    # True = pad → mask out
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

  def get_subsequent_mask(self, seq):
    # seq: (B, T)
    L = seq.size(1)
    subsequent = torch.triu(
        torch.ones(L, L, device=seq.device), diagonal=1
    ).bool()   # upper triangular → future = True (mask)

    # shape → (1, 1, T, T)
    return subsequent.unsqueeze(0).unsqueeze(1)

  def forward(self,src,tgt):
    if src.dim() == 1:
        src = src.unsqueeze(0)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)

    src=src.to(self.device)
    tgt=tgt.to(self.device)
    src_mask=self.get_pad_mask(src,self.pad_idx)
    tgt_mask=self.get_pad_mask(tgt,self.pad_idx) | self.get_subsequent_mask(tgt)


    encoder_out=self.encoder(src,src_mask)
    decoder_out=self.decoder(tgt,encoder_out,src_mask,tgt_mask)
    gen=self.generator(decoder_out)
    return gen


# Initialize mo