import torch
import torch.nn as nn
import math

class SlefAttention(nn.Modual):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,Q,K,V,mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out = torch.matmul(attn,V)
        
        return out,attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0 , 'WRONG1'
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
    
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k - nn.Linear(d_model,d_model)
        self.W_v - nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model)
        
        self.attention = SlefAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,q,k,v,mask=None): 
        batch_size = q.size(0)
        Q = self.W_q(q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        K = self.W_k(k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V = self.W_v(v).vies(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)

        out, attn = self.attention(Q,K,V,mask)
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads*self.d_k)
        out = self.fc(out)
        out = self.dropout(out)
        return self.norm(out+q),attn
            

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,x):
        out = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        return self.norm(out+x)
    
class EncoderLayer(nn.Modele):
    def __init__(self,d_module,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_module,n_heads,dropout)
        self.fnn = FeedForward(d_module,d_ff,dropout)

    def forward(self,src,src_mask=None):
        out,_ = self.self_attn(src,src,src,src_mask)
        out = self.ffn(out)
        return out

class DecoderLayer(nn.module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self__attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.cross_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.ffn = FeedForward(d_model,d_ff,dropout)

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        out,_ = self.self__attn(tgt,tgt,tgt,tgt_mask)
        out,_ = self.cross_attn(out,memory,memory,memory_mask)
        out = self.ffn(out)
        return out
    
class PositionalEnodering(nn.Modele):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

class Encoder(nn.Module):

          
    
    





        