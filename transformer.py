import torch
import torch.nn as nn
import math

class SlefAttention(nn.Modual):
    def __init__(self,drop=0.1):
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


        