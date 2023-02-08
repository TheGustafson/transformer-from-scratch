# using attention.py as a reference, write an encoder transformer block
# using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
if __name__ == "__main__":
    sample = EncoderBlock(512, 8, 0.0, 4)
    q = torch.randn((64, 10, 512))
    k = torch.randn((64, 20, 512))
    v = torch.randn((64, 20, 512))
    mask = torch.randint(0, 2, (64, 1, 1, 20))
    out = sample(v, k, q, mask)
    print(out.shape)