# using encoder.py as a reference, write a decoder transformer block
# using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm1(attention + x))
        attention = self.attention(value, key, query, src_mask)
        out = self.dropout(self.norm2(attention + query))
        forward = self.feed_forward(out)
        out = self.dropout(self.norm3(forward + out))
        return out

if __name__ == "__main__":
    sample = DecoderBlock(512, 8, 0.0, 4)
    q = torch.randn((64, 10, 512))
    k = torch.randn((64, 20, 512))
    v = torch.randn((64, 20, 512))
    src_mask = torch.randint(0, 2, (64, 1, 1, 20))
    trg_mask = torch.randint(0, 2, (64, 1, 10, 10))
    out = sample(q, v, k, src_mask, trg_mask)
    print(out.shape)
    