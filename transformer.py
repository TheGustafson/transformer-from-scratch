# Using encoder and decoder files as a reference, write a transformer class
# using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
from attention import Attention

class Transformer(nn.module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cpu"):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, forward_expansion, heads, dropout, device)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, forward_expansion, heads, dropout, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

if __name__ == "__main__":
    sample = Transformer(100, 100, 0, 0)
    src = torch.randint(1, 100, (64, 32))
    trg = torch.randint(1, 100, (64, 32))
    out = sample(src, trg)
    print(out.shape)
    