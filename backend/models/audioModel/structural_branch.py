import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

from preprocessing.audio_process import CFG



# -------------------------
# Structural Branch (Transformer) - SSL pretrain
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class StructuralBranch(nn.Module):
    """
    Transformer-based encoder for mel-spectrogram sequences.
    Input: mel (n_mels x time) -> transposed to (B, T, n_mels)
    """
    def __init__(self, n_mels=CFG["mel_n_mels"], d_model=CFG["transformer_d_model"], 
                 nhead=CFG["transformer_nhead"], num_layers=CFG["transformer_layers"], dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=1000)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                                    dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # SSL components
        self.mask_token = nn.Parameter(torch.randn(d_model))
        self.reconstruction_head = nn.Linear(d_model, n_mels)
        # classification embedding head
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, d_model//2), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model//2, CFG["embedding_dim"])
        )
    
    def forward(self, mel, mask_ratio=0.0, do_reconstruct=False):
        x = mel.transpose(1,2)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        if mask_ratio > 0:
            x = self.apply_masking(x, mask_ratio)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        emb = self.classifier_head(pooled)
        if do_reconstruct:
            recon = self.reconstruction_head(x)
            return emb, recon
        return emb
    
    def apply_masking(self, x, mask_ratio):
        B, T, D = x.shape
        num_mask = max(1, int(T * mask_ratio))
        mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
        for i in range(B):
            idx = torch.randperm(T)[:num_mask]
            mask[i, idx] = True
        x[mask] = self.mask_token
        return x

print("Structural Branch (Transformer) defined successfully!")
